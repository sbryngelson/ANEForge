// ane_layer_numerics_probe.mm
//
// Numerics-focused probe for the 9 layer kinds Agent #53 reachability-verified
// (Dropout, Flatten, Shape, L2Normalization, ScaledElementWise, SpaceToBatch,
// BatchToSpace, SpaceToChannel, ChannelToSpace).
//
// Generalizes ane_sdpa_fused_invoke_probe.mm to any netplist + weights pair
// while keeping the SDPA-style --input SYM=PATH / --output SYM=PATH binding.
// Output payloads are written in *logical* (B, C, D, H, W) order using the
// LiveOutput stride metadata, so the caller can compare bytes to a numpy
// reference directly.
//
// Build (aneforge/_bridges compiles this once into a per-machine binary cache):
//   xcrun clang++ -O2 -Wall -Wextra -fobjc-arc -std=gnu++17 \
//     -framework Foundation -framework IOSurface \
//     aneforge/_invokers/layer_invoker.mm \
//     -o <cache>/ane_layer_invoker
//
// Invocation:
//   ane_layer_numerics_probe \
//     --net-plist /tmp/case/net.plist \
//     --weights   /tmp/case/weights.0 \
//     --input     x=/tmp/case/in_x.f16 \
//     --output    y=/tmp/case/out_y.f16 \
//     [--output-int32 y_i=/tmp/case/out_y.i32]  # Shape layer outputs i32/i16
//     [--repeats N] [--warmup N] [--qos N] [--dtype-out fp16|int32|int16]
//
// Output: one JSON line to stdout summarizing status + sizes + LiveOutput
// info; output payload bytes go to --output PATH (logical-order, dtype-
// matched).
//
// Also dumps the LiveOutput dict (BatchStride / Channels / Width / etc.) to
// the JSON output so the driver can reconstruct the layout exactly.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static id call_obj0(id obj, const char *sel_name) {
  SEL sel = sel_registerName(sel_name);
  if (!obj || ![obj respondsToSelector:sel]) return nil;
  IMP imp = [obj methodForSelector:sel];
  id (*fn)(id, SEL) = (id (*)(id, SEL))imp;
  return fn(obj, sel);
}

static IOSurfaceRef make_surface(size_t bytes) {
  if (bytes < 1) bytes = 1;
  NSDictionary *props = @{
    (id)kIOSurfaceWidth : @(bytes),
    (id)kIOSurfaceHeight : @1,
    (id)kIOSurfaceBytesPerElement : @1,
    (id)kIOSurfaceBytesPerRow : @(bytes),
    (id)kIOSurfaceAllocSize : @(bytes),
    (id)kIOSurfacePixelFormat : @0,
  };
  IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
  if (!s) {
    printf("{\"status\":\"surface_alloc_failed\",\"bytes\":%zu}\n", bytes);
    exit(11);
  }
  return s;
}

static NSUInteger uint_info(NSDictionary *info, NSString *key, NSUInteger fallback) {
  NSNumber *v = info[key];
  if ([v respondsToSelector:@selector(unsignedLongLongValue)]) {
    NSUInteger parsed = (NSUInteger)[v unsignedLongLongValue];
    if (parsed > 0) return parsed;
  }
  return fallback;
}

static size_t bytes_for_live_io(NSDictionary *info) {
  NSNumber *batch_stride = info[@"BatchStride"];
  NSNumber *batches = info[@"Batches"];
  if (batch_stride && batches) {
    size_t bytes = (size_t)[batch_stride unsignedLongLongValue] *
                   (size_t)[batches unsignedLongLongValue];
    if (bytes > 0) return bytes;
  }
  return 65536;
}

// Get the element size in bytes for the io_type symbol in the LiveIO dict.
// Most ops are fp16; Shape may output i32 or i16.
static size_t elem_bytes_for(NSDictionary *info) {
  NSString *type = info[@"Type"];
  if ([type isKindOfClass:[NSString class]]) {
    if ([type isEqualToString:@"Int32"] || [type isEqualToString:@"UInt32"]) return 4;
    if ([type isEqualToString:@"Int16"] || [type isEqualToString:@"UInt16"]) return 2;
    if ([type isEqualToString:@"Int8"]  || [type isEqualToString:@"UInt8"])  return 1;
    if ([type isEqualToString:@"Float32"]) return 4;
  }
  return 2;  // Float16 default
}

static size_t logical_offset_for(NSDictionary *info,
                                 NSUInteger b, NSUInteger c,
                                 NSUInteger d, NSUInteger h, NSUInteger w,
                                 size_t elem) {
  size_t batch_stride = uint_info(info, @"BatchStride", 65536);
  size_t depth_stride = uint_info(info, @"DepthStride", batch_stride);
  size_t plane_stride = uint_info(info, @"PlaneStride", depth_stride);
  size_t row_stride = uint_info(info, @"RowStride", plane_stride);
  return b * batch_stride + d * depth_stride + c * plane_stride + h * row_stride + w * elem;
}

static bool fill_input_from_payload(IOSurfaceRef surface, NSDictionary *info, NSData *data) {
  size_t elem = elem_bytes_for(info);
  NSUInteger batches = uint_info(info, @"Batches", 1);
  NSUInteger channels = uint_info(info, @"Channels", 1);
  NSUInteger depth = uint_info(info, @"Depth", 1);
  NSUInteger height = uint_info(info, @"Height", 1);
  NSUInteger width = uint_info(info, @"Width", 1);
  NSUInteger logical_count = batches * channels * depth * height * width;
  size_t required = (size_t)logical_count * elem;
  if ([data length] < required) {
    fprintf(stderr, "input payload too small: have %lu need %zu (B=%lu C=%lu D=%lu H=%lu W=%lu elem=%zu)\n",
            (unsigned long)[data length], required,
            (unsigned long)batches, (unsigned long)channels, (unsigned long)depth,
            (unsigned long)height, (unsigned long)width, elem);
    return false;
  }
  IOSurfaceLock(surface, 0, NULL);
  size_t bytes = IOSurfaceGetAllocSize(surface);
  uint8_t *dst = (uint8_t *)IOSurfaceGetBaseAddress(surface);
  const uint8_t *src = (const uint8_t *)[data bytes];
  memset(dst, 0, bytes);
  NSUInteger li = 0;
  for (NSUInteger b = 0; b < batches; b++) {
    for (NSUInteger c = 0; c < channels; c++) {
      for (NSUInteger d = 0; d < depth; d++) {
        for (NSUInteger h = 0; h < height; h++) {
          for (NSUInteger w = 0; w < width; w++, li++) {
            size_t off = logical_offset_for(info, b, c, d, h, w, elem);
            if (off + elem <= bytes) memcpy(dst + off, src + li * elem, elem);
          }
        }
      }
    }
  }
  IOSurfaceUnlock(surface, 0, NULL);
  return true;
}

static bool write_output_payload(IOSurfaceRef surface, NSDictionary *info, NSString *path) {
  size_t elem = elem_bytes_for(info);
  NSUInteger batches = uint_info(info, @"Batches", 1);
  NSUInteger channels = uint_info(info, @"Channels", 1);
  NSUInteger depth = uint_info(info, @"Depth", 1);
  NSUInteger height = uint_info(info, @"Height", 1);
  NSUInteger width = uint_info(info, @"Width", 1);
  NSUInteger logical_count = batches * channels * depth * height * width;
  NSMutableData *data = [NSMutableData dataWithLength:(NSUInteger)((size_t)logical_count * elem)];
  IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
  size_t bytes = IOSurfaceGetAllocSize(surface);
  const uint8_t *src = (const uint8_t *)IOSurfaceGetBaseAddress(surface);
  uint8_t *dst = (uint8_t *)[data mutableBytes];
  NSUInteger li = 0;
  for (NSUInteger b = 0; b < batches; b++) {
    for (NSUInteger c = 0; c < channels; c++) {
      for (NSUInteger d = 0; d < depth; d++) {
        for (NSUInteger h = 0; h < height; h++) {
          for (NSUInteger w = 0; w < width; w++, li++) {
            size_t off = logical_offset_for(info, b, c, d, h, w, elem);
            if (off + elem <= bytes) memcpy(dst + li * elem, src + off, elem);
          }
        }
      }
    }
  }
  IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
  return [data writeToFile:path atomically:YES];
}

// Dump the raw IOSurface to a separate file (full surface bytes, no
// reordering) so the caller can inspect strided layout if needed.
static bool dump_raw_surface(IOSurfaceRef surface, NSString *path) {
  IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
  size_t bytes = IOSurfaceGetAllocSize(surface);
  NSData *d = [NSData dataWithBytes:IOSurfaceGetBaseAddress(surface) length:bytes];
  IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
  return [d writeToFile:path atomically:YES];
}

static NSArray *indices_for_count(NSUInteger count) {
  NSMutableArray *out = [NSMutableArray arrayWithCapacity:count];
  for (NSUInteger i = 0; i < count; i++) [out addObject:@(i)];
  return out;
}

// -------------------------------------------------------------------------
// CLI parser.

struct NamedFile {
  std::string symbol;
  std::string path;
};

struct Args {
  std::string netplist;
  std::vector<std::string> weights;
  std::vector<NamedFile> inputs;
  std::vector<NamedFile> outputs;
  std::vector<NamedFile> raw_dumps;  // optional --raw-output SYM=PATH
  int repeats = 1;
  int warmup = 0;
  unsigned int qos = 33;
};

static bool parse_named(const std::string &arg, NamedFile &out) {
  auto pos = arg.find('=');
  if (pos == std::string::npos) return false;
  out.symbol = arg.substr(0, pos);
  out.path = arg.substr(pos + 1);
  return true;
}

static int parse_args(int argc, char **argv, Args &out) {
  for (int i = 1; i < argc; i++) {
    std::string flag = argv[i];
    auto next = [&]() -> const char * {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing argument for %s\n", flag.c_str());
        exit(2);
      }
      return argv[++i];
    };
    if (flag == "--net-plist") out.netplist = next();
    else if (flag == "--weights") out.weights.push_back(next());
    else if (flag == "--input") {
      NamedFile nf;
      if (!parse_named(next(), nf)) { fprintf(stderr, "bad --input\n"); return 2; }
      out.inputs.push_back(nf);
    } else if (flag == "--output") {
      NamedFile nf;
      if (!parse_named(next(), nf)) { fprintf(stderr, "bad --output\n"); return 2; }
      out.outputs.push_back(nf);
    } else if (flag == "--raw-output") {
      NamedFile nf;
      if (!parse_named(next(), nf)) { fprintf(stderr, "bad --raw-output\n"); return 2; }
      out.raw_dumps.push_back(nf);
    } else if (flag == "--repeats") out.repeats = atoi(next());
    else if (flag == "--warmup") out.warmup = atoi(next());
    else if (flag == "--qos") out.qos = (unsigned int)atoi(next());
    else if (flag == "--help" || flag == "-h") {
      printf("usage: %s --net-plist PATH --weights PATH [--weights PATH...] "
             "[--input SYM=PATH...] [--output SYM=PATH...] "
             "[--raw-output SYM=PATH...] "
             "[--repeats N] [--warmup N] [--qos N]\n", argv[0]);
      exit(0);
    } else {
      fprintf(stderr, "unknown flag: %s\n", flag.c_str());
      return 2;
    }
  }
  if (out.netplist.empty()) {
    fprintf(stderr, "--net-plist required\n");
    return 2;
  }
  return 0;
}

static NSArray *live_io_list(id model, NSString *key) {
  id attrs = call_obj0(model, "modelAttributes");
  if (![attrs isKindOfClass:[NSDictionary class]]) return @[];
  NSArray *networks = ((NSDictionary *)attrs)[@"NetworkStatusList"];
  if (![networks isKindOfClass:[NSArray class]] || [networks count] == 0) return @[];
  NSDictionary *network = networks[0];
  if (![network isKindOfClass:[NSDictionary class]]) return @[];
  NSArray *list = network[key];
  if (![list isKindOfClass:[NSArray class]]) return @[];
  return list;
}

// Pull a small JSON-ish dump of the live-io entry (only scalar keys).
static std::string json_for_live_io(NSDictionary *info) {
  std::string out = "{";
  bool first = true;
  for (NSString *key in info) {
    id v = info[key];
    if (![v isKindOfClass:[NSNumber class]] && ![v isKindOfClass:[NSString class]]) continue;
    if (!first) out += ",";
    first = false;
    out += "\"";
    out += [key UTF8String];
    out += "\":";
    if ([v isKindOfClass:[NSNumber class]]) {
      out += [[v stringValue] UTF8String];
    } else {
      out += "\"";
      const char *cs = [(NSString *)v UTF8String];
      if (cs) out += cs;
      out += "\"";
    }
  }
  out += "}";
  return out;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  Args args;
  if (int rc = parse_args(argc, argv, args)) return rc;

  @autoreleasepool {
    void *handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW);
    if (!handle) {
      fprintf(stderr, "dlopen failed: %s\n", dlerror());
      printf("{\"status\":\"dlopen_failed\"}\n");
      return 1;
    }

    NSString *netplist_path = [NSString stringWithUTF8String:args.netplist.c_str()];
    NSData *net = [NSData dataWithContentsOfFile:netplist_path];
    if (!net) {
      fprintf(stderr, "cannot read netplist %s\n", args.netplist.c_str());
      printf("{\"status\":\"netplist_read_failed\"}\n");
      return 1;
    }
    NSMutableDictionary *weights = [NSMutableDictionary dictionary];
    for (const auto &wpath : args.weights) {
      NSString *p = [NSString stringWithUTF8String:wpath.c_str()];
      NSData *d = [NSData dataWithContentsOfFile:p];
      if (!d) {
        fprintf(stderr, "cannot read weight %s\n", wpath.c_str());
        printf("{\"status\":\"weights_read_failed\"}\n");
        return 1;
      }
      NSString *name = [p lastPathComponent];
      weights[name] = @{@"offset": @0, @"data": d};
    }

    Class desc_cls = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class model_cls = NSClassFromString(@"_ANEInMemoryModel");
    SEL desc_sel = @selector(modelWithNetworkDescription:weights:optionsPlist:);

    id desc = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
        desc_cls, desc_sel, net, weights, nil);
    if (!desc) {
      printf("{\"status\":\"descriptor_creation_failed\"}\n");
      return 3;
    }
    id model = ((id (*)(Class, SEL, id))objc_msgSend)(
        model_cls, @selector(inMemoryModelWithDescriptor:), desc);
    if (!model) {
      printf("{\"status\":\"model_creation_failed\"}\n");
      return 3;
    }

    NSString *local_path = call_obj0(model, "localModelPath");
    if ([local_path isKindOfClass:[NSString class]] && [local_path length] > 0) {
      [[NSFileManager defaultManager] createDirectoryAtPath:local_path
                                withIntermediateDirectories:YES
                                                 attributes:nil
                                                      error:nil];
      [net writeToFile:[local_path stringByAppendingPathComponent:@"net.plist"] atomically:YES];
      for (NSString *wname in weights) {
        NSData *d = weights[wname][@"data"];
        [d writeToFile:[local_path stringByAppendingPathComponent:wname] atomically:YES];
      }
    }

    NSError *err = nil;
    SEL compile_sel = @selector(compileWithQoS:options:error:);
    SEL load_sel = @selector(loadWithQoS:options:error:);

    uint64_t tc0 = now_ns();
    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, compile_sel, args.qos, @{}, &err);
    uint64_t tc1 = now_ns();
    double compile_ms = (double)(tc1 - tc0) / 1e6;
    if (!ok) {
      printf("{\"status\":\"compile_failed\",\"compile_ms\":%.3f}\n", compile_ms);
      return 4;
    }

    uint64_t tl0 = now_ns();
    ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, load_sel, args.qos, @{}, &err);
    uint64_t tl1 = now_ns();
    double load_ms = (double)(tl1 - tl0) / 1e6;
    if (!ok) {
      printf("{\"status\":\"load_failed\",\"compile_ms\":%.3f,\"load_ms\":%.3f}\n",
             compile_ms, load_ms);
      return 5;
    }

    NSArray *live_inputs = live_io_list(model, @"LiveInputList");
    NSArray *live_outputs = live_io_list(model, @"LiveOutputList");

    std::map<std::string, std::string> input_path_by_symbol;
    std::map<std::string, std::string> output_path_by_symbol;
    std::map<std::string, std::string> raw_path_by_symbol;
    for (const auto &nf : args.inputs)    input_path_by_symbol[nf.symbol]  = nf.path;
    for (const auto &nf : args.outputs)   output_path_by_symbol[nf.symbol] = nf.path;
    for (const auto &nf : args.raw_dumps) raw_path_by_symbol[nf.symbol]    = nf.path;

    Class surface_cls = NSClassFromString(@"_ANEIOSurfaceObject");
    Class request_cls = NSClassFromString(@"_ANERequest");

    NSMutableArray *input_surfaces = [NSMutableArray array];
    NSMutableArray *wrapped_inputs = [NSMutableArray array];
    std::string in_io_json = "[";
    for (NSUInteger i = 0; i < [live_inputs count]; i++) {
      NSDictionary *info = live_inputs[i];
      NSString *symbol = info[@"Symbol"];
      if (i) in_io_json += ",";
      in_io_json += "{\"symbol\":\"";
      in_io_json += symbol ? [symbol UTF8String] : "?";
      in_io_json += "\",\"info\":";
      in_io_json += json_for_live_io(info);
      in_io_json += "}";
      IOSurfaceRef surface = make_surface(bytes_for_live_io(info));
      auto it = input_path_by_symbol.find([symbol UTF8String] ?: "");
      if (it != input_path_by_symbol.end()) {
        NSData *d = [NSData dataWithContentsOfFile:
                                [NSString stringWithUTF8String:it->second.c_str()]];
        if (!d || !fill_input_from_payload(surface, info, d)) {
          fprintf(stderr, "failed to load input %s from %s\n",
                  symbol.UTF8String, it->second.c_str());
          printf("{\"status\":\"input_load_failed\"}\n");
          return 6;
        }
      } else {
        IOSurfaceLock(surface, 0, NULL);
        memset(IOSurfaceGetBaseAddress(surface), 0, IOSurfaceGetAllocSize(surface));
        IOSurfaceUnlock(surface, 0, NULL);
      }
      [input_surfaces addObject:[NSValue valueWithPointer:surface]];
      [wrapped_inputs addObject:((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                                    surface_cls, @selector(objectWithIOSurface:), surface)];
    }
    in_io_json += "]";

    NSMutableArray *output_surfaces = [NSMutableArray array];
    NSMutableArray *wrapped_outputs = [NSMutableArray array];
    std::string out_io_json = "[";
    for (NSUInteger i = 0; i < [live_outputs count]; i++) {
      NSDictionary *info = live_outputs[i];
      NSString *symbol = info[@"Symbol"];
      if (i) out_io_json += ",";
      out_io_json += "{\"symbol\":\"";
      out_io_json += symbol ? [symbol UTF8String] : "?";
      out_io_json += "\",\"info\":";
      out_io_json += json_for_live_io(info);
      out_io_json += "}";
      IOSurfaceRef surface = make_surface(bytes_for_live_io(info));
      [output_surfaces addObject:[NSValue valueWithPointer:surface]];
      [wrapped_outputs addObject:((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                                     surface_cls, @selector(objectWithIOSurface:), surface)];
    }
    out_io_json += "]";

    id request = ((id (*)(Class, SEL, id, id, id, id, id))objc_msgSend)(
        request_cls,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:),
        wrapped_inputs, indices_for_count([wrapped_inputs count]),
        wrapped_outputs, indices_for_count([wrapped_outputs count]), @0);

    NSError *map_err = nil;
    BOOL mapped = ((BOOL (*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
        model, @selector(mapIOSurfacesWithRequest:cacheInference:error:),
        request, YES, &map_err);
    if (!mapped) {
      fprintf(stderr, "map failed: %s\n", map_err ? map_err.description.UTF8String : "(nil)");
      printf("{\"status\":\"map_failed\"}\n");
      return 7;
    }

    SEL eval_sel = @selector(evaluateWithQoS:options:request:error:);
    BOOL (*eval_fn)(id, SEL, unsigned int, id, id, NSError **) =
        (BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend;

    for (int i = 0; i < args.warmup; i++) {
      eval_fn(model, eval_sel, args.qos, @{}, request, &err);
    }

    std::vector<double> eval_us;
    eval_us.reserve(args.repeats);
    for (int i = 0; i < args.repeats; i++) {
      uint64_t a = now_ns();
      BOOL e = eval_fn(model, eval_sel, args.qos, @{}, request, &err);
      uint64_t b = now_ns();
      if (!e) {
        printf("{\"status\":\"eval_failed\",\"compile_ms\":%.3f,\"load_ms\":%.3f}\n",
               compile_ms, load_ms);
        return 8;
      }
      eval_us.push_back((double)(b - a) / 1e3);
    }

    // Write outputs.
    for (NSUInteger i = 0; i < [live_outputs count]; i++) {
      NSDictionary *info = live_outputs[i];
      NSString *symbol = info[@"Symbol"];
      const char *sym_c = symbol ? [symbol UTF8String] : "";
      IOSurfaceRef surface = (IOSurfaceRef)[output_surfaces[i] pointerValue];
      auto it = output_path_by_symbol.find(sym_c);
      if (it != output_path_by_symbol.end()) {
        NSString *p = [NSString stringWithUTF8String:it->second.c_str()];
        if (!write_output_payload(surface, info, p)) {
          fprintf(stderr, "failed to write output %s -> %s\n", sym_c, it->second.c_str());
          printf("{\"status\":\"output_write_failed\"}\n");
          return 9;
        }
      }
      auto rit = raw_path_by_symbol.find(sym_c);
      if (rit != raw_path_by_symbol.end()) {
        NSString *p = [NSString stringWithUTF8String:rit->second.c_str()];
        dump_raw_surface(surface, p);
      }
    }

    std::sort(eval_us.begin(), eval_us.end());
    double p50 = eval_us.empty() ? 0.0 : eval_us[eval_us.size() / 2];

    printf("{\"status\":\"ok\","
           "\"compile_ms\":%.3f,"
           "\"load_ms\":%.3f,"
           "\"eval_p50_us\":%.3f,"
           "\"repeats\":%d,"
           "\"live_inputs\":%s,"
           "\"live_outputs\":%s}\n",
           compile_ms, load_ms, p50, args.repeats,
           in_io_json.c_str(), out_io_json.c_str());

    SEL unmap_sel = @selector(unmapIOSurfacesWithRequest:);
    if ([model respondsToSelector:unmap_sel]) {
      ((void (*)(id, SEL, id))objc_msgSend)(model, unmap_sel, request);
    }
    SEL unload_sel = @selector(unloadWithQoS:error:);
    if ([model respondsToSelector:unload_sel]) {
      BOOL (*unload)(id, SEL, unsigned int, NSError **) =
          (BOOL (*)(id, SEL, unsigned int, NSError **))[model methodForSelector:unload_sel];
      unload(model, unload_sel, args.qos, &err);
    }
    for (NSValue *v : input_surfaces)  CFRelease((IOSurfaceRef)[v pointerValue]);
    for (NSValue *v : output_surfaces) CFRelease((IOSurfaceRef)[v pointerValue]);
  }
  return 0;
}
