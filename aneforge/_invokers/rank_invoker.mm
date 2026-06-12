// ane_rank_numerics_probe.mm
//
// Numerics + I/O-introspection probe for the rank-family ANECIR layers
// (Sort, TopK, ArgMinMax, GlobalArgMinMax) authored directly as netplists.
//
// Unlike ane_unreachable_layer_probe (which zero-fills and reports only
// success/timing) and ane_sdpa_fused_invoke_probe (which assumes fp16
// output and a fixed logical layout), this probe:
//   * loads named fp16 inputs into LiveInput surfaces (by Symbol),
//   * evaluates once,
//   * dumps the FULL LiveInputList / LiveOutputList dictionaries as JSON
//     (so we can see Symbol, Width/Height/Channels/Batch, strides, and any
//     dtype hint the runtime reports for index outputs), and
//   * writes the RAW surface bytes for each output (no dtype assumption),
//     so the caller can decode fp16 values vs integer indices itself.
//
// Build (aneforge/_bridges compiles this once into a per-machine binary cache):
//   xcrun clang++ -O2 -Wall -Wextra -fobjc-arc -std=gnu++17 \
//     -framework Foundation -framework IOSurface \
//     aneforge/_invokers/rank_invoker.mm \
//     -o <cache>/ane_rank_invoker
//
// Invocation:
//   ane_invoke_probe_rank --net-plist net.plist --weights weights.0 \
//     --input x=in_x.f16 --output-raw y=out_y.bin --output-bytes 4096

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
#include <map>
#include <string>
#include <vector>

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

#include <string>
static std::string jesc(const char *s) {
  std::string o;
  if (!s) return o;
  for (size_t i = 0; s[i] && i < 1024; i++) {
    unsigned char c = (unsigned char)s[i];
    if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back((char)c); }
    else if (c == '\n') o += "\\n";
    else if (c == '\r') o += "\\r";
    else if (c == '\t') o += "\\t";
    else if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
    else o.push_back((char)c);
  }
  return o;
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

// fp16 inputs are copied densely into the surface using the runtime's
// reported strides, matching ane_sdpa_fused_invoke_probe layout.
static size_t logical_offset_for(NSDictionary *info,
                                 NSUInteger b, NSUInteger c,
                                 NSUInteger d, NSUInteger h, NSUInteger w) {
  size_t elem = sizeof(_Float16);
  size_t batch_stride = uint_info(info, @"BatchStride", 65536);
  size_t depth_stride = uint_info(info, @"DepthStride", batch_stride);
  size_t plane_stride = uint_info(info, @"PlaneStride", depth_stride);
  size_t row_stride = uint_info(info, @"RowStride", plane_stride);
  return b * batch_stride + d * depth_stride + c * plane_stride + h * row_stride + w * elem;
}

static bool fill_input_from_payload(IOSurfaceRef surface, NSDictionary *info, NSData *data) {
  size_t elem = sizeof(_Float16);
  NSUInteger batches = uint_info(info, @"Batches", 1);
  NSUInteger channels = uint_info(info, @"Channels", 1);
  NSUInteger depth = uint_info(info, @"Depth", 1);
  NSUInteger height = uint_info(info, @"Height", 1);
  NSUInteger width = uint_info(info, @"Width", 1);
  NSUInteger logical_count = batches * channels * depth * height * width;
  size_t required = (size_t)logical_count * elem;
  if ([data length] < required) {
    fprintf(stderr, "input payload too small: have %lu need %zu\n",
            (unsigned long)[data length], required);
    return false;
  }
  IOSurfaceLock(surface, 0, NULL);
  size_t bytes = IOSurfaceGetAllocSize(surface);
  uint8_t *dst = (uint8_t *)IOSurfaceGetBaseAddress(surface);
  const uint8_t *src = (const uint8_t *)[data bytes];
  memset(dst, 0, bytes);
  NSUInteger li = 0;
  for (NSUInteger b = 0; b < batches; b++)
    for (NSUInteger c = 0; c < channels; c++)
      for (NSUInteger d = 0; d < depth; d++)
        for (NSUInteger h = 0; h < height; h++)
          for (NSUInteger w = 0; w < width; w++, li++) {
            size_t off = logical_offset_for(info, b, c, d, h, w);
            if (off + elem <= bytes) memcpy(dst + off, src + li * elem, elem);
          }
  IOSurfaceUnlock(surface, 0, NULL);
  return true;
}

// Write the RAW first `nbytes` of the surface (no dtype assumption).
static bool write_raw_surface(IOSurfaceRef surface, NSString *path, size_t nbytes) {
  IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
  size_t avail = IOSurfaceGetAllocSize(surface);
  if (nbytes == 0 || nbytes > avail) nbytes = avail;
  NSData *d = [NSData dataWithBytes:IOSurfaceGetBaseAddress(surface) length:nbytes];
  IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
  return [d writeToFile:path atomically:YES];
}

static NSArray *indices_for_count(NSUInteger count) {
  NSMutableArray *out = [NSMutableArray arrayWithCapacity:count];
  for (NSUInteger i = 0; i < count; i++) [out addObject:@(i)];
  return out;
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

struct NamedFile { std::string symbol; std::string path; };

struct Args {
  std::string netplist;
  std::vector<std::string> weights;
  std::vector<NamedFile> inputs;
  std::vector<NamedFile> outputs;     // raw byte dumps
  size_t output_bytes = 0;            // 0 => whole surface
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
      if (i + 1 >= argc) { fprintf(stderr, "missing arg for %s\n", flag.c_str()); exit(2); }
      return argv[++i];
    };
    if (flag == "--net-plist") out.netplist = next();
    else if (flag == "--weights") out.weights.push_back(next());
    else if (flag == "--input") {
      NamedFile nf; if (!parse_named(next(), nf)) { fprintf(stderr, "bad --input\n"); return 2; }
      out.inputs.push_back(nf);
    } else if (flag == "--output-raw") {
      NamedFile nf; if (!parse_named(next(), nf)) { fprintf(stderr, "bad --output-raw\n"); return 2; }
      out.outputs.push_back(nf);
    } else if (flag == "--output-bytes") out.output_bytes = (size_t)atoll(next());
    else if (flag == "--qos") out.qos = (unsigned int)atoi(next());
    else { fprintf(stderr, "unknown flag: %s\n", flag.c_str()); return 2; }
  }
  if (out.netplist.empty()) { fprintf(stderr, "--net-plist required\n"); return 2; }
  return 0;
}

// Serialize a LiveIO dict to compact JSON (numbers + Symbol + OutputType).
static std::string io_json(NSDictionary *info) {
  NSError *e = nil;
  // Filter to plist-JSON-safe scalar entries.
  NSMutableDictionary *flat = [NSMutableDictionary dictionary];
  for (NSString *k in info) {
    id v = info[k];
    if ([v isKindOfClass:[NSString class]] || [v isKindOfClass:[NSNumber class]])
      flat[k] = v;
  }
  NSData *d = [NSJSONSerialization dataWithJSONObject:flat options:0 error:&e];
  if (!d) return "{}";
  return std::string((const char *)d.bytes, d.length);
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  Args args;
  if (int rc = parse_args(argc, argv, args)) return rc;

  @autoreleasepool {
    void *handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW);
    if (!handle) { printf("{\"status\":\"dlopen_failed\"}\n"); return 1; }

    NSData *net = [NSData dataWithContentsOfFile:
                       [NSString stringWithUTF8String:args.netplist.c_str()]];
    if (!net) { printf("{\"status\":\"netplist_read_failed\"}\n"); return 1; }
    NSMutableDictionary *weights = [NSMutableDictionary dictionary];
    for (const auto &wpath : args.weights) {
      NSString *p = [NSString stringWithUTF8String:wpath.c_str()];
      NSData *d = [NSData dataWithContentsOfFile:p];
      if (!d) { printf("{\"status\":\"weights_read_failed\"}\n"); return 1; }
      weights[[p lastPathComponent]] = @{@"offset": @0, @"data": d};
    }

    Class desc_cls = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class model_cls = NSClassFromString(@"_ANEInMemoryModel");
    id desc = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
        desc_cls, @selector(modelWithNetworkDescription:weights:optionsPlist:),
        net, weights, nil);
    if (!desc) { printf("{\"status\":\"descriptor_creation_failed\"}\n"); return 3; }
    id model = ((id (*)(Class, SEL, id))objc_msgSend)(
        model_cls, @selector(inMemoryModelWithDescriptor:), desc);
    if (!model) { printf("{\"status\":\"model_creation_failed\"}\n"); return 3; }

    NSString *local_path = call_obj0(model, "localModelPath");
    if ([local_path isKindOfClass:[NSString class]] && [local_path length] > 0) {
      [[NSFileManager defaultManager] createDirectoryAtPath:local_path
                                withIntermediateDirectories:YES attributes:nil error:nil];
      [net writeToFile:[local_path stringByAppendingPathComponent:@"net.plist"] atomically:YES];
      for (NSString *wname in weights)
        [weights[wname][@"data"] writeToFile:
            [local_path stringByAppendingPathComponent:wname] atomically:YES];
    }

    NSError *err = nil;
    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), args.qos, @{}, &err);
    if (!ok) {
      printf("{\"status\":\"compile_failed\",\"error\":\"%s\"}\n",
             jesc(err ? err.description.UTF8String : "(nil)").c_str());
      return 4;
    }
    ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), args.qos, @{}, &err);
    if (!ok) {
      printf("{\"status\":\"load_failed\",\"error\":\"%s\"}\n",
             jesc(err ? err.description.UTF8String : "(nil)").c_str());
      return 5;
    }

    NSArray *live_inputs = live_io_list(model, @"LiveInputList");
    NSArray *live_outputs = live_io_list(model, @"LiveOutputList");

    std::map<std::string, std::string> in_by_sym, out_by_sym;
    for (const auto &nf : args.inputs)  in_by_sym[nf.symbol]  = nf.path;
    for (const auto &nf : args.outputs) out_by_sym[nf.symbol] = nf.path;

    Class surface_cls = NSClassFromString(@"_ANEIOSurfaceObject");
    Class request_cls = NSClassFromString(@"_ANERequest");

    NSMutableArray *input_surfaces = [NSMutableArray array];
    NSMutableArray *wrapped_inputs = [NSMutableArray array];
    for (NSUInteger i = 0; i < [live_inputs count]; i++) {
      NSDictionary *info = live_inputs[i];
      NSString *symbol = info[@"Symbol"];
      IOSurfaceRef surface = make_surface(bytes_for_live_io(info));
      auto it = in_by_sym.find(symbol ? [symbol UTF8String] : "");
      if (it != in_by_sym.end()) {
        NSData *d = [NSData dataWithContentsOfFile:
                        [NSString stringWithUTF8String:it->second.c_str()]];
        if (!d || !fill_input_from_payload(surface, info, d)) {
          printf("{\"status\":\"input_load_failed\",\"symbol\":\"%s\"}\n",
                 symbol ? symbol.UTF8String : "?");
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

    NSMutableArray *output_surfaces = [NSMutableArray array];
    NSMutableArray *wrapped_outputs = [NSMutableArray array];
    for (NSUInteger i = 0; i < [live_outputs count]; i++) {
      NSDictionary *info = live_outputs[i];
      IOSurfaceRef surface = make_surface(bytes_for_live_io(info));
      IOSurfaceLock(surface, 0, NULL);
      memset(IOSurfaceGetBaseAddress(surface), 0, IOSurfaceGetAllocSize(surface));
      IOSurfaceUnlock(surface, 0, NULL);
      [output_surfaces addObject:[NSValue valueWithPointer:surface]];
      [wrapped_outputs addObject:((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                                     surface_cls, @selector(objectWithIOSurface:), surface)];
    }

    id request = ((id (*)(Class, SEL, id, id, id, id, id))objc_msgSend)(
        request_cls,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:),
        wrapped_inputs, indices_for_count([wrapped_inputs count]),
        wrapped_outputs, indices_for_count([wrapped_outputs count]), @0);

    BOOL mapped = ((BOOL (*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
        model, @selector(mapIOSurfacesWithRequest:cacheInference:error:),
        request, YES, &err);
    if (!mapped) {
      printf("{\"status\":\"map_failed\",\"error\":\"%s\"}\n",
             jesc(err ? err.description.UTF8String : "(nil)").c_str());
      return 7;
    }

    uint64_t e0 = now_ns();
    BOOL e_ok = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        model, @selector(evaluateWithQoS:options:request:error:),
        args.qos, @{}, request, &err);
    uint64_t e1 = now_ns();
    if (!e_ok) {
      printf("{\"status\":\"eval_failed\",\"error\":\"%s\"}\n",
             jesc(err ? err.description.UTF8String : "(nil)").c_str());
      return 8;
    }

    // Write raw output bytes.
    for (NSUInteger i = 0; i < [live_outputs count]; i++) {
      NSDictionary *info = live_outputs[i];
      NSString *symbol = info[@"Symbol"];
      auto it = out_by_sym.find(symbol ? [symbol UTF8String] : "");
      if (it == out_by_sym.end()) continue;
      IOSurfaceRef surface = (IOSurfaceRef)[output_surfaces[i] pointerValue];
      write_raw_surface(surface, [NSString stringWithUTF8String:it->second.c_str()],
                        args.output_bytes);
    }

    // Emit introspection JSON.
    std::string ins = "[";
    for (NSUInteger i = 0; i < [live_inputs count]; i++) {
      if (i) ins += ",";
      ins += io_json(live_inputs[i]);
    }
    ins += "]";
    std::string outs = "[";
    for (NSUInteger i = 0; i < [live_outputs count]; i++) {
      if (i) outs += ",";
      outs += io_json(live_outputs[i]);
    }
    outs += "]";

    printf("{\"status\":\"ok\",\"eval_us\":%.3f,\"live_inputs\":%s,\"live_outputs\":%s}\n",
           (double)(e1 - e0) / 1e3, ins.c_str(), outs.c_str());

    SEL unmap_sel = @selector(unmapIOSurfacesWithRequest:);
    if ([model respondsToSelector:unmap_sel])
      ((void (*)(id, SEL, id))objc_msgSend)(model, unmap_sel, request);
    for (NSValue *v : input_surfaces)  CFRelease((IOSurfaceRef)[v pointerValue]);
    for (NSValue *v : output_surfaces) CFRelease((IOSurfaceRef)[v pointerValue]);
  }
  return 0;
}
