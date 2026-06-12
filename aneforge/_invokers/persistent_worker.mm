// ane_persistent_worker.mm
//
// A2 throughput path: a PERSISTENT Path-A worker.
//
// The one-shot probe (ane_sdpa_fused_invoke_probe.mm) pays the full
// descriptor -> compileWithQoS -> loadWithQoS -> mapIOSurfaces tax on EVERY
// call, then evaluates once and exits.  For repeated calls of a fixed-shape
// netplist program (e.g. ag.sdpa at one shape inside a decode loop) that tax
// dominates.  This worker pays it ONCE at startup, then services many evals.
//
// Lifecycle:
//   1. argv: --net-plist PATH --weights PATH... --input SYM... --output SYM...
//      (input/output symbols give the wire order; the byte layout is taken
//      from the model's LiveInput/LiveOutput stride metadata, exactly like the
//      one-shot probe).
//   2. Build descriptor + model, compile, load, build IOSurfaces + request,
//      map ONCE.  Emit one JSON "ready" line on stdout (compile_ms / load_ms /
//      per-input + per-output logical element counts) so the Python driver
//      knows how many fp16 elements to send / receive per call.
//   3. Loop on a binary protocol over stdin/stdout:
//        request : a 4-byte little-endian uint32 count N (>=1), then N input
//                  sets packed back to back.  Each input set is, for each input
//                  symbol in argv order, the raw fp16 bytes (logical_count * 2).
//        reply   : N output sets packed back to back.  Each output set is, for
//                  each output symbol in argv order, the raw fp16 bytes
//                  (logical_count * 2).
//      Each input set triggers exactly one evaluateWithQoS into the already
//      mapped surfaces; the N evals share ONE pipe round-trip.  This is the key
//      win for per-row-tiled ops (topk/sort): the C row-tiles that used to be C
//      separate requests now ride in a single request (N=C), removing (C-1)
//      pipe round-trips.  N=1 reproduces the original single-eval behaviour
//      bit-for-bit.  EOF on stdin (before the count word) -> clean shutdown.
//
// The surfaces + request are reused across calls (no remap), so steady-state
// per call cost is host->surface memcpy + evaluateWithQoS + surface->host
// memcpy.  This is the load-once-eval-many win the one-shot probe's timings
// already pointed at.
//
// Build (aneforge/_bridges compiles this once into a per-machine binary cache,
// at a UNIQUE path so it never clobbers the one-shot invoker):
//   xcrun clang++ -O2 -Wall -Wextra -fobjc-arc -std=gnu++17 \
//     -framework Foundation -framework IOSurface \
//     aneforge/_invokers/persistent_worker.mm \
//     -o <cache>/ane_persistent_worker

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <map>
#include <string>
#include <vector>

// JSON-escape a C string for safe embedding in status lines (mirrors the
// helper in rank_invoker.mm).
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
    fflush(stdout);
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

// Logical element count (b*c*d*h*w) for a LiveInput/Output descriptor.
static NSUInteger logical_count_for(NSDictionary *info) {
  return uint_info(info, @"Batches", 1) * uint_info(info, @"Channels", 1) *
         uint_info(info, @"Depth", 1) * uint_info(info, @"Height", 1) *
         uint_info(info, @"Width", 1);
}

// Is the logical (B,C,D,H,W) layout DENSE in the surface, i.e. each stride
// equals the product of the inner dims so a single contiguous memcpy of
// logical_count*elem bytes maps 1:1 (no gaps)?  When true we skip the
// element-by-element strided loop (the dominant per-call cost for big tensors
// like SDPA's 16K-element Q/K/V) and do one bulk memcpy.
static bool layout_is_dense(NSDictionary *info, size_t *out_bytes) {
  size_t elem = sizeof(_Float16);
  NSUInteger B = uint_info(info, @"Batches", 1), C = uint_info(info, @"Channels", 1);
  NSUInteger D = uint_info(info, @"Depth", 1), H = uint_info(info, @"Height", 1);
  NSUInteger W = uint_info(info, @"Width", 1);
  size_t row = elem * (size_t)W;
  size_t plane = row * (size_t)H;
  size_t depth = plane * (size_t)C;     // channels live inside one "depth" slab
  size_t batch = depth * (size_t)D;
  size_t row_s = uint_info(info, @"RowStride", row);
  size_t plane_s = uint_info(info, @"PlaneStride", plane);
  size_t depth_s = uint_info(info, @"DepthStride", depth);
  size_t batch_s = uint_info(info, @"BatchStride", batch);
  if (out_bytes) *out_bytes = batch * (size_t)B;
  // logical_offset_for nests as b, d(depth), c(plane), h(row), w; the contiguous
  // wire buffer nests as b, c, d, h, w. Those memory orders agree iff D==1 or
  // C==1, so only take the bulk-memcpy path then (otherwise fall back to the
  // strided loop, which is always correct).
  if (D != 1 && C != 1) return false;
  return row_s == row && plane_s == plane && depth_s == depth && batch_s == batch;
}

// Scatter a contiguous logical fp16 buffer into a (possibly strided) surface.
static void scatter_into_surface(IOSurfaceRef surface, NSDictionary *info,
                                 const uint8_t *src) {
  size_t elem = sizeof(_Float16);
  NSUInteger B = uint_info(info, @"Batches", 1), C = uint_info(info, @"Channels", 1);
  NSUInteger D = uint_info(info, @"Depth", 1), H = uint_info(info, @"Height", 1);
  NSUInteger W = uint_info(info, @"Width", 1);
  IOSurfaceLock(surface, 0, NULL);
  size_t bytes = IOSurfaceGetAllocSize(surface);
  uint8_t *dst = (uint8_t *)IOSurfaceGetBaseAddress(surface);
  size_t dense_bytes = 0;
  if (layout_is_dense(info, &dense_bytes) && dense_bytes <= bytes) {
    // one bulk copy; no memset needed (we overwrite exactly the live region,
    // and any trailing pad bytes are never read by the engine).
    memcpy(dst, src, dense_bytes);
    IOSurfaceUnlock(surface, 0, NULL);
    return;
  }
  memset(dst, 0, bytes);
  NSUInteger li = 0;
  for (NSUInteger b = 0; b < B; b++)
    for (NSUInteger c = 0; c < C; c++)
      for (NSUInteger d = 0; d < D; d++)
        for (NSUInteger h = 0; h < H; h++)
          for (NSUInteger w = 0; w < W; w++, li++) {
            size_t off = logical_offset_for(info, b, c, d, h, w);
            if (off + elem <= bytes) memcpy(dst + off, src + li * elem, elem);
          }
  IOSurfaceUnlock(surface, 0, NULL);
}

// Gather a (possibly strided) surface into a contiguous logical fp16 buffer.
static void gather_from_surface(IOSurfaceRef surface, NSDictionary *info,
                                uint8_t *dst) {
  size_t elem = sizeof(_Float16);
  NSUInteger B = uint_info(info, @"Batches", 1), C = uint_info(info, @"Channels", 1);
  NSUInteger D = uint_info(info, @"Depth", 1), H = uint_info(info, @"Height", 1);
  NSUInteger W = uint_info(info, @"Width", 1);
  IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
  size_t bytes = IOSurfaceGetAllocSize(surface);
  const uint8_t *src = (const uint8_t *)IOSurfaceGetBaseAddress(surface);
  size_t dense_bytes = 0;
  if (layout_is_dense(info, &dense_bytes) && dense_bytes <= bytes) {
    memcpy(dst, src, dense_bytes);
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return;
  }
  NSUInteger li = 0;
  for (NSUInteger b = 0; b < B; b++)
    for (NSUInteger c = 0; c < C; c++)
      for (NSUInteger d = 0; d < D; d++)
        for (NSUInteger h = 0; h < H; h++)
          for (NSUInteger w = 0; w < W; w++, li++) {
            size_t off = logical_offset_for(info, b, c, d, h, w);
            if (off + elem <= bytes) memcpy(dst + li * elem, src + off, elem);
          }
  IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
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

// CLI ----------------------------------------------------------------------
struct Args {
  std::string netplist;
  std::vector<std::string> weights;
  std::vector<std::string> input_syms;   // wire order
  std::vector<std::string> output_syms;  // wire order
  unsigned int qos = 33;
};

static int parse_args(int argc, char **argv, Args &out) {
  for (int i = 1; i < argc; i++) {
    std::string flag = argv[i];
    auto next = [&]() -> const char * {
      if (i + 1 >= argc) { fprintf(stderr, "missing arg for %s\n", flag.c_str()); exit(2); }
      return argv[++i];
    };
    if (flag == "--net-plist") out.netplist = next();
    else if (flag == "--weights") out.weights.push_back(next());
    else if (flag == "--input") out.input_syms.push_back(next());
    else if (flag == "--output") out.output_syms.push_back(next());
    else if (flag == "--qos") out.qos = (unsigned int)atoi(next());
    else { fprintf(stderr, "unknown flag: %s\n", flag.c_str()); return 2; }
  }
  if (out.netplist.empty()) { fprintf(stderr, "--net-plist required\n"); return 2; }
  return 0;
}

// Read exactly n bytes from fd (stdin). Returns false on EOF/short read.
static bool read_exact(int fd, uint8_t *buf, size_t n) {
  size_t got = 0;
  while (got < n) {
    ssize_t r = read(fd, buf + got, n - got);
    if (r <= 0) return false;
    got += (size_t)r;
  }
  return true;
}

static bool write_exact(int fd, const uint8_t *buf, size_t n) {
  size_t put = 0;
  while (put < n) {
    ssize_t w = write(fd, buf + put, n - put);
    if (w <= 0) return false;
    put += (size_t)w;
  }
  return true;
}

int main(int argc, char **argv) {
  Args args;
  if (int rc = parse_args(argc, argv, args)) return rc;

  @autoreleasepool {
    void *handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW);
    if (!handle) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }

    NSString *netplist_path = [NSString stringWithUTF8String:args.netplist.c_str()];
    NSData *net = [NSData dataWithContentsOfFile:netplist_path];
    if (!net) { fprintf(stderr, "cannot read netplist %s\n", args.netplist.c_str()); return 1; }
    NSMutableDictionary *weights = [NSMutableDictionary dictionary];
    for (const auto &wpath : args.weights) {
      NSString *p = [NSString stringWithUTF8String:wpath.c_str()];
      NSData *d = [NSData dataWithContentsOfFile:p];
      if (!d) { fprintf(stderr, "cannot read weight %s\n", wpath.c_str()); return 1; }
      weights[[p lastPathComponent]] = @{@"offset": @0, @"data": d};
    }

    Class desc_cls = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class model_cls = NSClassFromString(@"_ANEInMemoryModel");
    SEL desc_sel = @selector(modelWithNetworkDescription:weights:optionsPlist:);

    id desc = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(desc_cls, desc_sel, net, weights, nil);
    if (!desc) { fprintf(stderr, "descriptor creation failed\n"); return 3; }
    id model = ((id (*)(Class, SEL, id))objc_msgSend)(
        model_cls, @selector(inMemoryModelWithDescriptor:), desc);
    if (!model) { fprintf(stderr, "model creation failed\n"); return 3; }

    NSString *local_path = call_obj0(model, "localModelPath");
    if ([local_path isKindOfClass:[NSString class]] && [local_path length] > 0) {
      [[NSFileManager defaultManager] createDirectoryAtPath:local_path
                                withIntermediateDirectories:YES attributes:nil error:nil];
      [net writeToFile:[local_path stringByAppendingPathComponent:@"net.plist"] atomically:YES];
      for (NSString *wname in weights)
        [weights[wname][@"data"] writeToFile:[local_path stringByAppendingPathComponent:wname]
                                  atomically:YES];
    }

    NSError *err = nil;
    uint64_t tc0 = now_ns();
    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), args.qos, @{}, &err);
    double compile_ms = (double)(now_ns() - tc0) / 1e6;
    if (!ok) {
      printf("{\"status\":\"compile_failed\",\"error\":\"%s\"}\n",
             jesc(err ? err.description.UTF8String : "(nil)").c_str());
      fflush(stdout);
      return 4;
    }
    uint64_t tl0 = now_ns();
    ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), args.qos, @{}, &err);
    double load_ms = (double)(now_ns() - tl0) / 1e6;
    if (!ok) { printf("{\"status\":\"load_failed\"}\n"); fflush(stdout); return 5; }

    NSArray *live_inputs = live_io_list(model, @"LiveInputList");
    NSArray *live_outputs = live_io_list(model, @"LiveOutputList");

    // Map symbol -> LiveIO descriptor.
    std::map<std::string, NSDictionary *> in_by_sym, out_by_sym;
    for (NSDictionary *info in live_inputs)
      in_by_sym[[(NSString *)info[@"Symbol"] UTF8String] ?: ""] = info;
    for (NSDictionary *info in live_outputs)
      out_by_sym[[(NSString *)info[@"Symbol"] UTF8String] ?: ""] = info;

    // Build the input/output surfaces in the *device* LiveIO order, but record
    // for each wire symbol its surface + descriptor + logical byte count.
    Class surface_cls = NSClassFromString(@"_ANEIOSurfaceObject");
    Class request_cls = NSClassFromString(@"_ANERequest");

    NSMutableArray *wrapped_inputs = [NSMutableArray array];
    NSMutableArray *wrapped_outputs = [NSMutableArray array];
    std::vector<IOSurfaceRef> in_surfaces(live_inputs.count);
    std::vector<IOSurfaceRef> out_surfaces(live_outputs.count);
    std::map<std::string, size_t> in_dev_index, out_dev_index;
    for (NSUInteger i = 0; i < live_inputs.count; i++) {
      NSDictionary *info = live_inputs[i];
      IOSurfaceRef s = make_surface(bytes_for_live_io(info));
      in_surfaces[i] = s;
      in_dev_index[[(NSString *)info[@"Symbol"] UTF8String] ?: ""] = i;
      [wrapped_inputs addObject:((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                                    surface_cls, @selector(objectWithIOSurface:), s)];
    }
    for (NSUInteger i = 0; i < live_outputs.count; i++) {
      NSDictionary *info = live_outputs[i];
      IOSurfaceRef s = make_surface(bytes_for_live_io(info));
      out_surfaces[i] = s;
      out_dev_index[[(NSString *)info[@"Symbol"] UTF8String] ?: ""] = i;
      [wrapped_outputs addObject:((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                                     surface_cls, @selector(objectWithIOSurface:), s)];
    }

    id request = ((id (*)(Class, SEL, id, id, id, id, id))objc_msgSend)(
        request_cls,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:),
        wrapped_inputs, indices_for_count(wrapped_inputs.count),
        wrapped_outputs, indices_for_count(wrapped_outputs.count), @0);

    NSError *map_err = nil;
    BOOL mapped = ((BOOL (*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
        model, @selector(mapIOSurfacesWithRequest:cacheInference:error:), request, YES, &map_err);
    if (!mapped) { printf("{\"status\":\"map_failed\"}\n"); fflush(stdout); return 7; }

    // Resolve wire order -> (surface, descriptor, logical byte count).
    struct WireIO { IOSurfaceRef surface; NSDictionary *info; size_t bytes; };
    std::vector<WireIO> in_wire, out_wire;
    size_t in_total = 0, out_total = 0;
    for (const auto &sym : args.input_syms) {
      auto it = in_dev_index.find(sym);
      if (it == in_dev_index.end()) { fprintf(stderr, "no live input %s\n", sym.c_str()); return 6; }
      NSDictionary *info = live_inputs[it->second];
      size_t b = (size_t)logical_count_for(info) * sizeof(_Float16);
      in_wire.push_back({in_surfaces[it->second], info, b});
      in_total += b;
    }
    for (const auto &sym : args.output_syms) {
      auto it = out_dev_index.find(sym);
      if (it == out_dev_index.end()) { fprintf(stderr, "no live output %s\n", sym.c_str()); return 6; }
      NSDictionary *info = live_outputs[it->second];
      size_t b = (size_t)logical_count_for(info) * sizeof(_Float16);
      out_wire.push_back({out_surfaces[it->second], info, b});
      out_total += b;
    }

    // Announce readiness + the wire byte counts (so the driver can frame).
    {
      NSMutableArray *in_counts = [NSMutableArray array];
      NSMutableArray *out_counts = [NSMutableArray array];
      for (auto &w : in_wire) [in_counts addObject:@(w.bytes / 2)];
      for (auto &w : out_wire) [out_counts addObject:@(w.bytes / 2)];
      NSDictionary *ready = @{
        @"status": @"ready", @"compile_ms": @(compile_ms), @"load_ms": @(load_ms),
        @"input_elems": in_counts, @"output_elems": out_counts,
      };
      NSData *j = [NSJSONSerialization dataWithJSONObject:ready options:0 error:nil];
      fwrite([j bytes], 1, [j length], stdout);
      fputc('\n', stdout);
      fflush(stdout);
    }

    SEL eval_sel = @selector(evaluateWithQoS:options:request:error:);
    BOOL (*eval_fn)(id, SEL, unsigned int, id, id, NSError **) =
        (BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend;

    // Reusable scratch; grown lazily to hold a whole batch (N input/output sets)
    // so a batched request needs no per-call allocation in steady state.
    std::vector<uint8_t> inbuf, outbuf;

    // Service loop: read a uint32 count N + N packed input sets, run N evals,
    // write N packed output sets -- all in ONE pipe round-trip.
    for (;;) {
      uint8_t hdr[4];
      if (!read_exact(STDIN_FILENO, hdr, 4)) break;  // EOF before count -> shutdown
      uint32_t n = (uint32_t)hdr[0] | ((uint32_t)hdr[1] << 8) |
                   ((uint32_t)hdr[2] << 16) | ((uint32_t)hdr[3] << 24);
      if (n == 0) { fprintf(stderr, "batch count 0\n"); return 8; }
      size_t batch_in = in_total * (size_t)n, batch_out = out_total * (size_t)n;
      if (inbuf.size() < batch_in) inbuf.resize(batch_in);
      if (outbuf.size() < batch_out) outbuf.resize(batch_out);
      if (!read_exact(STDIN_FILENO, inbuf.data(), batch_in)) break;  // short read -> shutdown

      size_t out_off = 0;
      for (uint32_t e_i = 0; e_i < n; e_i++) {
        size_t in_off = (size_t)e_i * in_total;
        for (auto &w : in_wire) { scatter_into_surface(w.surface, w.info, inbuf.data() + in_off); in_off += w.bytes; }

        BOOL e = eval_fn(model, eval_sel, args.qos, @{}, request, &err);
        if (!e) { fprintf(stderr, "eval failed: %s\n", err ? err.description.UTF8String : "(nil)"); return 8; }

        for (auto &w : out_wire) { gather_from_surface(w.surface, w.info, outbuf.data() + out_off); out_off += w.bytes; }
      }
      if (!write_exact(STDOUT_FILENO, outbuf.data(), batch_out)) break;
    }

    SEL unmap_sel = @selector(unmapIOSurfacesWithRequest:);
    if ([model respondsToSelector:unmap_sel])
      ((void (*)(id, SEL, id))objc_msgSend)(model, unmap_sel, request);
    SEL unload_sel = @selector(unloadWithQoS:error:);
    if ([model respondsToSelector:unload_sel])
      ((BOOL (*)(id, SEL, unsigned int, NSError **))[model methodForSelector:unload_sel])(
          model, unload_sel, args.qos, &err);
    for (IOSurfaceRef s : in_surfaces) CFRelease(s);
    for (IOSurfaceRef s : out_surfaces) CFRelease(s);
  }
  return 0;
}
