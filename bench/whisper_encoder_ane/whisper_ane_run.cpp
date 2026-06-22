// Standalone Apple Neural Engine runner for the whisper-tiny encoder: loads the
// program ANEForge produced (model.mil + weights.bin) and dispatches it on the ANE,
// with no Python. This is the shape of the call a C inference engine (e.g. a
// whisper.cpp backend) would make at model-load time.
//
//   whisper_ane_run BUILD_DIR  IN0 N0 F0  IN1 N1 F1  OUT NOUT  REF.f32
//
// IN*/N*/F*: input port name, element count, raw-fp16 file (from export_bundle.py).
// OUT/NOUT: output port name and element count. REF.f32: fp32 reference to score.
//
// It links libane_e5rt_dispatch.dylib and calls ane_e5rt_program_compile. Note: the
// cached on-device ANEF macho is NOT cold-loadable by a fresh process
// (program_library_create succeeds but creating the precompiled operation fails with
// "Must re-compile the E5 bundle"). So the supported reuse path is compile-with-cache:
// ship model.mil + weights.bin, compile once per process at model load (cached after).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

typedef struct ane_e5rt_program ane_e5rt_program_t;
extern "C" {
ane_e5rt_program_t *ane_e5rt_program_compile(
    const char *mil_path, const char *cache_dir, uint64_t device_mask,
    const char *const *input_names, const size_t *input_sizes, size_t n_inputs,
    const char *const *output_names, const size_t *output_sizes, size_t n_outputs);
int  ane_e5rt_program_set_input_fp16(ane_e5rt_program_t *, const char *, const uint16_t *, size_t);
int  ane_e5rt_program_get_output_fp16(ane_e5rt_program_t *, const char *, uint16_t *, size_t);
int  ane_e5rt_program_execute(ane_e5rt_program_t *);
void ane_e5rt_program_release(ane_e5rt_program_t *);
}

static float half2float(uint16_t h) {
  uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, out;
  if (e == 0)         out = m ? ((s << 31) | (0x38800000u + (m << 13))) : (s << 31);
  else if (e == 0x1f) out = (s << 31) | 0x7f800000u | (m << 13);
  else                out = (s << 31) | ((e + 112) << 23) | (m << 13);
  float f; memcpy(&f, &out, 4); return f;
}

static std::vector<uint16_t> read_f16(const char *path, size_t n) {
  std::vector<uint16_t> v(n);
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "open %s failed\n", path); exit(2); }
  if (fread(v.data(), 2, n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(2); }
  fclose(f);
  return v;
}

int main(int argc, char **argv) {
  if (argc != 11) {
    fprintf(stderr, "usage: %s BUILD_DIR IN0 N0 F0 IN1 N1 F1 OUT NOUT REF.f32\n", argv[0]);
    return 1;
  }
  char mil_path[2048], cache_dir[2048];
  snprintf(mil_path, sizeof(mil_path), "%s/model.mil", argv[1]);
  snprintf(cache_dir, sizeof(cache_dir), "%s/cache", argv[1]);
  const char *in_names[2] = {argv[2], argv[5]};
  size_t in_nelems[2] = {(size_t)atoll(argv[3]), (size_t)atoll(argv[6])};
  const char *in_files[2] = {argv[4], argv[7]};
  const char *out_name = argv[8];
  size_t out_nelems = (size_t)atoll(argv[9]);
  size_t in_bytes[2] = {in_nelems[0] * 2, in_nelems[1] * 2};
  size_t out_bytes = out_nelems * 2;

  auto t0 = std::chrono::high_resolution_clock::now();
  ane_e5rt_program_t *prog = ane_e5rt_program_compile(
      mil_path, cache_dir, 0x4 /*ANE*/, in_names, in_bytes, 2, &out_name, &out_bytes, 1);
  if (!prog) { fprintf(stderr, "ane_e5rt_program_compile failed\n"); return 3; }
  auto t1 = std::chrono::high_resolution_clock::now();
  printf("compiled+loaded encoder in %.0f ms (one-time, per process)\n",
         std::chrono::duration<double, std::milli>(t1 - t0).count());

  for (int i = 0; i < 2; i++) {
    auto v = read_f16(in_files[i], in_nelems[i]);
    if (ane_e5rt_program_set_input_fp16(prog, in_names[i], v.data(), in_nelems[i]) != 0) {
      fprintf(stderr, "set_input %s failed\n", in_names[i]);
      return 4;
    }
  }

  if (ane_e5rt_program_execute(prog) != 0) { fprintf(stderr, "execute failed\n"); return 5; }
  const int REP = 20;
  auto e0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < REP; i++) ane_e5rt_program_execute(prog);
  auto e1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(e1 - e0).count() / REP;

  std::vector<uint16_t> out16(out_nelems);
  if (ane_e5rt_program_get_output_fp16(prog, out_name, out16.data(), out_nelems) != 0) {
    fprintf(stderr, "get_output failed\n");
    return 6;
  }

  std::vector<float> ref(out_nelems);
  FILE *rf = fopen(argv[10], "rb");
  if (!rf || fread(ref.data(), 4, out_nelems, rf) != out_nelems) { fprintf(stderr, "ref read failed\n"); return 7; }
  fclose(rf);
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < out_nelems; i++) {
    double a = half2float(out16[i]), b = ref[i];
    dot += a * b; na += a * a; nb += b * b;
  }
  printf("ANE encode latency: %.3f ms/call (avg of %d)\n", ms, REP);
  printf("cosine vs torch reference: %.6f\n", dot / (sqrt(na) * sqrt(nb)));
  ane_e5rt_program_release(prog);
  return 0;
}
