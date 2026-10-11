#define _POSIX_C_SOURCE 199309L
/* Per-call cost through the C ABI: empty call, a 2x2x2 contraction
   through TAPP (TAPP_execute_product), serial and on a 4-thread pool created
   from C. Prints
   case,variant,threads,median_ns,samples (per call; each sample is the mean
   of INNER calls). Pair with the Rust `capi_rust` bench binary. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tcontract/tcontract.h"

#define INNER 1000
#define SAMPLES 101

static double now_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static DLTensor f64t(double *d, int32_t nd, int64_t *shape, int64_t *strides) {
  DLTensor t;
  memset(&t, 0, sizeof t);
  t.data = d; t.device.device_type = kDLCPU; t.ndim = nd;
  t.dtype.code = kDLFloat; t.dtype.bits = 64; t.dtype.lanes = 1;
  t.shape = shape; t.strides = strides;
  return t;
}

static volatile uint32_t sink;

int main(int argc, char **argv) {
  int threads = argc > 1 ? atoi(argv[1]) : 1;
  /* threads == 1 is an explicit serial executor (no workers, no pool). */
  TAPP_executor ex = 0;
  if (tcontract_tapp_executor_create_rayon(&ex, (size_t)threads, NULL) != TCONTRACT_OK) { fprintf(stderr, "exec: %s\n", tcontract_last_error()); return 1; }
  size_t pool_size = 0, budget = 0;
  tcontract_tapp_executor_get_threads(ex, &pool_size, &budget);
  printf("# threads: requested=%d pool=%zu budget=%zu\n", threads, pool_size, budget);
  printf("case,variant,threads,median_ns,samples\n");
  double s[SAMPLES];

  for (int k = 0; k < SAMPLES; k++) {
    double t0 = now_ns();
    for (int i = 0; i < INNER; i++) sink += tcontract_abi_version();
    s[k] = (now_ns() - t0) / INNER;
  }
  qsort(s, SAMPLES, sizeof(double), cmp);
  printf("empty_call,c,%d,%.1f,%d\n", threads, s[SAMPLES / 2], SAMPLES);

  int64_t s3[3] = {2, 2, 2}, st3[3] = {1, 2, 4}, s4[4] = {2, 2, 2, 2}, st4[4] = {1, 2, 4, 8};
  double x[8], y[8], z[16];
  for (int i = 0; i < 8; i++) { x[i] = i; y[i] = 8 - i; }
    /* z[a,b,c,d] = sum_k x[a,b,k] y[k,c,d]. */
  TAPP_handle handle;
  TAPP_tensor_info ix, iy, iz;
  int64_t lx[3] = {'a', 'b', 'k'}, ly[3] = {'k', 'c', 'd'}, lz[4] = {'a', 'b', 'c', 'd'};
  TAPP_tensor_product plan;
  if (!TAPP_check_success(TAPP_create_handle(&handle)) ||
      !TAPP_check_success(TAPP_create_tensor_info(&ix, TAPP_F64, 3, s3, st3)) ||
      !TAPP_check_success(TAPP_create_tensor_info(&iy, TAPP_F64, 3, s3, st3)) ||
      !TAPP_check_success(TAPP_create_tensor_info(&iz, TAPP_F64, 4, s4, st4)) ||
      !TAPP_check_success(TAPP_create_tensor_product(&plan, handle, TAPP_IDENTITY, ix, lx, TAPP_IDENTITY, iy, ly,
                                                     TAPP_IDENTITY, iz, lz, TAPP_IDENTITY, iz, lz, TAPP_DEFAULT_PREC))) {
    fprintf(stderr, "%s\n", tcontract_last_error()); return 1;
  }
  for (int k = 0; k < SAMPLES; k++) {
    double t0 = now_ns();
    for (int i = 0; i < INNER; i++)
      TAPP_execute_product(plan, ex, NULL, &one, x, y, &zero, TAPP_IN_PLACE, z);
    s[k] = (now_ns() - t0) / INNER;
  }
  qsort(s, SAMPLES, sizeof(double), cmp);
  printf("contract_2x2x2,c,%d,%.1f,%d\n", threads, s[SAMPLES / 2], SAMPLES);
  TAPP_destroy_tensor_product(plan);
  TAPP_destroy_tensor_info(ix);
  TAPP_destroy_tensor_info(iy);
  TAPP_destroy_tensor_info(iz);
  TAPP_destroy_handle(handle);

  double t0 = now_ns();
  int st_close = TAPP_destroy_executor(ex);
  printf("# destroy: status=%d %.1f us\n", st_close, (now_ns() - t0) / 1e3);
  return 0;
}
