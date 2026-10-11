/* A consumer that includes only the pinned upstream TAPP headers (no tcontract
   header) and performs a serial contraction D = A * B through libtcontract_capi:
   the standard ABI is enough for serial use. */
#include <stdio.h>
#include <tapp.h>

#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); return 1; } } while (0)

int main(void) {
  TAPP_handle handle;
  TAPP_executor exec;
  CHECK(TAPP_check_success(TAPP_create_handle(&handle)), "create handle");
  CHECK(TAPP_check_success(TAPP_create_executor(&exec)), "create executor");

  /* A: 2x3, B: 3x2, D: 2x2, all column-major (strides in elements). */
  int64_t ea[2] = {2, 3}, sa[2] = {1, 2};
  int64_t eb[2] = {3, 2}, sb[2] = {1, 3};
  int64_t ed[2] = {2, 2}, sd[2] = {1, 2};
  TAPP_tensor_info ia, ib, id;
  CHECK(TAPP_check_success(TAPP_create_tensor_info(&ia, TAPP_F64, 2, ea, sa)), "info A");
  CHECK(TAPP_check_success(TAPP_create_tensor_info(&ib, TAPP_F64, 2, eb, sb)), "info B");
  CHECK(TAPP_check_success(TAPP_create_tensor_info(&id, TAPP_F64, 2, ed, sd)), "info D");

  int64_t la[2] = {'i', 'k'}, lb[2] = {'k', 'j'}, ld[2] = {'i', 'j'};
  TAPP_tensor_product plan;
  CHECK(TAPP_check_success(TAPP_create_tensor_product(
            &plan, handle, TAPP_IDENTITY, ia, la, TAPP_IDENTITY, ib, lb,
            TAPP_IDENTITY, id, ld, TAPP_IDENTITY, id, ld, TAPP_DEFAULT_PREC)),
        "create product");

  double a[6] = {1, 2, 3, 4, 5, 6};   /* A[i + 2k] */
  double b[6] = {1, 2, 3, 4, 5, 6};   /* B[k + 3j] */
  double d[4] = {-1, -1, -1, -1};
  double alpha = 1.0, beta = 0.0;
  TAPP_status status = -1;
  CHECK(TAPP_check_success(TAPP_execute_product(plan, exec, &status, &alpha, a, b, &beta, TAPP_IN_PLACE, d)),
        "execute");
  CHECK(status == 0, "status written");
  /* D[i,j] = sum_k A[i,k] B[k,j] */
  CHECK(d[0] == 22 && d[1] == 28 && d[2] == 49 && d[3] == 64, "values");

  /* The default executor 0 is the same serial executor. */
  d[0] = d[1] = d[2] = d[3] = -1;
  CHECK(TAPP_check_success(TAPP_execute_product(plan, 0, NULL, &alpha, a, b, &beta, TAPP_IN_PLACE, d)),
        "execute on executor 0");
  CHECK(d[0] == 22 && d[3] == 64, "values on executor 0");

  CHECK(TAPP_check_success(TAPP_destroy_status(status)), "destroy status");
  CHECK(TAPP_check_success(TAPP_destroy_tensor_product(plan)), "destroy product");
  CHECK(TAPP_check_success(TAPP_destroy_tensor_info(ia)), "destroy A");
  CHECK(TAPP_check_success(TAPP_destroy_tensor_info(ib)), "destroy B");
  CHECK(TAPP_check_success(TAPP_destroy_tensor_info(id)), "destroy D");
  CHECK(TAPP_check_success(TAPP_destroy_executor(exec)), "destroy executor");
  CHECK(TAPP_check_success(TAPP_destroy_handle(handle)), "destroy handle");
  printf("standard_consumer ok\n");
  return 0;
}
