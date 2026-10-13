/* End-to-end test of libtcontract_capi from C: pools, TAPP products sharing
   one executor, error reporting. Exit status 0 on success. */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "tcontract/tcontract.h"

#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, msg, tcontract_last_error()); return 1; } } while (0)

int main(void) {
  CHECK(tcontract_abi_version() == TCONTRACT_ABI_VERSION, "abi version");
  CHECK(tcontract_has_part("tapp") && tcontract_has_part("core"), "parts");
  CHECK(!tcontract_has_part("blas"), "blas is not part of libtcontract_capi");
  CHECK(!tcontract_has_part("linalg"), "linalg not in this slice");

  /* A and B of the product below: A is 2x3 row-major, B is 3x2 column-major. */
  double b[6] = {1, 3, 5, 2, 4, 6}; /* columns [1,3,5], [2,4,6] */
  double one = 1.0, zero = 0.0;

  /* One executor type for every part: 0 (serial), TAPP_create_executor, an
     owned Rayon pool, and a Rayon pool the host owns and lends, all go to TAPP
     products. */
  TAPP_executor pool = 0, serial = 0;
  CHECK(tcontract_tapp_executor_create_rayon(&pool, 4, NULL) == TCONTRACT_OK && pool != 0, "pool create");
  size_t psize = 99, pbudget = 99;
  CHECK(tcontract_tapp_executor_get_threads(pool, &psize, &pbudget) == TCONTRACT_OK && psize == 4 && pbudget == 4, "pool size");
  CHECK(tcontract_tapp_executor_set_budget(pool, 2) == TCONTRACT_OK, "budget");
  CHECK(tcontract_tapp_executor_get_threads(pool, &psize, &pbudget) == TCONTRACT_OK && psize == 4 && pbudget == 2, "budget set");
  CHECK(tcontract_tapp_executor_set_budget(pool, 0) == TCONTRACT_ERR_INVALID_ARGUMENT, "zero budget");
  CHECK(tcontract_tapp_executor_set_budget(pool, 4) == TCONTRACT_OK, "budget reset");
  CHECK(TAPP_check_success(TAPP_create_executor(&serial)), "serial executor");
  CHECK(tcontract_tapp_executor_get_threads(serial, &psize, &pbudget) == TCONTRACT_OK && psize == 0 && pbudget == 1, "serial threads");

  /* A pool the host owns: one handle, one borrowed executor on it. Its budget
     is clamped to the pool width exactly as set_budget clamps it. */
  tcontract_pool host = 0;
  TAPP_executor borrowed = 0;
  CHECK(tcontract_pool_create_rayon(&host, 3, NULL) == TCONTRACT_OK && host != 0, "host pool create");
  CHECK(tcontract_tapp_executor_borrow_pool(&borrowed, host, 2) == TCONTRACT_OK && borrowed != 0, "borrow pool");
  CHECK(tcontract_tapp_executor_get_threads(borrowed, &psize, &pbudget) == TCONTRACT_OK && psize == 3 && pbudget == 2, "borrowed width and budget");
  CHECK(tcontract_tapp_executor_set_budget(borrowed, 9) == TCONTRACT_OK, "borrowed budget");
  CHECK(tcontract_tapp_executor_get_threads(borrowed, &psize, &pbudget) == TCONTRACT_OK && psize == 3 && pbudget == 3, "borrowed budget clamped to the pool");
  CHECK(tcontract_pool_destroy(host) == TCONTRACT_BUSY, "pool destroy refused while an executor is live");
  CHECK(tcontract_pool_create_rayon(&host, 0, NULL) == TCONTRACT_ERR_INVALID_ARGUMENT, "zero pool workers");
  CHECK(tcontract_pool_destroy(0) == TCONTRACT_ERR_INVALID_ARGUMENT, "no pool handle");
  CHECK(tcontract_tapp_executor_borrow_pool(&borrowed, 0, 1) == TCONTRACT_ERR_INVALID_ARGUMENT, "nothing to borrow");
  TAPP_executor execs[4] = {0, serial, pool, borrowed};
  /* D = A * B with labels
     i,k / k,j / i,j; A is row-major (strides {3,1}), B and D column-major. */
  {
    TAPP_handle handle;
    CHECK(TAPP_check_success(TAPP_create_handle(&handle)), "handle");
    int64_t ea[2] = {2, 3}, sa_[2] = {3, 1}, eb[2] = {3, 2}, sb_[2] = {1, 3}, ed[2] = {2, 2}, sd_[2] = {1, 2};
    TAPP_tensor_info ia, ib, id;
    CHECK(TAPP_check_success(TAPP_create_tensor_info(&ia, TAPP_F64, 2, ea, sa_)), "info A");
    CHECK(TAPP_check_success(TAPP_create_tensor_info(&ib, TAPP_F64, 2, eb, sb_)), "info B");
    CHECK(TAPP_check_success(TAPP_create_tensor_info(&id, TAPP_F64, 2, ed, sd_)), "info D");
    int64_t la[2] = {'i', 'k'}, lb[2] = {'k', 'j'}, ld[2] = {'i', 'j'};
    TAPP_tensor_product plan;
    CHECK(TAPP_check_success(TAPP_create_tensor_product(&plan, handle, TAPP_IDENTITY, ia, la, TAPP_IDENTITY, ib, lb,
                                                        TAPP_IDENTITY, id, ld, TAPP_IDENTITY, id, ld, TAPP_DEFAULT_PREC)),
          "plan create");
    CHECK(TAPP_create_tensor_product(&plan, 0, TAPP_IDENTITY, ia, la, TAPP_IDENTITY, ib, lb, TAPP_IDENTITY, id, ld,
                                     TAPP_IDENTITY, id, ld, TAPP_DEFAULT_PREC) == TCONTRACT_ERR_INVALID_ARGUMENT,
          "zero handle rejected");
    CHECK(TAPP_check_success(TAPP_create_tensor_product(&plan, handle, TAPP_IDENTITY, ia, la, TAPP_IDENTITY, ib, lb,
                                                        TAPP_IDENTITY, id, ld, TAPP_IDENTITY, id, ld, TAPP_DEFAULT_PREC)),
          "plan create again");
    double ra[6] = {1, 3, 5, 2, 4, 6}; /* row-major rows [1,3,5], [2,4,6] */
    for (int i = 0; i < 4; i++) {
      double d[4] = {NAN, NAN, NAN, NAN};
      TAPP_status status = -1;
      tcontract_status st = TAPP_execute_product(plan, execs[i], &status, &one, ra, b, &zero, TAPP_IN_PLACE, d);
      CHECK(st == TCONTRACT_OK && status == 0, "TAPP execute");
      CHECK(d[0] == 35 && d[1] == 44 && d[2] == 44 && d[3] == 56, "TAPP values");
    }
    /* Output aliasing an input is rejected before any write. */
    CHECK(TAPP_execute_product(plan, 0, NULL, &one, ra, b, &zero, TAPP_IN_PLACE, ra) == TCONTRACT_ERR_ALIASED,
          "aliasing rejected");
    CHECK(strlen(tcontract_last_error()) > 0, "aliasing message");
    CHECK(TAPP_check_success(TAPP_destroy_tensor_product(plan)), "plan destroy");
    CHECK(TAPP_check_success(TAPP_destroy_tensor_info(ia)) && TAPP_check_success(TAPP_destroy_tensor_info(ib)) &&
              TAPP_check_success(TAPP_destroy_tensor_info(id)),
          "info destroy");
    CHECK(TAPP_check_success(TAPP_destroy_handle(handle)), "handle destroy");
  }

  /* Destruction: an owned pool is stopped and joined, a borrowed executor
     releases its borrow before the host pool is destroyed; the default executor
     is a no-op. */
  CHECK(TAPP_check_success(TAPP_destroy_executor(pool)), "destroy pool joins");
  CHECK(TAPP_check_success(TAPP_destroy_executor(serial)), "destroy serial");
  CHECK(TAPP_check_success(TAPP_destroy_executor(borrowed)), "destroy borrowed executor");
  CHECK(TAPP_check_success(tcontract_pool_destroy(host)), "destroy host pool");
  CHECK(TAPP_check_success(TAPP_destroy_executor(0)), "destroy default executor");
  CHECK(tcontract_tapp_executor_create_rayon(&pool, 0, NULL) == TCONTRACT_ERR_INVALID_ARGUMENT, "zero threads");
  char msg[64];
  CHECK(TAPP_explain_error(TCONTRACT_BUSY, sizeof msg, msg) == strlen(msg) && strlen(msg) > 0, "explain");
  printf("abi_test ok\n");
  return 0;
}
