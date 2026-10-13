/*
 * A C consumer of the TAPP surface, doing real work.
 *
 * This exists to make "a C or C++ project can link this" a *tested* claim
 * rather than an inferred one. `tests/abi_layout.rs` already re-declares the
 * whole interface and drives a contraction through it — but it does that from
 * Rust, so no C compiler has ever seen `include/tapp.h`, and a header that
 * disagreed with the library would not have been caught anywhere.
 *
 * So this program is compiled by a C compiler, against the shipped header,
 * linked against the built library, and checks numerical results rather than
 * just return codes. A wrong stride convention, a mis-declared prototype or a
 * complex layout mismatch all show up as a wrong number here.
 *
 * It covers: real and complex element types, non-trivial extents, alpha/beta,
 * an explicit C term, conjugation on an operand, the in-place refusal, and the
 * error-explanation path.
 */

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tapp.h>
#include <tcontract/tcontract.h>

/* D is M x N, A is M x K, B is K x N, all row-major. */
#define M 3
#define K 4
#define N 2

static int failures = 0;

static void check(int cond, const char* what) {
    if (cond) {
        printf("  ok    %s\n", what);
    } else {
        printf("  FAIL  %s\n", what);
        failures++;
    }
}

/* Report a TAPP_error with the library's own explanation, and abort the run:
 * once a call has failed there is nothing meaningful to compare against. */
static void must(TAPP_error err, const char* what) {
    if (!TAPP_check_success(err)) {
        char msg[256];
        TAPP_explain_error(err, sizeof msg, msg);
        fprintf(stderr, "%s failed: %s (code %d)\n", what, msg, err);
        exit(1);
    }
}

/* Labels: D[i,j] = sum_k A[i,k] * B[k,j]. Mode 2 is the contracted one. */
static const int64_t idx_a[2] = {0, 2};
static const int64_t idx_b[2] = {2, 1};
static const int64_t idx_cd[2] = {0, 1};

static const int64_t ext_a[2] = {M, K};
static const int64_t ext_b[2] = {K, N};
static const int64_t ext_d[2] = {M, N};

static const int64_t str_a[2] = {K, 1};
static const int64_t str_b[2] = {N, 1};
static const int64_t str_d[2] = {N, 1};

/* ---------------------------------------------------------------- f64 case */

static void test_f64(TAPP_handle handle, TAPP_executor exec) {
    double a[M * K], b[K * N], c[M * N], d[M * N], want[M * N];
    const double alpha = 2.5, beta = -0.5;

    for (int i = 0; i < M * K; i++) a[i] = 0.5 * (i + 1);
    for (int i = 0; i < K * N; i++) b[i] = 1.0 - 0.25 * i;
    for (int i = 0; i < M * N; i++) c[i] = 3.0 + i;

    /* Reference, computed here so the check is independent of the engine. */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) acc += a[i * K + k] * b[k * N + j];
            want[i * N + j] = alpha * acc + beta * c[i * N + j];
        }
    }

    TAPP_tensor_info ia, ib, ic, id;
    must(TAPP_create_tensor_info(&ia, TAPP_F64, 2, ext_a, str_a), "create A");
    must(TAPP_create_tensor_info(&ib, TAPP_F64, 2, ext_b, str_b), "create B");
    must(TAPP_create_tensor_info(&ic, TAPP_F64, 2, ext_d, str_d), "create C");
    must(TAPP_create_tensor_info(&id, TAPP_F64, 2, ext_d, str_d), "create D");

    TAPP_tensor_product plan;
    must(TAPP_create_tensor_product(&plan, handle, TAPP_IDENTITY, ia, idx_a,
                                    TAPP_IDENTITY, ib, idx_b, TAPP_IDENTITY, ic,
                                    idx_cd, TAPP_IDENTITY, id, idx_cd,
                                    TAPP_DEFAULT_PREC),
         "plan f64");

    must(TAPP_execute_product(plan, exec, NULL, &alpha, a, b, &beta, c, d),
         "execute f64");

    int good = 1;
    for (int i = 0; i < M * N; i++)
        if (fabs(d[i] - want[i]) > 1e-12 * (1.0 + fabs(want[i]))) good = 0;
    check(good, "f64 D = alpha*A*B + beta*C matches a C reference loop");

    /* Round-trip the descriptor accessors while a live info is to hand. */
    int64_t got_ext[2] = {0, 0};
    TAPP_get_extents(ia, got_ext);
    check(TAPP_get_nmodes(ia) == 2 && got_ext[0] == M && got_ext[1] == K,
          "tensor-info accessors round-trip");

    must(TAPP_destroy_tensor_product(plan), "destroy plan");
    must(TAPP_destroy_tensor_info(ia), "destroy A");
    must(TAPP_destroy_tensor_info(ib), "destroy B");
    must(TAPP_destroy_tensor_info(ic), "destroy C");
    must(TAPP_destroy_tensor_info(id), "destroy D");
}

/* ---------------------------------------------------------------- c64 case */

/*
 * The complex case is the one that matters most for the ABI: it asserts that a
 * C `double _Complex*` is the *same bytes* as what the engine reads, with no
 * conversion. Conjugation on A is exercised because it is applied inside
 * packing rather than by a pre-pass, so a sign error would not show up
 * anywhere else in this program.
 */
static void test_c64_conj(TAPP_handle handle, TAPP_executor exec) {
    double _Complex a[M * K], b[K * N], c[M * N], d[M * N], want[M * N];
    const double _Complex alpha = 1.5 - 0.5 * I, beta = 0.25 + 0.75 * I;

    for (int i = 0; i < M * K; i++) a[i] = (0.5 * (i + 1)) + (0.25 * i) * I;
    for (int i = 0; i < K * N; i++) b[i] = (1.0 - 0.25 * i) + (0.5 - 0.1 * i) * I;
    for (int i = 0; i < M * N; i++) c[i] = (3.0 + i) - (0.5 * i) * I;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double _Complex acc = 0.0;
            for (int k = 0; k < K; k++)
                acc += conj(a[i * K + k]) * b[k * N + j];
            want[i * N + j] = alpha * acc + beta * c[i * N + j];
        }
    }

    TAPP_tensor_info ia, ib, ic, id;
    must(TAPP_create_tensor_info(&ia, TAPP_C64, 2, ext_a, str_a), "create A");
    must(TAPP_create_tensor_info(&ib, TAPP_C64, 2, ext_b, str_b), "create B");
    must(TAPP_create_tensor_info(&ic, TAPP_C64, 2, ext_d, str_d), "create C");
    must(TAPP_create_tensor_info(&id, TAPP_C64, 2, ext_d, str_d), "create D");

    TAPP_tensor_product plan;
    must(TAPP_create_tensor_product(&plan, handle, TAPP_CONJUGATE, ia, idx_a,
                                    TAPP_IDENTITY, ib, idx_b, TAPP_IDENTITY, ic,
                                    idx_cd, TAPP_IDENTITY, id, idx_cd,
                                    TAPP_DEFAULT_PREC),
         "plan c64");

    must(TAPP_execute_product(plan, exec, NULL, &alpha, a, b, &beta, c, d),
         "execute c64");

    int good = 1;
    for (int i = 0; i < M * N; i++)
        if (cabs(d[i] - want[i]) > 1e-12 * (1.0 + cabs(want[i]))) good = 0;
    check(good, "c64 with TAPP_CONJUGATE on A matches a C reference loop");

    must(TAPP_destroy_tensor_product(plan), "destroy plan");
    must(TAPP_destroy_tensor_info(ia), "destroy A");
    must(TAPP_destroy_tensor_info(ib), "destroy B");
    must(TAPP_destroy_tensor_info(ic), "destroy C");
    must(TAPP_destroy_tensor_info(id), "destroy D");
}

/* ------------------------------------------------------- documented refusals */

/*
 * Two behaviours the header promises in prose. If either stopped holding, a
 * caller would get a silently different answer rather than an error, so they
 * are worth a C-level check rather than only a Rust one.
 */
static void test_refusals(TAPP_handle handle, TAPP_executor exec) {
    double a[M * K] = {0}, b[K * N] = {0}, d[M * N] = {0};
    const double alpha = 1.0, nonzero_beta = 1.0;

    TAPP_tensor_info ia, ib, ic, id;
    must(TAPP_create_tensor_info(&ia, TAPP_F64, 2, ext_a, str_a), "create A");
    must(TAPP_create_tensor_info(&ib, TAPP_F64, 2, ext_b, str_b), "create B");
    must(TAPP_create_tensor_info(&ic, TAPP_F64, 2, ext_d, str_d), "create C");
    must(TAPP_create_tensor_info(&id, TAPP_F64, 2, ext_d, str_d), "create D");

    TAPP_tensor_product plan;
    must(TAPP_create_tensor_product(&plan, handle, TAPP_IDENTITY, ia, idx_a,
                                    TAPP_IDENTITY, ib, idx_b, TAPP_IDENTITY, ic,
                                    idx_cd, TAPP_IDENTITY, id, idx_cd,
                                    TAPP_DEFAULT_PREC),
         "plan");

    /* A null C with a non-zero beta is refused, not reinterpreted as C == D. */
    TAPP_error err = TAPP_execute_product(plan, exec, NULL, &alpha, a, b,
                                          &nonzero_beta, TAPP_IN_PLACE, d);
    check(!TAPP_check_success(err), "TAPP_IN_PLACE with beta != 0 is refused");

    /* Every attribute key is refused, but the symbols must exist to link. */
    void* slot = (void*)0x1234;
    check(TAPP_attr_get(0, 0, &slot) == TCONTRACT_ERR_UNSUPPORTED && slot == NULL,
          "attribute API links, refuses, and leaves a defined value");

    must(TAPP_destroy_tensor_product(plan), "destroy plan");
    must(TAPP_destroy_tensor_info(ia), "destroy A");
    must(TAPP_destroy_tensor_info(ib), "destroy B");
    must(TAPP_destroy_tensor_info(ic), "destroy C");
    must(TAPP_destroy_tensor_info(id), "destroy D");
}

/*
 * The library's two version statements agree: `TAPP_implementation_version()`
 * ("major.minor.patch") and `tcontract_abi_version()` (major * 10000 + minor * 100 +
 * patch). The pinned upstream `tapp.h` has no version macros, so the runtime
 * symbols are the only source; a distribution that mixes headers and libraries
 * from different versions still fails on `TCONTRACT_ABI_VERSION`.
 */
static void test_version_agreement(void) {
    const char* linked = TAPP_implementation_version();
    unsigned major = 0, minor = 0, patch = 0;
    check(sscanf(linked, "%u.%u.%u", &major, &minor, &patch) == 3, "version string parses");
    printf("  library %s, ABI %u\n", linked, (unsigned)tcontract_abi_version());
    check(tcontract_abi_version() == major * 10000 + minor * 100 + patch,
          "ABI version matches the library version string");
    check(tcontract_abi_version() == TCONTRACT_ABI_VERSION, "header TCONTRACT_ABI_VERSION matches the linked library");
}

/*
 * Reference every function `<tapp.h>` declares.
 *
 * The tests above call the entry points that do work, which leaves eight
 * declarations that no C compiler has ever checked and no linker has ever
 * resolved: `TAPP_set_nmodes`, `TAPP_set_extents`, `TAPP_set_strides`,
 * `TAPP_get_strides`, `TAPP_execute_batched_product`, `TAPP_destroy_status`,
 * `TAPP_attr_set` and `TAPP_attr_clear`. A wrong prototype for one of those
 * would have been caught only by `abi_layout.rs`'s independent Rust
 * transcription — which is a second hand-maintained copy, not an oracle.
 *
 * Taking each address is enough. It forces the declaration to be well-formed and
 * the symbol to resolve at link time, without this file having to invent
 * semantics for the ones that are deliberately inert.
 */
static void test_every_declaration_links(void) {
    void* const decls[] = {
        (void*)TAPP_check_success,
        (void*)TAPP_explain_error,
        (void*)TAPP_create_handle,
        (void*)TAPP_destroy_handle,
        (void*)TAPP_create_executor,
        (void*)TAPP_destroy_executor,
        (void*)TAPP_destroy_status,
        (void*)TAPP_attr_set,
        (void*)TAPP_attr_get,
        (void*)TAPP_attr_clear,
        (void*)TAPP_create_tensor_info,
        (void*)TAPP_destroy_tensor_info,
        (void*)TAPP_get_nmodes,
        (void*)TAPP_set_nmodes,
        (void*)TAPP_get_extents,
        (void*)TAPP_set_extents,
        (void*)TAPP_get_strides,
        (void*)TAPP_set_strides,
        (void*)TAPP_create_tensor_product,
        (void*)TAPP_destroy_tensor_product,
        (void*)TAPP_execute_product,
        (void*)TAPP_execute_batched_product,
        (void*)TAPP_implementation_name,
        (void*)TAPP_implementation_version,
        (void*)tcontract_pool_create_rayon,
        (void*)tcontract_pool_destroy,
        (void*)tcontract_tapp_executor_borrow_pool,
    };
    const size_t n = sizeof decls / sizeof *decls;

    int all_resolved = 1;
    for (size_t i = 0; i < n; i++) {
        if (decls[i] == NULL) {
            all_resolved = 0;
        }
    }
    /* 22 TAPP symbols plus the five non-standard extensions. */
    check(n == 27 && all_resolved, "every declaration in <tapp.h> and <tcontract/*.h> links");
}

int main(void) {
    printf("TAPP provider: %s\n", TAPP_implementation_name());

    test_version_agreement();
    test_every_declaration_links();

    TAPP_handle handle;
    TAPP_executor exec;
    must(TAPP_create_handle(&handle), "create handle");
    must(TAPP_create_executor(&exec), "create executor");

    test_f64(handle, exec);
    test_c64_conj(handle, exec);
    test_refusals(handle, exec);

    must(TAPP_destroy_executor(exec), "destroy executor");
    must(TAPP_destroy_handle(handle), "destroy handle");

    if (failures) {
        printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    printf("\nall checks passed\n");
    return 0;
}
