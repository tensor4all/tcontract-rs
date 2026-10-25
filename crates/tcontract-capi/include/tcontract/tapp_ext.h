/* tcontract extensions to the TAPP C API (not part of standard TAPP). Include
   <tapp.h> alone for serial contraction; this header adds worker-pool executors and
   the host-owned pool they can share.

   The names are tcontract-specific. A TAPP_executor is an intptr_t owned by the
   caller: it is created here (or by TAPP_create_executor), shared by any
   number of plans of this library, and destroyed by the
   standard TAPP_destroy_executor, which stops an owned pool and waits for its
   workers to exit. Handles
   of another TAPP provider cannot be mixed in.

   One pool owner is the host. A host that wants one shared pool creates a
   single tcontract_pool handle here, builds one borrowed-pool executor on it,
   shares that executor across plans, and does not let
   tcontract_tapp_executor_create_rayon create a second pool. The width is never
   inferred from the environment. BLAS/LAPACK providers (cpueinsum-blas,
   tlinalg-blas) own their threads outside any executor and are not controlled
   by a TAPP_executor. */
#ifndef TCONTRACT_TAPP_EXT_H
#define TCONTRACT_TAPP_EXT_H
#include <stddef.h>
#include "tcontract/core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { size_t stack_size; /* bytes; 0 = 16 MiB */ } tcontract_rayon_opts;

/* An executor owning a private worker pool of `nthreads` workers. nthreads == 0
   is TCONTRACT_ERR_INVALID_ARGUMENT; nthreads == 1 is a serial executor with no
   workers. The width is never inferred from the environment. `opts` may be
   NULL. */
TAPP_error tcontract_tapp_executor_create_rayon(TAPP_executor *out, size_t nthreads,
                                             const tcontract_rayon_opts *opts);

/* An opaque handle to a worker pool the *host* owns: created here, lent to the
   executors below, and released by tcontract_pool_destroy. intptr_t, so 0 means
   "no pool" and is rejected wherever a live pool is required. Not
   interchangeable with a TAPP_executor. */
typedef intptr_t tcontract_pool;

/* Create a host-owned worker pool of `nthreads` workers; release it with
   tcontract_pool_destroy. nthreads == 0 is TCONTRACT_ERR_INVALID_ARGUMENT.
   Unlike tcontract_tapp_executor_create_rayon, nthreads == 1 is a real
   one-worker pool, because the handle is a pool and not an executor. `opts` may
   be NULL; `stack_size` is honoured exactly as in
   tcontract_tapp_executor_create_rayon. The width is never inferred from the
   environment. */
TAPP_error tcontract_pool_create_rayon(tcontract_pool *out, size_t nthreads,
                                       const tcontract_rayon_opts *opts);

/* Stop a pool created here, wait for its workers, then free the handle. 0 is
   TCONTRACT_ERR_INVALID_ARGUMENT. TCONTRACT_BUSY while an executor built from
   the pool is still live, and TCONTRACT_ERR_WOULD_DEADLOCK when called from one
   of the pool's own workers; both leave the handle live, so destroy the
   executor (or leave the worker) and retry. A live handle is destroyed
   successfully once; the caller synchronizes destruction with new borrows. */
TAPP_error tcontract_pool_destroy(tcontract_pool pool);

/* An executor that borrows `pool` instead of owning one: it runs on the host's
   pool, so no second pool is created. `budget` (>= 1) is the most threads a
   call may use, clamped to the pool width exactly as
   tcontract_tapp_executor_set_budget clamps it. Ownership: the pool must
   outlive every executor built from it and every call that uses such an
   executor, so destroy the executor (or every executor) with
   TAPP_destroy_executor before tcontract_pool_destroy. TAPP_destroy_executor
   releases the borrow and never stops a borrowed pool; its TCONTRACT_BUSY and
   TCONTRACT_ERR_WOULD_DEADLOCK behaviour is unchanged. Build one such executor
   per pool and share it across plans: each borrowed executor carries its own
   workspace for the shared pool, so concurrent numerical work
   belongs on one of them. */
TAPP_error tcontract_tapp_executor_borrow_pool(TAPP_executor *out, tcontract_pool pool,
                                               size_t budget);

/* Calls started afterwards use at most `budget` threads (>= 1, clamped to the
   pool width; the pool is not resized). A call in progress keeps the budget it
   started with. Executor 0 and serial executors accept and ignore it. */
TAPP_error tcontract_tapp_executor_set_budget(TAPP_executor exec, size_t budget);

/* Pool width and budget; either pointer may be NULL. Serial executors report
   pool_size 0 and budget 1. The budget bounds a call's width; it is not the
   active width or an affinity promise. */
TAPP_error tcontract_tapp_executor_get_threads(TAPP_executor exec, size_t *pool_size,
                                            size_t *budget);

/* TAPP_destroy_executor on an owned pool: stops it, waits for every worker (thread
   exit included) and frees the handle. On a borrowed-pool executor it releases
   the borrow only and never touches the host's pool. TCONTRACT_BUSY (calls in
   flight) and TCONTRACT_ERR_WOULD_DEADLOCK (called from one of its workers)
   leave the handle live. Executor 0 is a no-op. A live handle is destroyed
   successfully once; the caller synchronizes destruction with the start of new
   calls. */

#ifdef __cplusplus
}
#endif
#endif
