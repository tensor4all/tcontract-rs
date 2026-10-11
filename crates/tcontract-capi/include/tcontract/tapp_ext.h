/* tcontract extensions to the TAPP C API (not part of standard TAPP). Include
   <tapp.h> alone for serial contraction; this header adds the Rayon executor.

   The names are tcontract-specific. A TAPP_executor is an intptr_t owned by the
   caller: it is created here (or by TAPP_create_executor), shared by any
   number of plans of this library, and destroyed by the
   standard TAPP_destroy_executor, which stops and joins an owned pool. Handles
   of another TAPP provider cannot be mixed in. */
#ifndef TCONTRACT_TAPP_EXT_H
#define TCONTRACT_TAPP_EXT_H
#include <stddef.h>
#include "tcontract/core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { size_t stack_size; /* bytes; 0 = 16 MiB */ } tcontract_rayon_opts;

/* An executor owning a private Rayon pool of `nthreads` workers. nthreads == 0
   is TCONTRACT_ERR_INVALID_ARGUMENT; nthreads == 1 is a serial executor with no
   workers. The width is never inferred from the environment. `opts` may be
   NULL. */
TAPP_error tcontract_tapp_executor_create_rayon(TAPP_executor *out, size_t nthreads,
                                             const tcontract_rayon_opts *opts);

/* Calls started afterwards use at most `budget` threads (>= 1, clamped to the
   pool width; the pool is not resized). A call in progress keeps the budget it
   started with. Executor 0 and serial executors accept and ignore it. */
TAPP_error tcontract_tapp_executor_set_budget(TAPP_executor exec, size_t budget);

/* Pool width and budget; either pointer may be NULL. Serial executors report
   pool_size 0 and budget 1. The budget bounds a call's width; it is not the
   active width or an affinity promise. */
TAPP_error tcontract_tapp_executor_get_threads(TAPP_executor exec, size_t *pool_size,
                                            size_t *budget);

/* TAPP_destroy_executor on an owned pool: stops it, joins every worker (thread
   exit included) and frees the handle. TCONTRACT_BUSY (calls in flight) and
   TCONTRACT_ERR_WOULD_DEADLOCK (called from one of its workers) leave the handle
   live. Executor 0 is a no-op. A live handle is destroyed successfully once; the
   caller synchronizes destruction with the start of new calls. */

#ifdef __cplusplus
}
#endif
#endif
