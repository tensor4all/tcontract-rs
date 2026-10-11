/* A host that includes the real dlpack.h before tcontract. */
#include "dlpack/dlpack.h"
#include "tcontract/tcontract.h"
int main(void) { DLTensor t; tcontract_tensor x = tcontract_tensor_borrow_raw(&t, DLPACK_FLAG_BITMASK_READ_ONLY); (void)x; return kDLCPU == 1 ? 0 : 1; }
