#ifndef PPT_BASICS_HEADER
#define PPT_BASICS_HEADER

#include "../util/types_and_headers.h"
#include "../models/sites/dm_spinarb.h"
#include "../dtensor/dtensor_index.h"
#include "../dtensor/dtensor.h"
#include "../dtensor/dtensor_op.h"
#include "../density/density_matrix_util.h"
#include "../density/trotter.h"
#include "../density/vectorizer.h"
#include "../densitygates/density_gates.h"
#include "../mps/tt.h"
#include "../mps/ptt.h"
#include "density_matrix_util.h"

using cplx = std::complex<double>;

//Applies a 2-site gate (in the form of the gate_tensors vector) to the vectorized density matrix rho, truncating according to given parameters
//while trying to maintain the purity of the density matrix.
//max_bd = maximum bond dimension.
//cutoff = minimum untruncated singular value.
//mps_site = leftmost site to apply the 2-site gate to.
double Apply2SiteGatePurity(MPS<cplx> &rho, vector<dtensor<cplx>> &gate_tensors, double cutoff, size_t max_bd, size_t mps_site=0, size_t gate_site=0);

//The following are utility functions called by Apply2SiteGatePurity.
//Most of these can technically be defined over doubles instead of cplx, though Apply2SiteGatePurity only uses the cplx version.

//Finds the partial trace from 0 to site_i ( if from_left) or site_i+1 to N (if from right), not including site_i.
//Stores result in the trace_derivative tensor.
//The partial trace is what you get if you try to compute the trace of a vectorized DM, but stop right before site_i.
template<typename T>
void partial_trace_decomposition(MPS<T> &rho, size_t site_i, dtensor<T> &trace_derivative, bool from_left);

//Finds the partial purity from 0 to site_i (if from_left) or site_i+1 to N (if from right), not including site_i.
//Stores result in the purity_derivative tensor.
//The partial purity is what you get if you try to compute the purity of a vectorized DM through a MPS contraction, but stop the contraction right before site_i.
//For a Hermitian DM whose canonical center is at site_i or beyond, this should be the identity matrix on the bond dimension to the left of site_i.
template<typename T>
void partial_purity_decomposition(MPS<T> &rho, size_t site_i, dtensor<T> &purity_derivative, bool from_left);


//Transposes the original physical indices on the vectorized DM-physical index in inds[0],
//i.e. tspos becomes the matrix [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]] and gets indices (inds[0],inds[1]).
template <typename T>
void transposer(dtensor<T> &tspos, vector<dtensor_index> inds);


//Utility functions that can replace the index objects on a tensor.
template<typename T>
void ReplaceIndex(dtensor<T> &tens, dtensor_index &original_ind, dtensor_index &new_ind);

template<typename T>
void ReplaceInds(dtensor<T> &tens, vector<dtensor_index> &original_inds, vector<dtensor_index> &new_inds);

template<typename T>
void ReplaceInds(dtensor<T> &tens, vector<dtensor_index> &new_inds);


#endif