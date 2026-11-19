Code for purity-preserving truncation (PPT) on an MPS representing a vectorized density matrix. 
The main files ppt_basics.cpp, ppt_basics.h use the tensor-tools framework based on CTF (https://github.com/ClarkResearchGroup/tensor-tools). The ppt_itensor.h file uses ITensor instead.

The main function in both functions is Apply2SiteGatePurity, which applies a 2-site gate onto an MPS representing a vectorized density matrix, using purity-preserving truncation (https://doi.org/10.1103/wlcd-zrxm).
