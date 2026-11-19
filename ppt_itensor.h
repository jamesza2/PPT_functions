#ifndef PPT_ITENSOR_HEADER
#define PPT_ITENSOR_HEADER

#include "itensor/all.h"

#include <ctime>
#include <cmath>
#include <random>
#include <iomanip>

using vMPS = itensor::MPS; //Renaming the MPS to this because the "MPS" we're dealing with is actually a vectorized MPO
using Tensor = itensor::ITensor;
using Ind = itensor::Index;
using Inds = itensor::IndexSet;

//Returns a single index tensor for tracing out a DM-physical index.
Tensor tracer(const Ind &i){
	assert(itensor::dim(i) == 4);
	Tensor t(i);
	t.set(i=1,1); t.set(i=4,1);
	return t;
}

//Returns a tensor with indices (i,i') for transposing a DM-physical index.
Tensor transposer(const Ind &i){
	assert(itensor::dim(i) == 4);
	Ind ip = itensor::prime(i);
	Tensor t(i, ip);
	t.set(i=1,ip=1,1); t.set(i=4,ip=4,1);
	t.set(i=2,ip=3,1); t.set(i=3,ip=2,1);
	return t;
}

//Get the "partial trace vector" of the section of rho up to (but not including) site_i, 
//giving a left/right environment tensor with a single link index.
//Essentially performs the trace of rho but stops before reaching site_i.
//from_left determines if we get a left environment or a right environment.
Tensor partial_trace_decomposition(const vMPS &rho, size_t site_i, bool from_left){
	Tensor pt(1);
	if(from_left){
		for(size_t site = 1; site < site_i; site++){
			pt *= rho(site);
			pt *= tracer(itensor::siteIndex(rho,site));
		}
	}
	else{
		for(size_t site = itensor::length(rho); site > site_i; site--){
			pt *= rho(site);
			pt *= tracer(itensor::siteIndex(rho,site));
		}
	}
	return pt;
}

//Finds the trace of the vectorized density matrix rho.
std::complex<double> vmps_trace(const vMPS &rho){
	Tensor left_cap(1);
	for(size_t site = 1; site <= itensor::length(rho); site++){
		left_cap *= rho(site);
		left_cap *= tracer(itensor::siteIndex(rho,site));
	}
	return itensor::eltC(left_cap);
}

//Finds the purity of the vectorized density matrix rho.
double vmps_purity(const vMPS &rho){
	Tensor left_cap(1);
	for(size_t site = 1; site <= itensor::length(rho); site++){
		left_cap *= rho(site);
		left_cap *= transposer(itensor::siteIndex(rho,site));
		left_cap.prime(-1,"Site");
		left_cap *= itensor::prime(rho(site), "Link"); //We don't conjugate because it's a density matrix tensor
	}
	return itensor::norm(left_cap);
}

//Applies the 2-site channel gate_tensor to the vectorized MPO rho at the sites (mps_site, mps_site+1).
//Performs PPT if the bond dimension requires truncation.
//gate_tensor is a channel acting on 2 sites, so it has two pairs of vectorized DM-physical indices of dimension 4.
//The indices of the gate tensor should be in the form (g1, g2, g1', g2') where g1 has the tag "i=1", g2 has the tag "i=2", and all
//indices have dimension 4.
double Apply2SiteGatePurity(vMPS &rho, const Tensor &gate_tensor, double cutoff, size_t max_bd, size_t mps_site){
	Ind gate_site_i = itensor::noPrime(itensor::findInds(gate_tensor,"i=1")(1));
	Ind gate_site_j = itensor::noPrime(itensor::findInds(gate_tensor,"i=2")(1));
	assert(itensor::hasIndex(gate_tensor,gate_i)); assert(itensor::hasIndex(gate_tensor,gate_j));
	assert(itensor::hasIndex(gate_tensor,itensor::prime(gate_i))); assert(itensor::hasIndex(gate_tensor,itensor::prime(gate_j)));
	assert(itensor::dim(gate_site_i) == 4); assert(itensor::dim(gate_site_j) == 4);

	//Technically the "canonical form" of a vectorized density matrix doesn't mean much, but this should make sure the
	//purity double-derivative matrix is the identity, if the density matrix is Hermitian.
	rho.position(mps_site); 

	Ind rho_site_i = itensor::siteIndex(rho, mps_site);
	Ind rho_site_j = itensor::siteIndex(rho, mps_site+1);

	//Create the two-site tensor
	Tensor two_site_tensor = rho(mps_site)*rho(mps_site+1);
	two_site_tensor *= itensor::delta(rho_site_i, gate_site_i);
	two_site_tensor *= itensor::delta(rho_site_j, gate_site_j);
	two_site_tensor *= gate_tensor;
	two_site_tensor *= itensor::delta(rho_site_i, itensor::prime(gate_site_i));
	two_site_tensor *= itensor::delta(rho_site_j, itensor::prime(gate_site_j));

	//Perform truncated SVD
	//S is the original truncated singular values vector Lambda' (in the form of a diagonal matrix)
	Ind rho_link_i = itensor::leftLinkIndex(rho, mps_site);
	auto [U,S,V] = itensor::svd(two_site_tensor, {rho_link_i, rho_site_i}, {"Cutoff", cutoff, "MaxDim", max_bd, "LeftTags", itensor::tags(itensor::rightLinkIndex(rho,mps_site))});
	Ind US = itensor::commonIndex(U,S); Ind VS = itensor::commonIndex(V,S);
	Tensor new_S = S; //new_S will become the PPT-adjusted singular values if truncation happened

	if(itensor::dim(US) >= max_bd){ //Indicates that truncation happened
		//Partial trace up to the edges of the two-site tensor
		Tensor lower_partial_trace = partial_trace_decomposition(rho, mps_site, true);
		Tensor upper_partial_trace = partial_trace_decomposition(rho, mps_site+1, false);

		//Find the truncated trace vector T'
		Tensor lower_trace_vector = lower_partial_trace*(U*tracer(rho_site_i));
		Tensor upper_trace_vector = upper_partial_trace*(V*tracer(rho_site_j));
		
		std::vector<double> trace_vector_data(itensor::dim(US), 0);
		for(size_t tvi = 1; tvi <= itensor::dim(US); tvi++){
			std::complex<double> T1 = lower_trace_vector.eltC(US=tvi);
			std::complex<double> T2 = upper_trace_vector.eltC(VS=tvi);
			trace_vector_data[tvi-1] = (T1*T2).real();
		}
		//Trace vector is the element-wise product of the above vectors. Although it has 2 indices, it's a diagonal matrix.
		Tensor trace_vector = itensor::diagITensor(trace_vector_data, US, VS);

		//Get original trace and purity
		Tensor original_trace_tensor = lower_partial_trace;
		original_trace_tensor *= two_site_tensor*itensor::delta(rho_site_i, itensor::prime(rho_site_i));
		original_trace_tensor *= itensor::delta(rho_site_j, itensor::prime(rho_site_j));
		original_trace_tensor *= upper_partial_trace;

		double original_trace = itensor::norm(original_trace_tensor); //Original trace
		double purity = vmps_purity(rho);
		double lnorm = std::sqrt(purity); //Norm of untruncated singular values Lambda
		double truncated_lnorm = itensor::norm(S); //Norm of truncated singular values Lambda'

		double truncated_trace = vmps_trace(rho).real(); //Original truncated trace
		double truncated_trace_norm = itensor::norm(trace_vector);

		//Get cosines/tangents of target angle theta, 
		double costheta = original_trace/(truncated_trace_norm*lnorm); //theta=desired angle between singular values and truncated trace vector
		double cossigma = truncated_trace/(truncated_trace_norm*truncated_lnorm); //sigma=current angle between singular values and truncated trace vector
		double tantheta = 0;
		if(costheta > 1){ //If cos(theta) > 1, we are in the uncorrectable regime, so the best we can do is project directly onto T'
			std::cout << "WARNING: Desired cos(theta) of " << costheta << "uncorrectable." << std::endl; //Turn off if unneccessary
			costheta=1;
		}
		else{
			tantheta = std::sqrt(1-costheta*costheta)/costheta;
		}
		double tansigma = std::sqrt(1-cossigma*cossigma)/cossigma; //cossigma should never be above 1

		Tensor orthogonal_component = S - trace_vector*(truncated_lnorm/truncated_trace_norm)*cossigma;
		new_S = S + orthogonal_component*(tantheta/tansigma-1); 
	}
	rho.set(mps_site, U);
	rho.set(mps_site+1, new_S*V);

	return itensor::norm(new_S-S); //PPT error is interpreted as the difference between truncated and PPT-adjusted singular values
}


#endif