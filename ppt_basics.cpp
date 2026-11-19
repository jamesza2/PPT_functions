
#include "ppt_basics.h"


template<typename T>
void transposer(dtensor<T> &tspos, vector<dtensor_index> inds){
	assert(inds.size() == 2); assert(inds[0].size() == inds[1].size());
	int rank;
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  	
	int real_pd = phy_sqrt(inds[0].size());
	assert(real_pd != 0);
	//dtensor_index indprime(inds[0]); indprime.prime(); inds.push_back(indprime);
	//dtensor<T> tspos(inds);
	tspos.reset(inds);
	vector<T> data(inds[0].size()*inds[0].size(), 0); vector<long int> indices(inds[0].size()*inds[0].size(), 0);
	int collapsed_index = 0;
	for(int p1 = 0; p1 < real_pd; p1++){
		for(int p2 = 0; p2 < real_pd; p2++){
			for(int p1p = 0; p1p < real_pd; p1p++){
				for(int p2p = 0; p2p < real_pd; p2p++){
					if((p2 == p1p) && (p1 == p2p)){
						data[collapsed_index] = 1;
					}
					indices[collapsed_index] = collapsed_index;
					collapsed_index += 1;
				}
			}
		}
	}
	tspos.__T.write((rank) ? 0 : inds[0].size()*inds[0].size(), indices.data(), data.data());
	//return tspos;
}
template void transposer(dtensor<double> &tspos, vector<dtensor_index> inds);
template void transposer(dtensor<cplx> &tspos, vector<dtensor_index> inds);

template<typename T>
void partial_trace_decomposition(MPS<T> &rho, size_t site_i, dtensor<T> &trace_derivative, bool from_left){
	int rank;
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	unsigned length = rho.length;
	unsigned d = rho.phy_dim;
	unsigned phy_dim = static_cast<unsigned>(sqrt(d));
	assert(phy_dim*phy_dim == d);
	assert(site_i < rho.length);

	dtensor_index original_site = rho.A[0].findIndex(Site, 0);
	dtensor<T> contractor = tracer(original_site, T(1));


	if(from_left){//Start finding the density matrix from the left, then stop before site i
		if(site_i == 0){//If site_i==0, trace derivative is nothing
			trace_derivative = dtensor<T>({rho.A[0].findIndex(Link, 0)});
			trace_derivative.setOne();
			return;
		}
		//Left environment we build up during tracing
		dtensor<T> cap({rho.A[0].findIndex(Link, 0)}); cap.setOne(); 
		for(size_t site = 0; site < site_i; site++){
			ReplaceIndex(contractor, contractor.idx_set[0], rho.A[site].findIndex(Site, site));
			dtensor<T> t1;
			t1 = std::move(rho.A[site]*contractor);
			cap = std::move(cap*t1);
		}
		trace_derivative = cap;
	}
	else{//Start finding the density matrix from the right, then stop before reaching site i
		if(site_i == rho.length-1){
			trace_derivative = dtensor<T>({rho.A[length-1].findIndex(Link, length)});
			trace_derivative.setOne();
			return;
		}
		//Right environment we build up during tracing
		dtensor<T> cap({rho.A[length-1].findIndex(Link, length)}); cap.setOne();
		for(size_t site = rho.length-1; site > site_i; site--){
			ReplaceIndex(contractor, contractor.idx_set[0], rho.A[site].findIndex(Site, site));
			dtensor<T> t1;
			t1 = std::move(rho.A[site]*contractor);
			cap = std::move(cap*t1);
		}
		trace_derivative = cap;
	}
}
template void partial_trace_decomposition(MPS<double> &rho, size_t site_i, dtensor<double> &trace_derivative, bool from_left);
template void partial_trace_decomposition(MPS<cplx> &rho, size_t site_i, dtensor<cplx> &trace_derivative, bool from_left);


template<typename T>
void partial_purity_decomposition(MPS<T> &rho, size_t site_i, dtensor<T> &purity_derivative, bool from_left){
	int rank;
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	unsigned length = rho.length;
	unsigned d = rho.phy_dim;
	unsigned phy_dim = static_cast<unsigned>(sqrt(d));
	assert(phy_dim*phy_dim == d);
	assert(site_i < rho.length);

	//Prepare the left and right cap tensors - the algorithm will pick one of them and calculate the DM purity until they reach that particular site
	dtensor<T> tspos;
	dtensor_index llink = rho.A[0].findIndex(Link, 0); dtensor_index llinkp = llink; llinkp.prime();
	dtensor_index rlink = rho.A[length-1].findIndex(Link, length); dtensor_index rlinkp = rlink; rlinkp.prime();
	dtensor<T> left_cap({llink, llinkp}), right_cap({rlink, rlinkp});
	left_cap.setOne(); right_cap.setOne();
	
	if(from_left){
		for(size_t site = 0; site < site_i; site++){
			dtensor<T> t1, t2, t3, t4;
			t1 = rho.A[site];
			t2 = t1; t1.prime();
			//For a vectorized operator O, the purity tr(O^2) isn't just the squared MPS norm as each doubled physical index needs to be swapped.
			//If there's a guarantee that the MPO corresponding to the density matrix stays Hermitian despite truncation, this might not be strictly necessary,
			//but it's not an expensive operation anyways.
			transposer(tspos, {t1.findIndex(Site, site), t2.findIndex(Site, site)});
			t3 = std::move(t1*tspos);
			t4 = std::move(left_cap*t3);
			left_cap = std::move(t2*t4);
		}
		purity_derivative = left_cap;
	}
	else{
		for(size_t site = rho.length-1; site > site_i; site--){
			dtensor<T> t1, t2, t3, t4;
			t1 = rho.A[site];
			t2 = t1; t1.prime();
			transposer(tspos, {t1.findIndex(Site, site), t2.findIndex(Site, site)});
			t3 = std::move(t1*tspos);
			t4 = std::move(right_cap*t3);
			right_cap = std::move(t2*t4);
		}
		purity_derivative = right_cap;
	}
}
template void partial_purity_decomposition(MPS<double> &rho, size_t site_i, dtensor<double> &purity_derivative, bool from_left);
template void partial_purity_decomposition(MPS<cplx> &rho, size_t site_i, dtensor<cplx> &purity_derivative, bool from_left);

//Applies a 2-site channel (in the form of the gate_tensors vector) to the vectorized density matrix rho, truncating according to given parameters
double Apply2SiteGatePurity(MPS<cplx> &rho, vector<dtensor<cplx>> &gate_tensors, double cutoff, size_t max_bd, size_t mps_site, size_t gate_site){
	assert(mps_site+gate_tensors.size() <= rho.length);
	//Put density matrix pseudo-canonical center near the 2-site gate 
	//(not a real canonical center but should still help keep truncation process more accurate)
	if(rho.center != static_cast<int>(mps_site)) rho.position(mps_site);
	int64_t max_bd_encountered = 0;
	double purity_adjustment_error = 0;

	//Set up timers 
	Timer initial_combo_time("Initial Combining time"); initial_combo_time.Start();
	Timer initial_svd_time("Initial SVD time");
	Timer trace_calc_time("Trace calculation time");
	Timer purity_calc_time("Purity calculation time");
	Timer purity_redundant_time("Purity redundant time");
	Timer pp_algorithm_time("Purity-Preserving algorithm time");
	Timer final_allocation_time("Final Combining time");
	Timer step_time("Total time"); step_time.Start();

	dtensor_index rho_index = rho.A[mps_site].findIndex(Site);
	dtensor_index gate_index = gate_tensors[0].findIndex(Site,-1,0);
	dtensor_index rho_index_right = rho.A[mps_site+1].findIndex(Site);
	dtensor_index gate_index_right = gate_tensors[1].findIndex(Site,-1,0);

	//Prepare the two-site product A_i*A_(i+1)*G_i*G_(i+1)
	dtensor<cplx> two_site_product = std::move(rho.A[mps_site]*rho.A[mps_site+1]);
	dtensor<cplx> left_gate_tensor = gate_tensors[0]; ReplaceIndex(left_gate_tensor, gate_index, rho_index);
	dtensor<cplx> right_gate_tensor = gate_tensors[1]; ReplaceIndex(right_gate_tensor, gate_index_right, rho_index_right);
	two_site_product = std::move(two_site_product*left_gate_tensor);
	two_site_product = std::move(two_site_product*right_gate_tensor);
	two_site_product.primeSite(-1); //Make sure the 2-site product has the original physical indices
	dtensor_index rip = rho_index; rip.prime(); dtensor_index rirp = rho_index_right; rirp.prime();
	ReplaceIndex(two_site_product, gate_index, rip);
	ReplaceIndex(two_site_product, gate_index_right, rirp);
	two_site_product.primeSite(-1);
	//Link indices on the MPS which are to the left and right of the two-site product
	dtensor_index left_rho_link = rho.A[mps_site].findIndex(Link, mps_site);
	dtensor_index right_rho_link = rho.A[mps_site+1].findIndex(Link, mps_site+2);

	//Perform truncated SVD on the two site product
	//S tensor is treated as a single index tensor, unlike in ITensor
	vector<dtensor_index> U_sites({left_rho_link, rho_index});
	vector<dtensor_index> V_sites({right_rho_link, rho_index_right});
	dtensor<cplx> U,S,V; 
	initial_combo_time.Stop(); initial_svd_time.Start();
	//svd method, as defined here, performs svd on two_site_product and stores output in U, V and S
	//S tensor is treated as a single index tensor, unlike in ITensor
	//NoMovement just means S isn't automatically applied to U or V 
	svd(two_site_product, U_sites, V_sites, U, V, S, NoMovement, cutoff, max_bd);
	
	dtensor_index mid = rho.A[mps_site].findIndex(Link, mps_site+1);
	mid.resize(S.idx_set[0].size());
	U.idx_set.back() = mid;
	V.idx_set[0] = mid;
	S.idx_set[0] = mid;
	initial_svd_time.Stop();

	dtensor<cplx> left_trace, right_trace; //Partial traces, in the form of single bond index tensors
	
	//Main bulk of PPT algorithm
	if(S.idx_set[0].size() >= max_bd){
		trace_calc_time.Start(); 
		//Compute the trace and purity beforehand for purity-preserving truncation
		partial_trace_decomposition(rho, mps_site, left_trace, true);
		partial_trace_decomposition(rho, mps_site+1, right_trace, false);

		//Compute the trace: trace_tensor should become a zero-index tensor containing the trace as its only value
		dtensor<cplx> contractor = tracer(rho_index, cplx(1));  //Tracer for the physical index at mps_site
		dtensor<cplx> trace_tensor = std::move(two_site_product*left_trace);
		trace_tensor = std::move(trace_tensor*right_trace);
		trace_tensor = std::move(trace_tensor*contractor);
		ReplaceIndex(contractor, rho_index, rho_index_right);
		trace_tensor = std::move(trace_tensor*contractor);
		
		trace_calc_time.Stop(); 
		
		//Computing the purity beforehand
		purity_redundant_time.Start();

		dtensor_index lrlp = left_rho_link; lrlp.prime();
		dtensor_index rrlp = right_rho_link; rrlp.prime();

		//Compute the partial purities
		dtensor<cplx> left_purity_cap, right_purity_cap;
		partial_purity_decomposition(rho, mps_site, left_purity_cap, true);
		partial_purity_decomposition(rho, mps_site+1, right_purity_cap, false);

		//Use the partial purities to calculate the full purity of the system, stored in the single-valued tensor purity_estimate
		//Full purity is used to determine |Lambda|, which in turn gives |T|cos(phi) with the full trace
		purity_redundant_time.Stop(); purity_calc_time.Start();
		dtensor_index left_product_index = two_site_product.commonIndex(left_gate_tensor); dtensor_index lpip = left_product_index; lpip.prime();
		dtensor_index right_product_index = two_site_product.commonIndex(right_gate_tensor); dtensor_index rpip = right_product_index; rpip.prime();
		dtensor<cplx> left_transposer; transposer(left_transposer, {left_product_index, lpip});
		dtensor<cplx> right_transposer; transposer(right_transposer, {right_product_index, rpip});
		dtensor<cplx> two_site_product_transposed = std::move(two_site_product*left_transposer); 
		two_site_product_transposed = std::move(two_site_product_transposed*right_transposer); 
		two_site_product_transposed.primeSite(-1); 
		two_site_product_transposed = std::move(two_site_product_transposed*left_purity_cap); two_site_product_transposed.findIndex(Link, mps_site, 1).prime(-1);
		two_site_product_transposed = std::move(two_site_product_transposed*right_purity_cap); two_site_product_transposed.findIndex(Link, mps_site+2, 1).prime(-1);
		dtensor<cplx> purity_estimate = two_site_product*two_site_product_transposed;
		
		purity_calc_time.Stop(); pp_algorithm_time.Start();

		//Compute the truncated trace vector T'_i
		ReplaceIndex(contractor, contractor.idx_set[0], rho_index);
		dtensor<cplx> leftU_trace = std::move(left_trace*U); leftU_trace = std::move(leftU_trace*contractor);
		ReplaceIndex(contractor, contractor.idx_set[0], rho_index_right);
		dtensor<cplx> rightV_trace = std::move(V*right_trace); rightV_trace = std::move(rightV_trace*contractor);
		dtensor<cplx> new_trace_vector({mid}); new_trace_vector.__T["i"] = leftU_trace.__T["i"]*rightV_trace.__T["i"]; //Trace vector

		dtensor<cplx> new_trace = new_trace_vector*S;

		double original_trace = trace_tensor.norm();
		if(original_trace > new_trace_vector.norm()*std::sqrt(purity_estimate.norm())){ //Indicates we're in the uncorrectable regime.
			perr << "WARNING: Cannot find singular values to keep purity of " << purity_estimate.norm() << " preserved" << endl;
			original_trace = std::real(trace(rho));
			perr << "USING ORIGINAL TRACE OF " << original_trace << endl;
			pout << "TRACE WARNING, Trace tensor: " << endl; trace_tensor.print(1);
		}

		//Target cosine = the desired overlap cos(theta) between the singular values and the truncated trace vector
		//Equal to the ratio between |T|cos(phi) = (original trace)/|Original Lambda| and the norm of the truncated trace vector
		double target_cosine = original_trace/(new_trace_vector.norm()*std::sqrt(purity_estimate.norm()));
		//Current cosine = the current overlap cos(sigma) between the truncated singular values and truncated trace vector
		double current_cosine = new_trace.norm()/(new_trace_vector.norm()*S.norm());
		//If target cos(theta) is greater than 1, we're in the uncorrectable regime, and theta=0 is the best we can do
		double target_tan = (target_cosine >= 1) ? 0 : std::sqrt(1-target_cosine*target_cosine)/target_cosine;
		double current_tan = std::sqrt(1-current_cosine*current_cosine)/current_cosine;

		//Compute the orthogonal component
		//This is proportional to the adjustment we need to make to the truncated singular values to 
		//project it to the cone of constant angle theta around T'.
		//This adjustment is orthogonal to the original truncated singular values.
		dtensor<cplx> ortho_comp({mid});
		ortho_comp.__T["i"] = S.__T["i"] - (new_trace_vector.__T["i"])*(new_trace.norm()/std::pow(new_trace_vector.norm(), 2));
		dtensor<cplx> ortho_test = ortho_comp*new_trace_vector;
		
		
		//New singular value adjustment = (tan(theta)/tan(sigma)-1)*orthogonal component
		//Stored in the tensor new_svals
		dtensor<cplx> new_svals({mid}); new_svals.__T["i"] = S.__T["i"] + (target_tan/current_tan - 1.)*ortho_comp.__T["i"];
		dtensor<cplx> ssprod = std::move(new_svals*S);

		new_trace = new_trace_vector*new_svals;
		//Compute the purity adjustment error - roughly the norm of (truncated singular values) - (new singular values)
		double purity_adjustment_cosine = ssprod.norm()/(S.norm()*new_svals.norm());
		purity_adjustment_error = std::sqrt(std::max(0.,1-purity_adjustment_cosine*purity_adjustment_cosine));
		perr << "Purity adjustment error: " << purity_adjustment_error << endl;
		S = new_svals;

		pp_algorithm_time.Stop();
	}

	final_allocation_time.Start();
	//Now that we've calculated the proper singular values through PPT, convert S to a diagonal matrix so it can be applied to V
	dtensor_index midp = mid; midp.prime(); dtensor<cplx> new_svals_tensor({mid, midp});
	new_svals_tensor.__T["ii"] = S.__T["i"];
	V = std::move(new_svals_tensor*V); ReplaceIndex(V, midp, mid);

	//Update the density matrix tensors
	rho.A[mps_site] = U;
	rho.A[mps_site+1] = V;
	rho.bond_dims[mps_site+1] = static_cast<unsigned>(mid.size());

	if(rho.center == static_cast<int>(mps_site)) rho.center += 1;

	final_allocation_time.Stop(); step_time.Stop(); double total_time = step_time.Time();
	
	return purity_adjustment_error;
}


template<typename T>
void ReplaceIndex(dtensor<T> &tens, dtensor_index &original_ind, dtensor_index &new_ind){
	if(original_ind == new_ind){return;}
	bool found_original = false;
	for(auto &i : tens.idx_set){found_original = found_original || (original_ind == i);}
	assert(found_original);
	dtensor<T> converter;
	Delta(converter, original_ind, new_ind); //Similar function to itensor::delta but stores the tensor in the 1st argument.
	tens = std::move(tens*converter);
}
template void ReplaceIndex(dtensor<double> &tens, dtensor_index &original_ind, dtensor_index &new_ind);
template void ReplaceIndex(dtensor<cplx> &tens, dtensor_index &original_ind, dtensor_index &new_ind);

template<typename T>
void ReplaceInds(dtensor<T> &tens, vector<dtensor_index> &original_inds, vector<dtensor_index> &new_inds){
  assert(new_inds.size() == original_inds.size());
  for(size_t i = 0; i < original_inds.size(); i++){
    assert(original_inds[i].size() == new_inds[i].size());
    ReplaceIndex(tens, original_inds[i], new_inds[i]);
  }
}
template void ReplaceInds(dtensor<double> &tens, vector<dtensor_index> &original_inds, vector<dtensor_index> &new_inds);
template void ReplaceInds(dtensor<cplx> &tens, vector<dtensor_index> &original_inds, vector<dtensor_index> &new_inds);

//Replaces entire original inds
template<typename T>
void ReplaceInds(dtensor<T> &tens, vector<dtensor_index> &new_inds){
  vector<dtensor_index> original_inds(tens.idx_set);
  ReplaceInds(tens, original_inds, new_inds);
}
template void ReplaceInds(dtensor<double> &tens, vector<dtensor_index> &new_inds);
template void ReplaceInds(dtensor<cplx> &tens, vector<dtensor_index> &new_inds);