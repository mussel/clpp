// http://code.google.com/p/back40computing/source/browse/trunk/b40c/scan/upsweep/kernel.cuh

#define WORK_LIMITS_GUARDED_OFFSET 
#define KERNELPOLICY_TILE_ELEMENTS

/**
 * Process a single, full tile
 */
inline oid ProcessFullTile(SizeT cta_offset)
{
	// Tiles of segmented scan elements and flags
	T				partials[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];
	Flag			flags[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];

	// Load tile of partials
	util::io::LoadTile<
		KernelConfig::LOG_LOADS_PER_TILE,
		KernelConfig::LOG_LOAD_VEC_SIZE,
		KernelConfig::THREADS,
		KernelConfig::READ_MODIFIER,
		true>::Invoke(						// unguarded I/O
			partials, d_partials_in + cta_offset);

	// Load tile of flags
	util::io::LoadTile<
		KernelConfig::LOG_LOADS_PER_TILE,
		KernelConfig::LOG_LOAD_VEC_SIZE,
		KernelConfig::THREADS,
		KernelConfig::READ_MODIFIER,
		true>::Invoke(						// unguarded I/O
			flags, d_flags_in + cta_offset);

	// Reduce tile with carry maintained by thread SrtsSoaDetails::CUMULATIVE_THREAD
	util::reduction::soa::CooperativeSoaTileReduction<
		SrtsSoaDetails,
		KernelConfig::LOAD_VEC_SIZE,
		KernelConfig::SoaScanOp>::template ReduceTileWithCarry<true, DataSoa>(
			srts_soa_details,
			DataSoa(partials, flags),
			carry);

	// Barrier to protect srts_soa_details before next tile
	__syncthreads();
}

inline void UpsweepPass(
	__global T* d_in,
	__global T* d_out
	)
	//util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	&work_decomposition,
	//typename KernelPolicy::SmemStorage							&smem_storage)
{
	//typedef reduction::upsweep::Cta<KernelPolicy>	Cta;
	//typedef typename KernelPolicy::SizeT 			SizeT;

	// Quit if we're the last workgroup (no need for it in upsweep)
	if (get_group_id(0) == get_num_groups(0) - 1) // == gridDim.x - 1)
		return;

	// CTA processing abstraction
	//Cta cta(smem_storage, d_in, d_out);

	// Determine our threadblock's work range
	//util::CtaWorkLimits<SizeT> work_limits;
	//work_decomposition.template GetCtaWorkLimits<KernelPolicy::LOG_TILE_ELEMENTS, KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

	//---- Since we're not the last block: process at least one full tile of tile_elements
	
	//cta.template ProcessFullTile<true>(work_limits.offset);
	//work_limits_offset += KERNELPOLICY_TILE_ELEMENTS;
	
	uint work_limits_offset = ???
	work_limits_offset += KERNELPOLICY_TILE_ELEMENTS;

	//---- Process any other full tiles
	while(work_limits_offset < WORK_LIMITS_GUARDED_OFFSET)
	{
		cta.ProcessFullTile<false>(work_limits_offset);
		work_limits_offset += KERNELPOLICY_TILE_ELEMENTS;
	}

	//---- Collectively reduce accumulated carry from each thread into output
	// destination (all thread have valid reduction partials)
	cta.OutputToSpine();
}

// Upsweep reduction kernel entry point
 //template <typename KernelPolicy>
//__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
//__global__

__kernel
void kernel__upsweep(
	__global T* d_in,
	__global T* d_spine)
	
	//util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	work_decomposition)
{
	// Shared storage for the kernel
	//__local typename KernelPolicy::SmemStorage smem_storage;

	//UpsweepPass<KernelPolicy>(d_in, d_spine, work_decomposition, smem_storage);
}