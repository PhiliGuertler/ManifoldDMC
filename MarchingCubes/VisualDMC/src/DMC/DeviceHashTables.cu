#include "DeviceHashTables.h"

namespace DMC {

	template <typename T>
	__global__ void fill(T* begin, T* end, T value) {
		LazyEngine::CUDAInfo info;
		auto threadId = info.getGlobalThreadId();
		size_t size = end - begin;
		if (threadId >= size) return;

		T* current = begin + threadId;
		*current = value;
	}


	// ##################################################################### //
	// ### HostHashTable ################################################### //
	// ##################################################################### //

	HostHashTable::HostHashTable(size_t maxCapacity)
		: d_keys(maxCapacity, "HostHashTable-Keys")
	{
		// empty
	}

	void HostHashTable::initialize() {
		thrust::fill(d_keys.begin(), d_keys.end(), HashTable::EMPTY_BUCKET);
	}


	// ##################################################################### //
	// ### QuadrilateralDeviceHashTable #################################### //
	// ##################################################################### //

	QuadrilateralDeviceHashTable::QuadrilateralDeviceHashTable(const thrust::device_vector<size_t>& keys, const thrust::device_vector<glm::ivec4>& quads, const thrust::device_vector<int>& quadColors)
		: m_quads(keys, quads)
		, m_quadColors(keys, quadColors)
	{
		assert(quads.size() == quadColors.size());
	}


	// ##################################################################### //
	// ### QuadrilateralHostHashTable ###################################### //
	// ##################################################################### //

	QuadrilateralHostHashTable::QuadrilateralHostHashTable(size_t maxCapacity)
		: HostHashTable(maxCapacity)
		, d_quads(maxCapacity, "QuadrilateralHostHashTable-Quads")
		, d_quadColors(maxCapacity, "QuadrilateralHostHashTable-QuadColors")
		, m_deviceTable(nullptr)
	{
		m_deviceTable = std::unique_ptr<QuadrilateralDeviceHashTable>(new QuadrilateralDeviceHashTable(d_keys, d_quads, d_quadColors));
	}

	void QuadrilateralHostHashTable::initialize() {
		HostHashTable::initialize();
		const glm::ivec4 invalidQuad = glm::ivec4(QuadrilateralDeviceHashTable::INVALID_INDEX);
		thrust::fill(d_quads.begin(), d_quads.end(), invalidQuad);
		thrust::fill(d_quadColors.begin(), d_quadColors.end(), QuadrilateralDeviceHashTable::INVALID_COLOR);
	}

}