#include "HalfedgeHashTables.h"

namespace DMC {

	// ##################################################################### //
	// ### HalfedgeDeviceHashTable ######################################### //
	// ##################################################################### //

	HalfedgeDeviceHashTable::HalfedgeDeviceHashTable(const thrust::device_vector<size_t>& keys, const thrust::device_vector<int>& faces, const thrust::device_vector<glm::ivec4>& twins)
		: m_numFaces(keys, faces)
		, m_halfedges(keys, twins)
	{
		assert(faces.size() == twins.size());
	}


	// ##################################################################### //
	// ### HalfedgeHostHashTable ########################################### //
	// ##################################################################### //

	HalfedgeHostHashTable::HalfedgeHostHashTable(size_t maxCapacity)
		: HostHashTable(maxCapacity)
		, d_numFaces(maxCapacity, "HalfedgeHostTable-numFaces")
		, d_halfedgeIDs(maxCapacity, "HalfedgeHostTable-halfedgeIDs")
		, m_deviceTable(nullptr)
	{
		initialize();
		m_deviceTable = std::unique_ptr<HalfedgeDeviceHashTable>(new HalfedgeDeviceHashTable(d_keys, d_numFaces, d_halfedgeIDs));
	}

	void HalfedgeHostHashTable::initialize() {
		HostHashTable::initialize();
		glm::ivec4 invalidTwins = glm::ivec4(HashTable::INVALID_INDEX);
		thrust::fill(d_numFaces.begin(), d_numFaces.end(), 0);
		thrust::fill(d_halfedgeIDs.begin(), d_halfedgeIDs.end(), invalidTwins);
	}

}