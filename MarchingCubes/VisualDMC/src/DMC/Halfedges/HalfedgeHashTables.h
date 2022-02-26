#pragma once

#include <LazyEngine/LazyEngine.h>
#include "Halfedge.h"

#include "../DeviceHashTables.h"
#include "../Utils/Utilities.h"

namespace DMC {

	// Forward-declarations
	class HalfedgeHostHashTable;
	class EdgeHostHashTable;

	// ##################################################################### //
	// ### HalfedgeHashTable ############################################### //
	// ##################################################################### //

	/**
	 *
	 */
	class HalfedgeDeviceHashTable {
	public:
		/**
		 *	returns the capacity of this hashtable (in number of elements)
		 */
		__device__ inline size_t capacity() const {
			// any capacity function is fine, just use the one of the colors
			return m_numFaces.capacity();
		}

		/**
		 *	If true, there is no element stored at this index in the hashtable
		 */
		__device__ inline bool isBucketEmpty(const size_t index) const {
			// any bucketIsEmpty function is fine, just use the one of the colors
			return m_numFaces.isBucketEmpty(index);
		}

		/**
		 *	Adds a Halfedge to the hash table, that connects two vertices.
		 *	@param vertex0: The first vertex
		 *	@param vertex1: The second vertex
		 *	@param halfedge: The halfedge
		 */
		__device__ inline bool addHalfedge(VertexID vertex0, VertexID vertex1, HalfedgeID halfedge) {
			size_t key = concatenateKeys(vertex0, vertex1);
			Insertion insertion = m_numFaces.insert(key);

			if (insertion.insertionSuccessful) {
				int offset = atomicAdd(&m_numFaces[insertion.hashIndex], 1);
#ifdef LAZYENGINE_DEBUG
				if (offset >= 4) {
					printf("Tha heck! Seems like there are %d halfedges on a single edge!\n", offset+1);
				}
				else {
#endif
					m_halfedges[insertion.hashIndex][offset] = halfedge;
#ifdef LAZYENGINE_DEBUG
				}
#endif
			}

			return insertion.insertionSuccessful;
		}

		__device__ inline int& getNumFaces(const size_t index) {
			return m_numFaces[index];
		}

		__device__ inline glm::ivec4& getHalfedges(const size_t index) {
			return m_halfedges[index];
		}

		__device__ inline size_t indexFromVertexIDs(VertexID vertex0, VertexID vertex1) const {
			size_t key = concatenateKeys(vertex0, vertex1);
			return m_halfedges.indexFromKey(key);
		}

		__device__ inline size_t stepIndex(size_t index, size_t iteration) const {
			return m_halfedges.stepIndex(index, iteration);
		}

		__device__ inline int getNumFaces(VertexID vertex0, VertexID vertex1) const {
			size_t key = concatenateKeys(vertex0, vertex1);
			size_t index = indexFromVertexIDs(vertex0, vertex1);
			// step over the hashTable until the key matches
			size_t currentKey = m_numFaces.getKeyAt(index);
			for (int i = 0; i < HashTable::MAX_ITERATIONS; ++i) {
				if (currentKey == key) {
					return m_numFaces.get(index);
				}
				// This bucket contains a different edge, move on to the next bucket
				index = stepIndex(index, i+1);
				currentKey = m_numFaces.getKeyAt(index);
			}
			// No Edge containing vertex0 and vertex1 exists!
			return -1;
		}

		__device__ inline glm::ivec4 getHalfedges(VertexID vertex0, VertexID vertex1) const {
			size_t key = concatenateKeys(vertex0, vertex1);
			size_t index = indexFromVertexIDs(vertex0, vertex1);
			// step over the hashTable until the key matches
			size_t currentKey = m_halfedges.getKeyAt(index);
			for (int i = 0; i < HashTable::MAX_ITERATIONS; ++i) {
				if (currentKey == key) {
					return m_halfedges.get(index);
				}
				// This bucket contains a different edge, move on to the next bucket
				index = stepIndex(index, i + 1);
				currentKey = m_halfedges.getKeyAt(index);
			}
			// No Edge containing vertex0 and vertex1 exists!
			return { -1, -1, -1, -1 };
		}

	protected:
		friend class HalfedgeHostHashTable;

		HalfedgeDeviceHashTable(const thrust::device_vector<size_t>& keys, const thrust::device_vector<int>& faces, const thrust::device_vector<glm::ivec4>& twins);

	protected:
		DeviceHashTable<int> m_numFaces;
		DeviceHashTable<glm::ivec4> m_halfedges;
	};

	/**
	 *
	 */
	class HalfedgeHostHashTable : public HostHashTable {
	public:
		HalfedgeHostHashTable(size_t maxCapacity);

		virtual void initialize() override;

		operator HalfedgeDeviceHashTable() {
			assert(m_deviceTable != nullptr);
			return *m_deviceTable;
		}

		inline LazyEngine::DataView<glm::ivec4> getDataView() const {
			return d_halfedgeIDs;
		}

		inline LazyEngine::DataView<int> getNumFacesDataView() const {
			return d_numFaces;
		}

	protected:
		// Stores the amount of faces connected to a halfedge-twin entry
		MonitoredThrustBuffer<int> d_numFaces;
		// Stores the halfedge-ids for a pair of vertices. Because of non-manifold cases,
		// a maximum of 4 halfedges might be connecting two vertices
		MonitoredThrustBuffer<glm::ivec4> d_halfedgeIDs;

		std::unique_ptr<HalfedgeDeviceHashTable> m_deviceTable;
	};

}