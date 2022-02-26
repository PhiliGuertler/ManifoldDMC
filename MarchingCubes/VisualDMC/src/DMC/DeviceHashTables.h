#pragma once

#include <LazyEngine/LazyEngine.h>
#include <thrust/device_vector.h>

#include <assert.h>

#include "Utils/Utilities.h"

namespace DMC {

	// ##################################################################### //
	// ### DeviceHashTable ################################################# //
	// ##################################################################### //

	namespace HashTable {
		static const int MAX_ITERATIONS = 128;
		static const size_t EMPTY_BUCKET = 0xFFFFFFFFFFFFFFFF;
		static const int INVALID_INDEX = -1;
	}

	struct Insertion {
		// If true, the bucket was previously empty
		bool bucketWasEmpty = false;
		// The hash-index of the inserted element.
		// Allows direct acces via operator[]
		size_t hashIndex = HashTable::EMPTY_BUCKET;
		// If true, the insertion was successful.
		bool insertionSuccessful = false;
	};

	/**
	 *	A Hashtable to be used in CUDA-Kernels.
	 *	It uses size_t's as keys.
	 *	Can only be created by a HostHashTable, which takes care of memory de/allocation.
	 */
	template <typename Value>
	class DeviceHashTable {
	public:
		__host__ DeviceHashTable(const LazyEngine::DataView<size_t>& keys, const LazyEngine::DataView<Value>& values)
			: m_keys(keys)
			, m_values(values)
		{
			// Keys and Values must have the same size
			assert(m_keys.size() == m_values.size());
		}


		/**
		 *	Returns the maximum capacity of this hash-table
		 */
		__device__ inline size_t capacity() const {
			return m_keys.size();
		}

		/**
		 *	A shorthand for both inserting a key and inserting a value.
		 *	NOTE: This is not useful for HashTables with multiple value-vectors!
		 *		These classes should instead use the key insertion and call operator[]
		 *		for each value themselves!
		 *	Inserts a value with a given key atomically into the hash-table
		 *	@returns true if the insertion was successful, false otherwise
		 */
		__device__ inline bool insert(const Value& value, const size_t key) {
			// insert the key first
			const Insertion insertion = insert(key);

			if (insertion.insertionSuccessful) {
				operator[](insertion.hashIndex) = value;
				return true;
			}

			// No free bucket matched the key.
			return false;
		}


		/**
		 *	Allows direct memory access to a bucket with a given index
		 */
		__device__ inline Value& operator[](const size_t index) {
			assert(index < m_values.size());
			return m_values[index];
		}

		/**
		 *	const-variant of operator[]
		 */
		__device__ inline Value get(const size_t index) const {
			assert(index < m_values.size());
			return m_values[index];
		}

		__device__ inline size_t getKeyAt(const size_t index) const {
			assert(index < m_keys.size());
			return m_keys[index];
		}

		/**
		 *	Returns true if the bucket at the given index is empty, false otherwise
		 */
		__device__ inline bool isBucketEmpty(const size_t index) const {
			assert(index < m_keys.size());
			return m_keys[index] == HashTable::EMPTY_BUCKET;
		}


		// ### Insertion for extending classes ############################# //

		/**
		 *	Performs a key insertion, which is comparable to reserving a space.
		 *	If the insertion was successful, the actual elements can be inserted directly
		 *	using operator[] on the hashIndex.
		 */
		__device__ inline Insertion insert(size_t key) {
			// compute the hash of this key
			size_t hash = hash1(key);
			// store an offset for quadratic steps
			size_t offset = 1;

			Insertion result;

			// Try 128 Buckets at most
			for (int i = 0; i < HashTable::MAX_ITERATIONS; ++i) {
				// Try to snatch this spot in the hashtable
				const size_t oldValue = atomicCAS(&m_keys[hash], HashTable::EMPTY_BUCKET, key);
				if (oldValue == HashTable::EMPTY_BUCKET) {
					// This bucket was empty, this is its first use
					result.insertionSuccessful = true;
					result.hashIndex = hash;
					result.bucketWasEmpty = true;
					return result;
				}
				else if (oldValue == key) {
					// This bucket already contains data, update it
					result.insertionSuccessful = true;
					result.hashIndex = hash;
					result.bucketWasEmpty = false;
					return result;
				}
				else {
					// This bucket is used differently, step on quadratically
					hash = (hash + offset * offset) % capacity();
					++offset;
				}
			}
			// No free bucket matched the key.
			return result;
		}

		/**
		 *	returns the first index for a key
		 */
		__device__ inline size_t indexFromKey(size_t key) const {
			return hash1(key);
		}

		/**
		 *	Returns the next index that would be returned if a collision had occured
		 *	@param index: The result of either indexFromKey or stepIndex
		 *	@param iteration: The number of the step iteration (starting at 1)
		 *	Example: When looking for a specific edge-entry that can is defined by its vertices,
		 *		the way to find the entry is:
		 *		First, call size_t index = indexFromKey(...); auto edges = get(index);
		 *		Check if at least one desired halfedge is part of the edge, or if the vertices are correct.
		 *		If not, call index = stepIndex(index, 1); and get the result, then index = stepIndex(index, 2), ...
		 */
		__device__ inline size_t stepIndex(size_t index, size_t iteration) const {
			return (index + iteration * iteration) % capacity();
		}

	protected:

		// ### Hash-Functions ############################################## //

		/**
		 *	simple hash function for 64bit key
		 */
		__device__ inline size_t hash1(const size_t k) const
		{
			return static_cast<size_t>(k % capacity());
		}
		/**
		 *	murmur like hash function
		 */
		__device__ inline size_t hash2(const int v0, const int v1) const
		{
			size_t h = concatenateKeys(v0, v1);
			h ^= h >> 33;
			h *= 0xff51afd7ed558ccdL;
			h ^= h >> 33;
			h *= 0xc4ceb9fe1a85ec53L;
			h ^= h >> 33;
			return static_cast<size_t>(h % capacity());
		}
		/**
		 *	64bit hash function
		 */
		__device__ inline size_t hash64shift(size_t key) const
		{
			key = (~key) + (key << 21); // key = (key << 21) - key - 1;
			key = key ^ (key >> 24);
			key = (key + (key << 3)) + (key << 8); // key * 265
			key = key ^ (key >> 14);
			key = (key + (key << 2)) + (key << 4); // key * 21
			key = key ^ (key >> 28);
			key = key + (key << 31);
			return key;
		}
		/**
		 *	Input 64bit key and return 32bit address
		 */
		__device__ inline size_t hash3(size_t key) const
		{
			key = (~key) + (key << 18); // key = (key << 18) - key - 1;
			key = key ^ (key >> 31);
			key = key * 21; // key = (key + (key << 2)) + (key << 4);
			key = key ^ (key >> 11);
			key = key + (key << 6);
			key = key ^ (key >> 22);
			return static_cast<size_t>(key % size());
		}

	protected:
		// A DataView pointing at the keys (see Host)
		LazyEngine::DataView<size_t> m_keys;
		// A DataView pointing at the values (see Host)
		LazyEngine::DataView<Value> m_values;
	};


	// ##################################################################### //
	// ### HostHashTable ################################################### //
	// ##################################################################### //

	/**
	 *	Simple base class for Host-Hashtables.
	 *	Extending classes must provide a cast function to their respective DeviceHashTable
	 *	and the device_vectors that are used for the values
	 */
	class HostHashTable {
	public:
		/**
		 *	constructor
		 *	@param maxCapacity: The maximum amount of elements
		 */
		HostHashTable(size_t maxCapacity);

		/**
		 *	default destructor
		 */
		virtual ~HostHashTable() = default;

		/**
		 *	initializes the keys
		 */
		virtual void initialize();

		/**
		 *	Clears the contents of the hashtable
		 */
		virtual inline void clear() {
			initialize();
		}

		/**
		 *	Returns the amount of elements that can be placed in this hashtable at most
		 */
		virtual inline size_t size() {
			return d_keys.size();
		}

	protected:
		MonitoredThrustBuffer<size_t> d_keys;
	};

	
	// ##################################################################### //
	// ### QuadrilateralHashTable ########################################## //
	// ##################################################################### //

	class QuadrilateralHostHashTable;

	/**
	 *	Hash-Table for Quads
	 *	Device-Side: This should be used in kernels
	 */
	class QuadrilateralDeviceHashTable {
	public:
		static constexpr uint32_t INVALID_INDEX = 0xFFFFFFFF;
		static constexpr uint32_t INVALID_COLOR = 0x1F;

	public:
		/**
		 *	returns the quad stored at a specific index
		 */
		__device__ inline glm::ivec4 getQuad(size_t index) const {
			return m_quads.get(index);
		}

		/**
		 *	returns the color stored at a specific index
		 */
		__device__ inline int getColor(size_t index) const {
			return m_quadColors.get(index);
		}

		/**
		 *	returns the capacity of this hashtable (in number of elements)
		 */
		__device__ inline size_t capacity() const {
			// any capacity function is fine, just use the one of the colors
			return m_quadColors.capacity();
		}

		/**
		 *	If true, there is no element stored at this index in the hashtable
		 */
		__device__ inline bool isBucketEmpty(const size_t index) const {
			// any bucketIsEmpty function is fine, just use the one of the colors
			return m_quadColors.isBucketEmpty(index);
		}

		/**
		 *	Adds a single vertex to the list of quadrilaterals
		 *	@param key: The key of the quad
		 *	@param position: The position of the vertex inside of the quad [0-3]
		 *	@param vertexID: The vertex's ID
		 *	@param color: [OPTIONAL]: The vertex's color (to be used for simplifications)
		 *	@returns true if the insertion was successful.
		 */
		__device__ inline bool addVertex(int key, int position, int vertexID, int color = -1) {
			// Insert the key in either of the two DeviceHashTables.
			Insertion insertion = m_quads.insert(key);

			if (insertion.insertionSuccessful) {
				// Insert the vertex ID into the quad
				m_quads[insertion.hashIndex][position] = vertexID;
				// Insert the color
				if (color != -1) {
					m_quadColors[insertion.hashIndex] = color;
				}
				return true;
			}

			// The insertion was not successful
			return false;
		}

	protected:
		friend class QuadrilateralHostHashTable;
		
		QuadrilateralDeviceHashTable(const thrust::device_vector<size_t>& keys, const thrust::device_vector<glm::ivec4>& quads, const thrust::device_vector<int>& quadColors);

	protected:
		DeviceHashTable<int> m_quadColors;
		DeviceHashTable<glm::ivec4> m_quads;
	};

	/**
	 *	A Hashtable for Quads
	 *	Host-Side: This is taking care of memory allocation/deallocation
	 */
	class QuadrilateralHostHashTable : public HostHashTable {
	public:
		QuadrilateralHostHashTable(size_t maxCapacity);

		virtual void initialize() override;

		operator QuadrilateralDeviceHashTable() {
			assert(m_deviceTable != nullptr);
			return *m_deviceTable;
		}

		inline LazyEngine::DataView<int> getDataView() {
			return d_quadColors;
		}

	protected:
		MonitoredThrustBuffer<glm::ivec4> d_quads;
		MonitoredThrustBuffer<int> d_quadColors;

		std::unique_ptr<QuadrilateralDeviceHashTable> m_deviceTable;
	};

}