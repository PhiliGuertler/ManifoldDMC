#pragma once

#include <LazyEngine/LazyEngine.h>
#include <thrust/device_vector.h>

#include "Mesh.h"

typedef int IndexType;

namespace DMC {


	// ##################################################################### //
	// ### DeviceVector #################################################### //
	// ##################################################################### //

	/**
	 *	The GPU-Side of DeviceVector.
	 *  This should be used inside of kernels, while HostVector manages its GPU-Memory (de/allocation) on Host-Side.
	 *	Can only be instantiated by an extending class
	 */
	class DeviceVector {
	public:
		static const IndexType INVALID_INDEX = static_cast<IndexType>(-1);

		/**
		 *	destructor
		 */
		virtual ~DeviceVector() = default;

		/**
		 *	Returns the maximum amount of elements that can be stored in this vector.
		 */
		__device__ inline IndexType getMaxCapacity() const {
			return m_maxCapacity;
		}

		/**
		 *	WARNING: This might be totally racy if used in a context, where other threads
		 *	add new elements!
		 */
		__device__ inline IndexType size() const {
			return *m_size;
		}


	protected:
		/**
		 *	protected constructor
		 *  @param globalSize: A pointer to a GPU-Allocated size_t that will be used to keep track
		 *		of the amount of items stored in this vector.
		 */
		DeviceVector(IndexType* globalSizePointer, IndexType maxCapacity)
			: m_size(globalSizePointer)
			, m_maxCapacity(maxCapacity)
		{
			// empty
		}

		/**
		 *	Increases the size by 1 atomically and returns the previous size.
		 */
		__device__ inline IndexType incrementSize() {
			IndexType result = atomicAdd(m_size, 1);
			return result;
		}

		/**
		 *	Decreases the size by 1 atomically and returns the previous size.
		 */
		__device__ inline IndexType decrementSize() {
			IndexType result = atomicAdd(m_size, static_cast<IndexType>(-1));
			return result;
		}

		/**
		 *	[DEBUG]: sets the size
		 */
		__device__ inline void setSize(int size) {
			*m_size = size;
		}

		/**
		 *	Returns the maximum amount of elements that can be stored in this vector.
		 */
		__device__ inline IndexType getCapacity() const {
			return m_maxCapacity;
		}

		/**
		 *	Computes the next free index and updates the size, as if a new element has already been inserted.
		 *	If there is no next free index, the Vector's size remains unchanged.
		 *	@returns INVALID_INDEX if the capacity of this vector doesn't allow for another entry, or the index otherwise.
		 */
		__device__ inline IndexType computeIndex() {
			// update the vector's size while getting the current index atomically
			const IndexType index = incrementSize();
			// check if the index is valid
			if (index >= getCapacity()) {
				// The index is invalid, undo the size-increase
				decrementSize();
				// return an invalid index
				return INVALID_INDEX;
			}
			// Return the valid index
			return index;
		}

	protected:
		// A Pointer to the GPU-Memory-Allocated Item-Count tracker-variable
		// tldr; Saves how many items are currently stored in this vector.
		IndexType* m_size;
		// Stores how many elements can fit into this device vector.
		IndexType m_maxCapacity;
	};

	/**
	 *	CPU-Side of DeviceVector
	 *	Manages GPU-Memory de/allocation for DeviceVector's data and size.
	 *	Can only be instantiated by an extending class
	 */
	class HostVector {
	public:
		/**
		 *	destructor
		 *  Deallocates the size variable
		 */
		virtual ~HostVector();

		/**
		 *	Returns the size of added vertices
		 */
		IndexType size() const;

		/**
		 *	Sets the size of this class's corresponding DeviceVector to 0
		 */
		void resetSize();

		/**
		 *	Resizes GPU buffers to fit numElements.
		 */
		virtual void resize(IndexType numElements) {
			throw std::exception("Resize not implemented!");
		}

		virtual IndexType capacity() const = 0;

	protected:
		/**
		 *	constructor
		 *  Allocates and initializes the size variable
		 */
		HostVector();

		void copyTo(HostVector& other) const;

	protected:
		// a single size_t that stores how many elements have been added so far
		// While d_size is the variable in GPU Memory, m_size is handling its deallocation on destruction
		std::shared_ptr<IndexType> m_size;
		IndexType* d_size;
	};


	// ##################################################################### //
	// ### Atomic Counter ################################################## //
	// ##################################################################### //

	class AtomicCounterHost;

	/**
	 *	A simple atomic counter to be used in CUDA kernels
	 */
	class AtomicCounter {
	public:
		/**
		 *	Increases the counter by 1 atomically and returns the previous value.
		 */
		__device__ inline IndexType incrementCounter() {
			IndexType result = atomicAdd(m_counter, 1);
			return result;
		}

		/**
		 *	Decreases the counter by 1 atomically and returns the previous value.
		 */
		__device__ inline IndexType decrementSize() {
			IndexType result = atomicAdd(m_counter, static_cast<IndexType>(-1));
			return result;
		}

	protected:
		friend class AtomicCounterHost;

		__host__ AtomicCounter(IndexType* counterPointer)
			: m_counter(counterPointer)
		{
			// empty
		}

	protected:
		IndexType* m_counter;
	};

	class AtomicCounterHost {
	public:
		AtomicCounterHost(IndexType initialValue = 0);

		void reset(IndexType resetValue = 0);

		operator AtomicCounter();

	protected:
		// a single size_t that stores the current counter value
		// While d_counter is the variable in GPU Memory, m_counter is handling its deallocation on destruction
		std::shared_ptr<IndexType> m_counter;
		IndexType* d_counter;

		std::unique_ptr<AtomicCounter> m_deviceCounter;
	};

	// ##################################################################### //
	// ### VertexDeviceVector ############################################## //
	// ##################################################################### //

	class VertexHostVector;

	/**
	 *	The device-part of a VertexDeviceVector, that takes care of adding vertices
	 */
	class VertexDeviceVector : public DeviceVector {
	public:
		virtual ~VertexDeviceVector() = default;

		/**
		 *	Adds a vertex and returns its index
		 */
		__device__ inline IndexType addVertex(const glm::vec3& position, const glm::vec3& normal) {
			const IndexType index = computeIndex();
			if (index == DeviceVector::INVALID_INDEX) {
				//printf("VertexDeviceVector: Capacity exceeded!\n");
				return index;
			}
			// insert the vertex data
			m_vertexPositions[index] = position;
			m_vertexNormals[index] = normal;
			// return the valid index of this vertex (to be used for indexing)
			return index;
		}

		__device__ inline glm::vec3& getPosition(IndexType index) {
			DEVICE_ASSERT((index < m_vertexPositions.size()), "Index out of Bounds: %llu/%llu\n", index, m_vertexPositions.size());
			return m_vertexPositions[index];
		}

		__device__ inline glm::vec3& getNormal(IndexType index) {
			DEVICE_ASSERT((index < m_vertexNormals.size()), "Index out of Bounds: %llu/%llu\n", index, m_vertexNormals.size());
			return m_vertexNormals[index];
		}

	protected:
		friend class VertexHostVector;

		VertexDeviceVector(const thrust::device_vector<glm::vec3>& positions, const thrust::device_vector<glm::vec3>& normals, IndexType* size);

		// contains the vertex positions
		LazyEngine::DataView<glm::vec3> m_vertexPositions;
		// contains the vertex normals
		LazyEngine::DataView<glm::vec3> m_vertexNormals;
	};

	/**
	 *	The Host-Part of a VertexDeviceVector that takes care of memory allocation.
	 */
	class VertexHostVector : public HostVector {
	public:
		/**
		 *	constructor
		 *  @param size: The maximum Size of the Device-Buffer (which will be allocated)
		 */
		VertexHostVector(IndexType size);
		virtual ~VertexHostVector() = default;

		/**
		 *	copies the contents of this device vector into the given vertexbuffer
		 */
		void copyToVertexBuffer(LazyEngine::DataView<Vertex> vertexBuffer, const glm::vec4& color = glm::vec4(1.f, 0.f, 1.f, 1.f));

		/**
		 *	cast operator to VertexDeviceVector (for usage in CUDA-Kernels)
		 */
		inline operator VertexDeviceVector() {
			assert(m_deviceVector != nullptr);
			return *m_deviceVector;
		}

		/**
		 *	Initializes the device vector with invalid data
		 */
		void clear();

		/**
		 *	This is meant to be used for kernel-call arguments using ToCudaArgs()
		 */
		inline LazyEngine::DataView<glm::vec3> getDataView() const {
			return d_vertexPositions;
		}

		inline IndexType capacity() const {
			return d_vertexPositions.size();
		}

		/**
		 *	copies both contents and size into @param other
		 */
		void copyTo(VertexHostVector& other) const;

	protected:
		// a device vector containing the vertex positions
		MonitoredThrustBuffer<glm::vec3> d_vertexPositions;
		// a device vector containing the vertex normals
		MonitoredThrustBuffer<glm::vec3> d_vertexNormals;

		// Reference to the device vector
		std::unique_ptr<VertexDeviceVector> m_deviceVector;
	};


	// ##################################################################### //
	// ### QuadrilateralDeviceVector ####################################### //
	// ##################################################################### //

	struct QuadrilateralAttribute {
		static constexpr int INVALID_COLOR = 0x1F;

		// A tightly packed bitmask consisting of:
		//	- Bits 0-4: A color (one of 24 possible colors)
		//	- Bit 5: The "Non-Manifold"-Flag
		//	- Bit 6: The "Vertex Valance Simplification Fulfillment"-Flag
		//	- Bit 7: The "Can be removed"-Flag. Together with Bit 6, this forms the Pattern Flags

		//	- Bit 8: The "isBoundary"-Flag
		typedef uint16_t InternalType;
		InternalType data;

		HostDevice QuadrilateralAttribute()
			: data(static_cast<InternalType>(INVALID_COLOR))
		{
		}

		HostDevice inline void initialize(const int color = INVALID_COLOR) {
			setColor(color);
			unsetNonManifoldFlag();
			clearPatternFlags();
		}

		// --- Flag Setters --- //

		HostDevice inline void setColor(int color) {
			// override the last 5 bits of the attribute with its new color
			data = (data & static_cast<InternalType>(0xE0)) | static_cast<InternalType>(color);
		}

		HostDevice inline void unsetColor() {
			// reset the color to INVALID_COLOR
			data = data | static_cast<InternalType>(INVALID_COLOR);
		}

		HostDevice inline void setNonManifoldFlag() {
			// override the 6th bit of the attribute
			data = data | static_cast<InternalType>(0x20);
		}

		HostDevice inline void unsetNonManifoldFlag() {
			// override the 6th bit of the attribute
			data = data & (~static_cast<InternalType>(0x20));
		}

		/**
		 *	AKA Pattern bit
		 */
		HostDevice inline void setVertexValenceFlag() {
			// override the 7th bit of the attribute
			data = data | static_cast<InternalType>(0x40);
		}

		HostDevice inline void setPatternFlag() {
			setVertexValenceFlag();
		}

		HostDevice inline void unsetVertexValenceFlag() {
			// override the 7th bit of the attribute
			data = data & (~static_cast<InternalType>(0x40));
		}

		HostDevice inline void setRemoveFlag() {
			// override the 8th bit of the attribute
			data = data | static_cast<InternalType>(0x80);
		}

		HostDevice inline void unsetRemoveFlag() {
			// override the 8th bit of the attribute
			data = data & (~static_cast<InternalType>(0x80));
		}

		HostDevice inline void clearPatternFlags() {
			// override Bits 7 and 8 with 0.
			data = data & static_cast<InternalType>(0x3F);
		}

		HostDevice inline void setIsBoundaryFlag() {
			data = data | static_cast<InternalType>(0x100);
		}

		HostDevice inline void unsetIsBoundaryFlag() {
			data = data & (~static_cast<InternalType>(0x100));
		}

		
		// --- Flag Getters --- //

		HostDevice inline int getColor() const {
			return static_cast<int>(data & static_cast<InternalType>(0x1F));
		}

		HostDevice inline bool isNonManifold() const {
			return data & static_cast<InternalType>(0x20);
		}

		HostDevice inline bool isBoundary() const {
			return data & static_cast<InternalType>(0x100);
		}

		HostDevice inline bool isVertexValenceSet() const {
			return data & static_cast<InternalType>(0x40);
		}

		HostDevice inline bool isRemoveable() const {
			return data & static_cast<InternalType>(0x80);
		}

		HostDevice inline bool isP3X3Y() const {
			return isVertexValenceSet();
		}

		HostDevice inline bool isP3333() const {
			return isVertexValenceSet();
		}
	};

	class QuadrilateralHostVector;

	class QuadrilateralDeviceVector : public DeviceVector {
	public:
		virtual ~QuadrilateralDeviceVector() = default;

		/**
		 *	Adds a quadrilateral to the device vector
		 */
		__device__ inline size_t addQuadrilateral(const glm::ivec4& quad, int color = QuadrilateralAttribute::INVALID_COLOR) {
			const IndexType index = computeIndex();
			if (index == DeviceVector::INVALID_INDEX) {
				return index;
			}

			m_quads[index] = quad;
			m_attributes[index].setColor(color);

			return index;
		}

		/**
		 *	returns the attribute at the given index
		 */
		__device__ QuadrilateralAttribute& getAttribute(IndexType index) {
			return m_attributes[index];
		}

		__device__ glm::ivec4& getQuad(IndexType index) {
			return m_quads[index];
		}

		__device__ inline IndexType capacity() const {
			return m_attributes.size();
		}

	protected:
		friend class QuadrilateralHostVector;

		QuadrilateralDeviceVector(const thrust::device_vector<glm::ivec4>& quads, const thrust::device_vector<QuadrilateralAttribute>& attributes, IndexType* size);

		LazyEngine::DataView<glm::ivec4> m_quads;
		LazyEngine::DataView<QuadrilateralAttribute> m_attributes;
	};


	class QuadrilateralHostVector : public HostVector {
	public:
		/**
		 *	constructor
		 *	@param size: The maximum size
		 */
		QuadrilateralHostVector(IndexType size);
		/**
		 *	destructor
		 */
		virtual ~QuadrilateralHostVector();

		/**
		 *	Converts the quadrilaterals into two triangles and copies them into index buffer.
		 *  The Indexbuffer should be at least 1.5x as big as size suggests.
		 */
		void copyToIndexBuffer(LazyEngine::DataView<uint32_t> indexBuffer);

		/**
		 *	cast operator to QuadrilateralDeviceVector
		 */
		inline operator QuadrilateralDeviceVector() {
			return *m_deviceVector;
		}

		/**
		 *	Initializes all the data with invalid values
		 */
		void clear();

		inline LazyEngine::DataView<glm::ivec4> getDataView() const {
			return d_quadrilaterals;
		}

		void copyTo(QuadrilateralHostVector& other) const;

		virtual inline IndexType capacity() const override {
			return d_quadrilaterals.size();
		}

	protected:
		// A list of quads, defined by their four vertex-indices
		MonitoredThrustBuffer<glm::ivec4> d_quadrilaterals;
		// The attributes for each quad
		MonitoredThrustBuffer<QuadrilateralAttribute> d_attributes;

		std::unique_ptr<QuadrilateralDeviceVector> m_deviceVector;
	};


	// ##################################################################### //
	// ### TemplateVector ################################################## //
	// ##################################################################### //

	/**
	 *	
	 */
	template <typename T>
	class TemplateHostVector;

	/**
	 *
	 */
	template <typename T>
	class TemplateDeviceVector : public DeviceVector {
	public:
		__device__ inline IndexType push(const T& element) {
			const IndexType index = computeIndex();
			if (index == DeviceVector::INVALID_INDEX) {
				//printf("TemplateDeviceVector: capacity exceeded!\n");
				return index;
			}
			m_elements[index] = element;
			return index;
		}

		__device__ inline T& operator[](IndexType index) {
			return m_elements[index];
		}

	protected:
		friend class TemplateHostVector<T>;

		__host__ inline TemplateDeviceVector(const thrust::device_vector<T>& quads, IndexType* size)
			: DeviceVector(size, quads.size())
			, m_elements(quads)
		{
			// empty
		}

	protected:
		LazyEngine::DataView<T> m_elements;
	};

	/**
	 *	
	 */
	template <typename T>
	class TemplateHostVector : public HostVector {
	public:
		__host__ inline TemplateHostVector(IndexType size)
			: d_elements(size)
			, m_deviceVector(std::unique_ptr<TemplateDeviceVector<T>>(new TemplateDeviceVector<T>(d_elements, d_size)))
		{
			// empty
		}

		__host__ inline void clear(const T& defaultValue) {
			thrust::fill(d_elements.begin(), d_elements.end(), defaultValue);
		}

		__host__ inline operator TemplateDeviceVector<T>() {
			assert(m_deviceVector != nullptr);
			return *m_deviceVector;
		}

		__host__ inline LazyEngine::DataView<T> getDataView() const {
			return d_elements;
		}

		__host__ virtual inline IndexType capacity() const override {
			return d_elements.size();
		}

	protected:
		thrust::device_vector<T> d_elements;
		std::unique_ptr<TemplateDeviceVector<T>> m_deviceVector;
	};

}