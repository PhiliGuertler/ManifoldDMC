#pragma once
#ifdef LAZYENGINE_CUDA

#include "LazyEngine/platform/OpenGL/OpenGLBuffer.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace LazyEngine {

	// forward declarations
	template <typename T>
	class ScopedCUDAInterop;
	
	class InteroperableCUDAElement;

	/**
	 *	A simple DataView for a mapped CUDA resource
	 */
	template <typename T>
	class DataView {
	protected:
		friend class InteroperableCUDAElement;
		template <typename T>
		friend class ScopedCUDAInterop;

		T* m_dataPointer = nullptr;
		size_t m_size = 0;
		size_t m_elementCount = 0;

		/**
		 *	constructor that will be used by CUDA-GL-Interoperations
		 */
		DataView(T* dataPointer, size_t size)
			: m_dataPointer(dataPointer)
			, m_size(size)
			, m_elementCount(size / sizeof(T))
		{
			// empty
		}

	public:
		/**
		 *	constructor for thrust device-vectors
		 */
		DataView(const thrust::device_vector<T>& vector)
			: m_dataPointer(const_cast<T*>(thrust::raw_pointer_cast(vector.data())))
			, m_size(vector.size() * sizeof(T))
			, m_elementCount(vector.size())
		{
			// empty
		}

		__host__ __device__
			inline T& operator[](size_t index) {
			return m_dataPointer[index];
		}

		__host__ __device__
			inline const T& operator[](size_t index) const {
			return m_dataPointer[index];
		}

		/**
		 *	Returns the number of <T> elements in the underlying array
		 */
		__host__ __device__
			inline size_t size() const {
			return m_elementCount;
		}

		/**
		 *	Returns the size of the underlying array in bytes
		 */
		__host__ __device__
			inline size_t getSizeInBytes() const {
			return m_size;
		}

		__host__ __device__
			inline size_t getNumThreadsPerBlock() const {
			// HACK: use 128 Threads per block by default
			return 128;
		}

		__host__ __device__
			inline size_t getNumBlocks() const {
			return static_cast<size_t>(std::ceil(static_cast<double>(size()) / static_cast<double>(getNumThreadsPerBlock())));
		}

		__host__ __device__
			inline T* getRawPointer() const {
			return m_dataPointer;
		}

		__host__
			inline thrust::device_ptr<T> begin() const {
			return thrust::device_ptr<T>(m_dataPointer);
		}

		__host__
			inline thrust::device_ptr<T> end() const {
			return thrust::device_ptr<T>(m_dataPointer + size());
		}
	};

	/**
	 *	An interface for Interopereable CUDA Elements like OpenGL VertexBuffers.
	 */
	class InteroperableCUDAElement {
	public:
		virtual ~InteroperableCUDAElement();

	protected:
		template <typename T>
		friend class ScopedCUDAInterop;

		/**
		 *	Protected constructor
		 */
		InteroperableCUDAElement();

		/**
		 *	maps the element in CUDA's memory space and populates m_mappedPointer and m_numBytes.
		 */
		void mapToCUDA();
		/**
		 *	unmaps the element from CUDA's memory space, so that it can be used with OpenGL again.
		 */
		void unmapFromCUDA();

		bool isMappedToCUDA() const;

		// --- To be extended --- //
		virtual void mapToCUDAImpl() = 0;
		virtual void unmapFromCUDAImpl() = 0;

		/**
		 *	Registers this graphics resource in CUDA and populates m_mappedResource
		 */
		virtual void registerGraphicsResource() = 0;

		/**
		 *	Creates a DataView of the mapped data.
		 */
		template <typename T>
		DataView<T> mapView(T* pointer, size_t numBytes) const {
			return DataView<T>(pointer, numBytes);
		}

	protected:
		// The result of a registered GL Buffer object in CUDA
		::cudaGraphicsResource* m_mappedResource;
		// a pointer pointing to the mapped GL Buffer in CUDA's memory space
		void* m_mappedPointer;
		// the size of the mapped buffer in bytes
		size_t m_numBytes;
		// Stores how many CUDAInterop Scopes currently exist around this element.
		// If this is > 0, the element cannot be used in OpenGL and must be unmapped first!
		uint32_t m_nestedScopeCount;
	};

	/**
	 *	Similar to std::scoped_lock, this class maps an InteroperableCUDAElement on construction
	 *  And unmaps it on destruction.
	 *  The template type T defines what kind of DataView will be created.
	 */
	template <typename T>
	class ScopedCUDAInterop {
	public:
		ScopedCUDAInterop(InteroperableCUDAElement& cudaElement)
			: m_element(cudaElement)
		{
			m_element.mapToCUDA();
		}

		virtual ~ScopedCUDAInterop() {
			m_element.unmapFromCUDA();
		}

		DataView<T> getMapping() const {
			if (!m_element.isMappedToCUDA()) {
				throw std::exception("InteroperableCUDAElement is not mapped!");
			}
			return m_element.mapView<T>((T*)(m_element.m_mappedPointer), m_element.m_numBytes);
		}

		operator DataView<T>() const {
			return getMapping();
		}

	protected:
		InteroperableCUDAElement& m_element;
	};

	/**
	 *	An OpenGL VertexBuffer that can be mapped into CUDA's memory space
	 */
	class InteroperableOpenGLVertexBuffer : public OpenGLVertexBuffer, public InteroperableCUDAElement {
	public:
		InteroperableOpenGLVertexBuffer(uint32_t size, BufferUsage usage);
		InteroperableOpenGLVertexBuffer(float *vertices, uint32_t size, BufferUsage usage);
		virtual ~InteroperableOpenGLVertexBuffer();

	protected:
		/**
		 *	maps the element in CUDA's memory space and populates m_mappedPointer and m_numBytes.
		 */
		virtual void mapToCUDAImpl() override;
		/**
		 *	unmaps the element from CUDA's memory space, so that it can be used with OpenGL again.
		 */
		virtual void unmapFromCUDAImpl() override;
		/**
		 *	Registers this graphics resource in CUDA and populates m_mappedResource
		 */
		virtual void registerGraphicsResource() override;
	};

	/**
	 *	An OpenGL IndexBuffer that can be mapped into CUDA's memory space
	 */
	class InteroperableOpenGLIndexBuffer : public OpenGLIndexBuffer, public InteroperableCUDAElement {
	public:
		InteroperableOpenGLIndexBuffer(uint32_t* indices, uint32_t size, BufferUsage usage);
		virtual ~InteroperableOpenGLIndexBuffer();

	protected:
		/**
		 *	maps the element in CUDA's memory space and populates m_mappedPointer and m_numBytes.
		 */
		virtual void mapToCUDAImpl() override;
		/**
		 *	unmaps the element from CUDA's memory space, so that it can be used with OpenGL again.
		 */
		virtual void unmapFromCUDAImpl() override;
		/**
		 *	Registers this graphics resource in CUDA and populates m_mappedResource
		 */
		virtual void registerGraphicsResource() override;
	};

}

#endif