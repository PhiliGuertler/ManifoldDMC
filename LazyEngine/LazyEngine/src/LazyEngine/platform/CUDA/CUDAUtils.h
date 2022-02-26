#pragma once
#ifdef LAZYENGINE_CUDA

#include "LazyEngine/gepch.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>


namespace LazyEngine {

	/**
	 *	defines the default CUDA block size
	 */
	constexpr unsigned int CUDA_Block_Size = 128;

	/**
	 *	Implementation of the cudaCheckErrors(err) definition that replaces @param file and @param line
	 *	with the file and line of invocation
	inline void __checkCudaErrors(cudaError_t err, const char* file, int line) {
		if (CUDA_SUCCESS != err) {
			const char* errorStr = cudaGetErrorString(err);
			LAZYENGINE_ERROR("checkCudaErrors() Driver API error = {0} \"{1}\"\n\tfrom file <{2}>,\n\tline {3}", err, errorStr, file, line);

			__debugbreak();
		}
	}
	 */

#define __checkCudaErrors(err, file, line)\
	if (CUDA_SUCCESS != err) {\
		const char *errorStr = cudaGetErrorString(err);\
		LAZYENGINE_ERROR("checkCudaErrors() Driver API error = {0} \"{1}\"\n\tfrom file <{2}>,\n\tline {3}", err, errorStr, file, line);\
		__debugbreak();\
	}

	/**
	 *	Checks for errors in CUDA-calls and triggers a debugbreak if any error occurs.
	 *	Can be used with any CUDA-function that returns cudaError_t.
	 */
#ifdef LAZYENGINE_ENABLE_ASSERTS
#define checkCudaErrors(err) cudaDeviceSynchronize(); /*::LazyEngine::*/__checkCudaErrors(err, __FILE__, __LINE__)
#else
#define checkCudaErrors(err) cudaDeviceSynchronize(); err
#endif

	/**
	 *	Implementation of the cudaCheckError() definition that replaces @param file and @param line
	 *  with the file and line of invocation
	inline void __cudaError(const char* file, int line)
	{
		cudaError_t e = cudaGetLastError();
		if (e != cudaSuccess)
		{
			const char* errorStr = cudaGetErrorString(e);
			LAZYENGINE_ERROR("checkCudaError(): \"{0}\" from file <{1}>, line {2}", errorStr, file, line);
			
			__debugbreak();
		}
	}
	 */

#define __cudaError(file, line) {\
		cudaError_t e = cudaGetLastError();\
		if (e != cudaSuccess) {\
			const char* errorStr = cudaGetErrorString(e);\
			LAZYENGINE_ERROR("checkCudaError(): \"{0}\" from file <{1}>, line {2}", errorStr, file, line);\
			__debugbreak();\
		}\
	}

	/**
	 *	Checks for recent errors in CUDA-calls
	 */
#ifdef LAZYENGINE_ENABLE_ASSERTS
#define cudaCheckError() cudaDeviceSynchronize(); /*::LazyEngine::*/__cudaError(__FILE__, __LINE__)
#else
#define cudaCheckError() cudaDeviceSynchronize(); __cudaError(__FILE__, __LINE__)
#endif

	struct CUDAMemoryInfo {
		size_t free;
		size_t total;
	};

	inline CUDAMemoryInfo cudaMemoryInfo() {
		CUDAMemoryInfo info;
		checkCudaErrors(cudaMemGetInfo(&info.free, &info.total));
		return info;
	}

	class ScopedCUDAMemoryLeakDetector {
	public:
		ScopedCUDAMemoryLeakDetector(const char *file, int line)
			: m_before(cudaMemoryInfo())
			, m_file(file)
			, m_line(line)
		{
			// empty
		}
		~ScopedCUDAMemoryLeakDetector() {
			CUDAMemoryInfo after = cudaMemoryInfo();
			if (after.free < m_before.free) {
				LAZYENGINE_WARN("Possible Memory Leak detected at file <{0}>, line {1}", m_file, m_line);
				LAZYENGINE_INFO("Before {0}B, After {1}B", m_before.free, after.free);
				//__debugbreak();
			}
		}

	protected:
		CUDAMemoryInfo m_before;
		const char *m_file;
		int m_line;
	};

#ifdef LAZYENGINE_ENABLE_ASSERTS
#define cudaTestMemoryLeakScope() cudaDeviceSynchronize(); ::LazyEngine::ScopedCUDAMemoryLeakDetector scopedMLD(__FILE__, __LINE__)
#else
#define cudaTestMemoryLeakScope()
#endif

	/**
	 *	Prints the amount of Free
	 */
	inline void printCudaMemory() {
		size_t freeMemory;
		size_t totalMemory;
		checkCudaErrors(cudaMemGetInfo(&freeMemory, &totalMemory));
		// compute a percentage of free to total memory
		float freePercentage = static_cast<float>(freeMemory) / static_cast<float>(totalMemory) * 100.f;
		LAZYENGINE_CORE_WARN("{0}% of Memory avaliable. Free Memory: {1} Bytes, Total Memory: {2} Bytes", freePercentage, freeMemory, totalMemory);
	}

	/**
	 *	To be used in kernels only.
	 *  This class queries various useful indices for the current thread in a CUDA kernel.
	 */
	class CUDAInfo {
	public:

		__device__ CUDAInfo()
			: m_blockThreadId(threadIdx.x)
			, m_blockId(blockIdx.x)
			, m_globalThreadId()
			, m_numThreadsPerBlock(blockDim.x)
			, m_numBlocks(gridDim.x)
		{
			m_globalThreadId = m_numThreadsPerBlock * m_blockId + m_blockThreadId;
		}

		__device__ inline size_t getGlobalThreadId() const {
			return m_globalThreadId;
		}
		
		__device__ inline size_t getLocalThreadId() const {
			return m_blockThreadId;
		}

		__device__ inline size_t getBlockId() const {
			return m_blockId;
		}

		__device__ inline size_t getNumThreadsPerBlock() const {
			return m_numThreadsPerBlock;
		}

		__device__ inline size_t getNumBlocks() const {
			return m_numBlocks;
		}

	protected:
		// The thread index in the current block (%tid.x)
		size_t m_blockThreadId;
		// The block index (%ctaid.x)
		size_t m_blockId;
		// This thread's global index
		size_t m_globalThreadId;

		// The number of threads per block
		size_t m_numThreadsPerBlock;
		// The number of active blocks
		size_t m_numBlocks;
	};

	/**
	 *	uploads given Data to a CUDA __constant__ variable.
	 *	checkCudaErrors(cudaMemcpyToSymbol(variableName, (void*)data.data(), data.size() * sizeof(T)));
	 */

	/**
	 *	A function that tricks the compiler into fewer load/store instructions, which can cause a significant speedup.
	 *  Example: copying a struct with 8 floats (32 bytes) will result in two 16-byte load/stores using this function,
	 *	instead of potentially eight 4-byte load/stores.
	template <typename T, typename DATA_TYPE = int4>
	__device__ inline void quickCopy(const T* source, T* destination) {
		static_assert(sizeof(T) % sizeof(DATA_TYPE) == 0, "quickCopy: Incompatible Types. Is your Datatype actually a whole multiple of 4 ints wide?");
		const DATA_TYPE* source4 = reinterpret_cast<const DATA_TYPE*>(source);
		DATA_TYPE* destination4 = reinterpret_cast<DATA_TYPE*>(destination);
#pragma unroll
		for (int i = 0; i < sizeof(T) / sizeof(DATA_TYPE); ++i) {
			destination4[i] = source4[i];
		}
	}
	 */

	/**
	 *	A simple wrapper for CUDA-Streams, that will be created on construction and destroyed on destruction.
	 *	CUDAStreams are necessary to start kernels in parallel.
	 */
	class CUDAStream {
	public:
		CUDAStream();
		virtual ~CUDAStream();

		operator cudaStream_t() {
			return m_streamHandle;
		}

		operator cudaStream_t() const {
			return m_streamHandle;
		}
	protected:
		cudaStream_t m_streamHandle;
	};

}

// These are actually defined by CUDA, but they are not part of any header.
// Usually, they are only accessible in .cu files, but forward declaring them here
// enables their usage in .h files as well (if they are included in .cu files)
__device__
int atomicCAS(int* address, int compare, int val);
__device__
uint32_t atomicCAS(uint32_t* address, uint32_t compare, uint32_t val);
__device__
size_t atomicCAS(size_t* address, size_t compare, size_t val);
__device__
int atomicAdd(int* address, int val);
__device__
unsigned long long int atomicAdd(unsigned long long int* address, unsigned long long int val);
__device__
int atomicAnd(int* address, int val);
__device__
unsigned int atomicAnd(unsigned int* address, unsigned int val);
__device__
unsigned long long int atomicAnd(unsigned long long int* address, unsigned long long int val);
__device__
int atomicOr(int* address, int val);
__device__
unsigned int atomicOr(unsigned int* address, unsigned int val);
__device__
unsigned long long int atomicOr(unsigned long long int* address, unsigned long long int val);
__device__
int atomicXor(int* address, int val);
__device__
unsigned int atomicXor(unsigned int* address, unsigned int val);
__device__
unsigned long long int atomicXor(unsigned long long int* address, unsigned long long int val);
__device__
int max(int a, int b);
__device__
unsigned int max(unsigned int a, unsigned int b);
__device__
unsigned long long max(unsigned long long a, unsigned long long b);
__device__
int min(int a, int b);
__device__
unsigned int min(unsigned int a, unsigned int b);
__device__
unsigned long long min(unsigned long long a, unsigned long long b);
#endif

#ifdef LAZYENGINE_CUDA
	#define HostDevice __host__ __device__
	#ifdef __INTELLISENSE__
		// By undefining CUDA_KERNEL for intellisense, it will only be expanded for regular compilation!
		#define CUDA_KERNEL(...)

	#else
		#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
	#endif
#else
	#define HostDevice
	#define CUDA_KERNEL(...)
#endif

