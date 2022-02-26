#ifdef LAZYENGINE_CUDA
#include "CUDAUtils.h"

namespace LazyEngine {

	CUDAStream::CUDAStream()
		: m_streamHandle()
	{
		cudaStreamCreate(&m_streamHandle);
	}

	CUDAStream::~CUDAStream() {
		// FIXME: This will throw an exception if the CUDA Context has already been destroyed!
		cudaStreamDestroy(m_streamHandle);
	}

}

#endif