#ifdef LAZYENGINE_CUDA

#include "CUDAGLInteroperabilty.h"

#include <cuda_gl_interop.h>
#include <cuda.h>
#include <stdio.h>

#include "CUDAUtils.h"

namespace LazyEngine {

	// ##################################################################### //
	// ### InteroperableCUDAElement ######################################## //
	// ##################################################################### //

	InteroperableCUDAElement::InteroperableCUDAElement()
		: m_mappedResource(nullptr)
		, m_mappedPointer(nullptr)
		, m_numBytes(0)
		, m_nestedScopeCount(0)
	{
		// empty
	}

	InteroperableCUDAElement::~InteroperableCUDAElement() {
		// empty
	}

	void InteroperableCUDAElement::mapToCUDA() {
		if (m_nestedScopeCount < 1) {
			// actually do the mapping, as this is the top level scope
			mapToCUDAImpl();
		}
		++m_nestedScopeCount;
	}

	void InteroperableCUDAElement::unmapFromCUDA() {
		--m_nestedScopeCount;
		if (m_nestedScopeCount < 1) {
			// actually do the unmapping, as this is the top level scope
			unmapFromCUDAImpl();

			m_mappedPointer = nullptr;
			m_numBytes = 0;
		}
	}

	bool InteroperableCUDAElement::isMappedToCUDA() const {
		return m_nestedScopeCount > 0;
	}



	// ##################################################################### //
	// ### InteroperableOpenGLVertexBuffer ################################# //
	// ##################################################################### //

	InteroperableOpenGLVertexBuffer::InteroperableOpenGLVertexBuffer(uint32_t size, BufferUsage usage)
		: OpenGLVertexBuffer(size, usage)
		, InteroperableCUDAElement()
	{
		registerGraphicsResource();
	}

	InteroperableOpenGLVertexBuffer::InteroperableOpenGLVertexBuffer(float *vertices, uint32_t size, BufferUsage usage)
		: OpenGLVertexBuffer(vertices, size, usage)
		, InteroperableCUDAElement()
	{
		registerGraphicsResource();
	}

	InteroperableOpenGLVertexBuffer::~InteroperableOpenGLVertexBuffer() {
		// FIXME: This causes a crash if the Graphics Context has already been destroyed (e.g. on Shutdown).
		cudaError_t error = cudaGraphicsUnregisterResource(m_mappedResource);
		if (error == cudaErrorInvalidGraphicsContext) {
			// ignore this error for now
		}
		else {
			checkCudaErrors(error);
		}
	}

	void InteroperableOpenGLVertexBuffer::registerGraphicsResource() {
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_mappedResource, m_rendererID, cudaGraphicsRegisterFlagsNone));
	}

	void InteroperableOpenGLVertexBuffer::mapToCUDAImpl() {
		// map the resource
		checkCudaErrors(cudaGraphicsMapResources(1, &m_mappedResource, 0));
		// update pointer and size
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_mappedPointer, &m_numBytes, m_mappedResource));
	}

	void InteroperableOpenGLVertexBuffer::unmapFromCUDAImpl() {
		// unmap the resource
		checkCudaErrors(cudaGraphicsUnmapResources(1, &m_mappedResource, 0));
	}


	// ##################################################################### //
	// ### InteroperableOpenGLIndexBuffer ################################## //
	// ##################################################################### //

	InteroperableOpenGLIndexBuffer::InteroperableOpenGLIndexBuffer(uint32_t* indices, uint32_t size, BufferUsage usage)
		: OpenGLIndexBuffer(indices, size, usage)
		, InteroperableCUDAElement()
	{
		registerGraphicsResource();
	}

	InteroperableOpenGLIndexBuffer::~InteroperableOpenGLIndexBuffer() {
		// FIXME: This causes a crash if the Graphics Context has already been destroyed (e.g. on Shutdown).
		cudaError_t error = cudaGraphicsUnregisterResource(m_mappedResource);
		if (error == cudaErrorInvalidGraphicsContext) {
			// ignore this error for now
		}
		else {
			checkCudaErrors(error);
		}
	}

	void InteroperableOpenGLIndexBuffer::mapToCUDAImpl() {
		// map the resource
		checkCudaErrors(cudaGraphicsMapResources(1, &m_mappedResource, 0));
		// update pointer and size
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_mappedPointer, &m_numBytes, m_mappedResource));
	}

	void InteroperableOpenGLIndexBuffer::unmapFromCUDAImpl() {
		// unmap the resource
		checkCudaErrors(cudaGraphicsUnmapResources(1, &m_mappedResource, 0));
	}

	void InteroperableOpenGLIndexBuffer::registerGraphicsResource() {
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_mappedResource, m_rendererID, cudaGraphicsRegisterFlagsNone));
	}

}

#endif