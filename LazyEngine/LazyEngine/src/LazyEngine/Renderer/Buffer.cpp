// ######################################################################### //
// ### Buffer.cpp ########################################################## //
// ### Implements Buffer.h                                               ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"

#include "Buffer.h"

#include "Renderer.h"

// --- OpenGL implementations --- //
#ifdef LAZYENGINE_CUDA
#include "LazyEngine/platform/CUDA/CUDAGLInteroperabilty.h"
#endif
#include "LazyEngine/platform/OpenGL/OpenGLBuffer.h"
// --- OpenGL implementations --- //

#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	// ##################################################################### //
	// ### VertexBuffer #################################################### //
	// ##################################################################### //

	Ref<VertexBuffer> VertexBuffer::create(uint32_t size, BufferUsage usage) {
		LAZYENGINE_PROFILE_FUNCTION();

		switch (Renderer::getAPI()) {
		case RendererAPI::API::None:
			// --- None --- //
			LAZYENGINE_CORE_ASSERT(false, "RendererAPI::None is currently not supported!");
			return nullptr;
		case RendererAPI::API::OpenGL:
			// --- OpenGL --- //
#ifdef LAZYENGINE_CUDA
			return createRef<InteroperableOpenGLVertexBuffer>(size, usage);
#else
			return createRef<OpenGLVertexBuffer>(size, usage);
#endif
		case RendererAPI::API::DirectX:
			// --- DirectX --- //
			LAZYENGINE_CORE_ASSERT(false, "RendererAPI::DirectX is currently not supported!");
			return nullptr;
		case RendererAPI::API::Vulcan:
			// --- Vulcan --- //
			LAZYENGINE_CORE_ASSERT(false, "RendererAPI::Vulcan is currently not supported!");
			return nullptr;
		}

		LAZYENGINE_CORE_ASSERT(false, "Unknown RendererAPI!");
		return nullptr;
	}

	Ref<VertexBuffer> VertexBuffer::create(float *vertices, uint32_t size, BufferUsage usage) {
		LAZYENGINE_PROFILE_FUNCTION();

		switch (Renderer::getAPI()) {
		case RendererAPI::API::None: 
			// --- None --- //
			LAZYENGINE_CORE_ASSERT(false, "RendererAPI::None is currently not supported!");
			return nullptr;
		case RendererAPI::API::OpenGL:
			// --- OpenGL --- //
#ifdef LAZYENGINE_CUDA
			return createRef<InteroperableOpenGLVertexBuffer>(vertices, size, usage);
#else
			return createRef<OpenGLVertexBuffer>(vertices, size, usage);
#endif
		case RendererAPI::API::DirectX:
			// --- DirectX --- //
			LAZYENGINE_CORE_ASSERT(false, "RendererAPI::DirectX is currently not supported!");
			return nullptr;
		case RendererAPI::API::Vulcan:
			// --- Vulcan --- //
			LAZYENGINE_CORE_ASSERT(false, "RendererAPI::Vulcan is currently not supported!");
			return nullptr;
		}

		LAZYENGINE_CORE_ASSERT(false, "Unknown RendererAPI!");
		return nullptr;
	}


	// ##################################################################### //
	// ### IndexBuffer ##################################################### //
	// ##################################################################### //

	Ref<IndexBuffer> IndexBuffer::create(uint32_t *indices, uint32_t count, BufferUsage usage) {
		LAZYENGINE_PROFILE_FUNCTION();

		switch (Renderer::getAPI()) {
		case RendererAPI::API::None:
			// --- None --- //
			LAZYENGINE_CORE_ASSERT(false, "RendererAPI::None is currently not supported!");
			return nullptr;
		case RendererAPI::API::OpenGL:
			// --- OpenGL --- //
#ifdef LAZYENGINE_CUDA
			return createRef<InteroperableOpenGLIndexBuffer>(indices, count, usage);
#else
			return createRef<OpenGLIndexBuffer>(indices, count, usage);
#endif
		case RendererAPI::API::DirectX:
			// --- DirectX --- //
			LAZYENGINE_CORE_ASSERT(false, "RendererAPI::DirectX is currently not supported!");
			return nullptr;
		case RendererAPI::API::Vulcan:
			// --- Vulcan --- //
			LAZYENGINE_CORE_ASSERT(false, "RendererAPI::Vulcan is currently not supported!");
			return nullptr;
		}

		LAZYENGINE_CORE_ASSERT(false, "Unknown RendererAPI!");
		return nullptr;
	}
}