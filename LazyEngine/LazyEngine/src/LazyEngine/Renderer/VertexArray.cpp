// ######################################################################### //
// ### VertexArray.cpp ##################################################### //
// ### Delegates the creation of a VertexArray to a specific API         ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"

#include "VertexArray.h"

#include "Renderer.h"

// --- OpenGL includes --- //
#include "LazyEngine/platform/OpenGL/OpenGLVertexArray.h"
// --- OpenGL includes --- //

#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	Ref<VertexArray> VertexArray::create() {
		LAZYENGINE_PROFILE_FUNCTION();

		switch (Renderer::getAPI()) {
		case RendererAPI::API::None:
			// --- None --- //
			LAZYENGINE_CORE_ASSERT(false, "RendererAPI::None is currently not supported!");
			return nullptr;
		case RendererAPI::API::OpenGL:
			// --- OpenGL --- //
			return createRef<OpenGLVertexArray>();
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