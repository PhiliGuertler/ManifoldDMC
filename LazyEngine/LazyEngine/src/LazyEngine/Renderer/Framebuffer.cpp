// ######################################################################### //
// ### Framebuffer.cpp ##################################################### //
// ### Delegates the creation of a Framebuffer to a specific API         ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"

#include "Framebuffer.h"

#include "Renderer.h"
#include "LazyEngine/Profiling/Profiler.h"

// --- OpenGL includes --- //
#include "LazyEngine/platform/OpenGL/OpenGLFramebuffer.h"
// --- OpenGL includes --- //

namespace LazyEngine {

	Ref<Framebuffer> Framebuffer::create(uint32_t width, uint32_t height) {
		LAZYENGINE_PROFILE_FUNCTION();

		switch (Renderer::getAPI()) {
		case RendererAPI::API::None:
			// --- None --- //
			LAZYENGINE_CORE_ASSERT(false, "RendererAPI::None is currently not supported!");
			return nullptr;
		case RendererAPI::API::OpenGL:
			// --- OpenGL --- //
			return createRef<OpenGLFramebuffer>(width, height);
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