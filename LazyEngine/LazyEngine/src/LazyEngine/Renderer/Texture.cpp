// ######################################################################### //
// ### Texture.cpp ######################################################### //
// ### implements Texture.h                                              ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"
#include "Texture.h"

#include "Renderer.h"

// --- OpenGL Implementation --- //
#include "LazyEngine/platform/OpenGL/OpenGLTexture.h"
// --- OpenGL Implementation --- //

#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	// ######################################################################## //
	// ### Texture2D ########################################################## //
	// ######################################################################## //

	Ref<Texture2D> Texture2D::create(const std::string& path) {
		LAZYENGINE_PROFILE_FUNCTION();

		switch (Renderer::getAPI()) {
		case RendererAPI::API::None:
			LAZYENGINE_CORE_ASSERT(false, "This API is currently not supported")
			return nullptr;
		case RendererAPI::API::OpenGL: 
			// --- OpenGL --- //
			return createRef<OpenGLTexture2D>(path);

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


	Ref<Texture2D> Texture2D::create(uint32_t width, uint32_t height, void *data, uint32_t size, TextureFormat format) {
		LAZYENGINE_PROFILE_FUNCTION();

		switch (Renderer::getAPI()) {
		case RendererAPI::API::None:
			LAZYENGINE_CORE_ASSERT(false, "This API is currently not supported")
				return nullptr;
		case RendererAPI::API::OpenGL:
			// --- OpenGL --- //
			return createRef<OpenGLTexture2D>(width, height, data, size, format);

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

	Ref<Texture2D> Texture2D::createDepthStencil(uint32_t width, uint32_t height) {
		LAZYENGINE_PROFILE_FUNCTION();

		switch (Renderer::getAPI()) {
		case RendererAPI::API::None:
			LAZYENGINE_CORE_ASSERT(false, "This API is currently not supported")
				return nullptr;
		case RendererAPI::API::OpenGL:
			// --- OpenGL --- //
			return createRef<OpenGLTexture2D>(width, height, nullptr, 0, TextureFormat::DepthStencil);

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


	// ######################################################################## //
	// ### SubTexture ######################################################### //
	// ######################################################################## //

	SubTexture2D::SubTexture2D(Ref<Texture2D> texture, const std::array<glm::vec2, 4>& textureCoordinates)
		: m_texture(texture)
		, m_textureCoordinates(textureCoordinates)
	{
		// empty
	}
}