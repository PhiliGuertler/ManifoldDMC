// ######################################################################### //
// ### Shader.cpp ########################################################## //
// ### Implements Shader.h                                               ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"

#include "Shader.h"

#include "Renderer.h"

// --- OpenGL implementation --- //
#include "LazyEngine/platform/OpenGL/OpenGLShader.h"
// --- OpenGL implementation --- //

#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	// ##################################################################### //
	// ### Shader ########################################################## //
	// ##################################################################### //

	/**
	 *	creates a shader from a vertex source and fragment source and dispatches
	 *	the command to the active renderer api
	 */
	Ref<Shader> Shader::create(const std::string& name, const std::string& vertexSrc, const std::string& fragmentSrc) {
		LAZYENGINE_PROFILE_FUNCTION();

		switch (Renderer::getAPI()) {
		case RendererAPI::API::None:
			// --- None --- //
			LAZYENGINE_CORE_ASSERT(false, "RendererAPI::None is currently not supported!");
			return nullptr;
		case RendererAPI::API::OpenGL:
			// --- OpenGL --- //
			return createRef<OpenGLShader>(name, vertexSrc, fragmentSrc);
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

	/**
	 *	creates a shader from a vertex source and fragment source and dispatches
	 *	the command to the active renderer api
	 */
	Ref<Shader> Shader::create(const std::string& name, const std::unordered_map<ShaderRoutine, std::string>& sources) {
		LAZYENGINE_PROFILE_FUNCTION();

		switch (Renderer::getAPI()) {
		case RendererAPI::API::None:
			// --- None --- //
			LAZYENGINE_CORE_ASSERT(false, "RendererAPI::None is currently not supported!");
			return nullptr;
		case RendererAPI::API::OpenGL:
			// --- OpenGL --- //
			return createRef<OpenGLShader>(name, sources);
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

	/**
	 *	creates a shader from a file at filepath and dispatches the command
	 *	to the active renderer api
	 */
	Ref<Shader> Shader::create(const std::string& filepath) {
		LAZYENGINE_PROFILE_FUNCTION();

		switch (Renderer::getAPI()) {
		case RendererAPI::API::None:
			// --- None --- //
			LAZYENGINE_CORE_ASSERT(false, "RendererAPI::None is currently not supported!");
			return nullptr;
		case RendererAPI::API::OpenGL:
			// --- OpenGL --- //
			return createRef<OpenGLShader>(filepath);
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


	/**
	 *	creates a shader from a file at filepath and dispatches the command
	 *	to the active renderer api
	 */
	Ref<Shader> Shader::create(const std::string& name, const std::string& filepath) {
		LAZYENGINE_PROFILE_FUNCTION();

		switch (Renderer::getAPI()) {
		case RendererAPI::API::None:
			// --- None --- //
			LAZYENGINE_CORE_ASSERT(false, "RendererAPI::None is currently not supported!");
			return nullptr;
		case RendererAPI::API::OpenGL:
			// --- OpenGL --- //
			return createRef<OpenGLShader>(name, filepath);
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
	// ### ShaderLibrary ################################################### //
	// ##################################################################### //

	void ShaderLibrary::add(const Ref<Shader>& shader) {
		LAZYENGINE_PROFILE_FUNCTION();

		// get shader name
		const std::string& shaderName = shader->getName();
		// check if shader is already in the map
		LAZYENGINE_CORE_ASSERT(!exists(shaderName), "Shader already exists in ShaderLibrary: \"{0}\". Every shader in this library must have a unique name!", shaderName);
		// add it to the map
		m_shaders[shaderName] = shader;
	}

	LazyEngine::Ref<LazyEngine::Shader> ShaderLibrary::load(const std::string& filepath) {
		LAZYENGINE_PROFILE_FUNCTION();

		auto shader = Shader::create(filepath);
		add(shader);
		return shader;
	}

	LazyEngine::Ref<LazyEngine::Shader> ShaderLibrary::load(const std::string& name, const std::string& filepath) {
		LAZYENGINE_PROFILE_FUNCTION();

		auto shader = Shader::create(name, filepath);
		add(shader);
		return shader;
	}

	LazyEngine::Ref<LazyEngine::Shader> ShaderLibrary::get(const std::string& name) {
		LAZYENGINE_PROFILE_FUNCTION();

		LAZYENGINE_CORE_ASSERT(exists(name), "Shader not found: \"{0}\"", name);
		return m_shaders[name];
	}

	bool ShaderLibrary::exists(const std::string& name) const {
		LAZYENGINE_PROFILE_FUNCTION();

		return m_shaders.find(name) != m_shaders.end();
	}
}