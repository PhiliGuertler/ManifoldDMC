#pragma once

// ######################################################################### //
// ### Shader.h ############################################################ //
// ### Abstracts Shaders into a non-RendererAPI specific Shader Class    ### //
// ######################################################################### //

#include <string>
#include <map>

#include "Buffer.h"

#include <glm/glm.hpp>

namespace LazyEngine {

	enum class ShaderRoutine {
		VertexShader,
		GeometryShader,
		FragmentShader
	};

	/**
	 *	Representation of a Shader program on the GPU
	 */
	class Shader {
	public:
		// Shaders have to be created with these factory methods

		/**
		 *	creates a shader from vertex-shader source and fragment-shader source while naming it 'name'
		 *	@param name: the name of the resulting shader
		 *	@param vertexSrc: code of the vertex shader
		 *	@param fragmentSrc: code of the fragment shader
		 */
		static Ref<Shader> create(const std::string& name, const std::string& vertexSrc, const std::string& fragmentSrc);
		/**
		 *	creates a shader from a bunch of shader source (that need at least a vertex- and fragmentshader
		 *  while naming it 'name'
		 *  @param name: the name of the resulting shader
		 *  @param source: A mapping from shader-routines to their sources.
		 */
		static Ref<Shader> create(const std::string& name, const std::unordered_map<ShaderRoutine, std::string>& sources);
		/**
		 *	creates a shader from a file at 'filepath'. the filepath itself will be its name
		 *	@param filepath: path to the file containing the source code for at least a vertex and a fragment shader
		 */
		static Ref<Shader> create(const std::string& filepath);
		/**
		 *	creates a shader from a file at 'filepath' while naming it 'name'
		 *	@param name: the name of the resulting shader
		 *	@param filepath: path to the file containing the source code for at least a vertex and a fragment shader
		 */
		static Ref<Shader> create(const std::string& name, const std::string& filepath);

	public:
		/**
		 *	default destructor
		 */
		virtual ~Shader() = default;

		/**
		 *	binds the shader
		 */
		virtual void bind() const = 0;
		/**
		 *	unbinds the shader
		 */
		virtual void unbind() const = 0;


		// FIXME: these will be replaced with buffer uploads later on
		// uniform methods. this is temporary and will be replaced later on

		/**
		 *	sets a uniform float
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformFloat(const std::string& name, float data) = 0;
		/**
		 *	sets a uniform vec2
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformFloat2(const std::string& name, const glm::vec2& data) = 0;
		/**
		 *	sets a uniform vec3
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformFloat3(const std::string& name, const glm::vec3& data) = 0;
		/**
		 *	sets a uniform vec4
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformFloat4(const std::string& name, const glm::vec4& data) = 0;
		/**
		 *	sets a uniform mat3
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformMat3(const std::string& name, const glm::mat3& data) = 0;
		/**
		 *	sets a uniform mat4
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformMat4(const std::string& name, const glm::mat4& data) = 0;
		/**
		 *	sets a uniform unsigned int
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformUInt(const std::string& name, unsigned int data) = 0;
		/**
		 *	sets a uniform int
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformInt(const std::string& name, int data) = 0;
		/**
		 *	sets a uniform int
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformIntArray(const std::string& name, int *data, uint32_t count) = 0;
		/**
		 *	sets a uniform ivec2
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformInt2(const std::string& name, const glm::ivec2& data) = 0;
		/**
		 *	sets a uniform ivec3
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformInt3(const std::string& name, const glm::ivec3& data) = 0;
		/**
		 *	sets a uniform ivec4
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformInt4(const std::string& name, const glm::ivec4& data) = 0;
		/**
		 *	sets a uniform bool
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformBool(const std::string& name, bool data) = 0;

		/**
		 *	reloads the source of the shader if it has been loaded from a file
		 */
		virtual bool reload() = 0;

		/**
		 *	returns the (unique) name of the shader
		 */
		virtual const std::string& getName() const = 0;
	};

	/**
	 *	Saves shaders and returns references to them.
	 *	TODO: there should be one instance owned by the renderer which should be kind of static
	 */
	class ShaderLibrary {
	public:
		/**
		 *	Adds a shader to the library.
		 *	--Will not increase the reference count of the shader!-- really?
		 *	@param shader: a reference of the shader to be added to the library.
		 */
		void add(const Ref<Shader>& shader);
		/**
		 *	Loads shader from file.
		 *	Filepath will be the unique name to be used with 'get'.
		 *	@param filepath: path to the file containing the source code for at least a vertex and a fragment shader
		 */
		Ref<Shader> load(const std::string& filepath);
		/**
		 *	loads shader from file. name will be the unique name to be used with 'get'
		 *	@param name:		unique name that will be used with 'get'
		 *	@param filepath:	shaderfile that should be loaded
		 */
		Ref<Shader> load(const std::string& name, const std::string& filepath);

		/**
		 *	get shader by its unique name
		 *	@param name: the unique name of the shader to be retrieved
		 */
		Ref<Shader> get(const std::string& name);

		/**
		 *	returns true if a shader named 'name' has already been added or loaded, false otherwise
		 *	@param name: the unique name of the shader to be checked
		 */
		bool exists(const std::string& name) const;
		
	private:
		// a map of the registered shaders
		std::unordered_map<std::string, Ref<Shader>> m_shaders;
	};

}