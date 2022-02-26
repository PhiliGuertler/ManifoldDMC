#pragma once

// ######################################################################### //
// ### OpenGLShader.h ###################################################### //
// ### Abstracts OpenGL Shaders into a Shader Class                      ### //
// ######################################################################### //

#include "LazyEngine/Renderer/Shader.h"
#include <glad/glad.h>

namespace LazyEngine {

	/**
	 *	Representation of an OpenGL Shader program on the GPU
	 */
	class OpenGLShader : public Shader {
	public:
		/**
		 *	constructor
		 *	@param name: the name of the resulting shader
		 *	@param vertexSrc: code of the vertex shader
		 *	@param fragmentSrc: code of the fragment shader
		 */
		OpenGLShader(const std::string& name, const std::string& vertexSrc, const std::string& fragmentSrc);
		/**
		 *	constructor
		 *  @param name: the name of the resulting shader
		 *  @param source: A map of source codes of the shader-programs
		 */
		OpenGLShader(const std::string& name, const std::unordered_map<ShaderRoutine, std::string>& sources);
		/**
		 *	constructor
		 *	@param filepath: path to the file containing the source code for at least a vertex and a fragment shader
		 */
		OpenGLShader(const std::string& filepath);
		/**
		 *	constructor
		 *	@param name: the name of the resulting shader
		 *	@param filepath: path to the file containing the source code for at least a vertex and a fragment shader
		 */
		OpenGLShader(const std::string& name, const std::string& filepath);
		/**
		 *	destructor
		 */
		virtual ~OpenGLShader();

		/**
		 *	binds the shader
		 */
		virtual void bind() const override;
		/**
		 *	unbinds the shader
		 */
		virtual void unbind() const override;

		// uniform methods

		/**
		 *	sets a uniform float
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformFloat(const std::string& name, float data) override;
		/**
		 *	sets a uniform vec2
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformFloat2(const std::string& name, const glm::vec2& data) override;
		/**
		 *	sets a uniform vec3
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformFloat3(const std::string& name, const glm::vec3& data) override;
		/**
		 *	sets a uniform vec4
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformFloat4(const std::string& name, const glm::vec4& data) override;
		/**
		 *	sets a uniform mat3
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformMat3(const std::string& name, const glm::mat3& data) override;
		/**
		 *	sets a uniform mat4
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformMat4(const std::string& name, const glm::mat4& data) override;
		/**
		 *	sets a uniform unsigned int
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformUInt(const std::string& name, unsigned int data) override;
		/**
		 *	sets a uniform int
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformInt(const std::string& name, int data) override;
		/**
		 *	sets a uniform int
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformIntArray(const std::string& name, int *data, uint32_t count) override;
		/**
		 *	sets a uniform ivec2
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformInt2(const std::string& name, const glm::ivec2& data) override;
		/**
		 *	sets a uniform ivec3
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformInt3(const std::string& name, const glm::ivec3& data) override;
		/**
		 *	sets a uniform ivec4
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformInt4(const std::string& name, const glm::ivec4& data) override;
		/**
		 *	sets a uniform bool
		 *	@param name: the name of the uniform variable in the shader code
		 *	@param data: the data to be uploaded
		 */
		virtual void uniformBool(const std::string& name, bool data) override;

		/**
		 *	reloads the source of the shader if it has been loaded from a file
		 */
		virtual bool reload() override;

		/**
		 *	returns the (unique) name of the shader
		 */
		virtual const std::string& getName() const override { return m_name; }

	private:
		/**
		 *	Reads a file from disk.
		 *	FIXME: this should be implemented in a file management system
		 *		to be platform independent
		 *	@param filepath: the path to the shader source to be read
		 *	@returns the content of the file as a single string
		 */
		std::string readFile(const std::string& filepath);
		/**
		 *	Preprocesses the source code of a shader.
		 *	@param source: the output of a readFile()-operation
		 *	@returns a map with the different shaders, e.g. a vertex- and a fragment-shader
		 */
		std::unordered_map<GLenum, std::string> preprocess(const std::string& source);
		/**
		 *	compiles the output of preprocess
		 *	@param shaderSources: the output of a compile()-operation
		 *	@returns true if the compilation was successful
		 */
		bool compile(const std::unordered_map<GLenum, std::string>& shaderSources);

		// the OpenGL handle of the shader
		uint32_t m_programID;

		// unique name of the shader for use in ShaderLibrary, for example
		std::string m_name;

		// for reloading purposes the filepath will be saved:
		std::string m_filepath;
	};
}