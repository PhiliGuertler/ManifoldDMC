// ######################################################################### //
// ### OpenGLShader.cpp #################################################### //
// ### Implements OpenGLShader.h ########################################### //
// ######################################################################### //

#include "LazyEngine/gepch.h"

#include "OpenGLShader.h"

#include <fstream>

#include <glm/gtc/type_ptr.hpp>

// FIXME: debugging
#include <filesystem>
// :EMXIF

#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	// converts from a string to a GL shader type
	static inline GLenum shaderTypeFromString(const std::string& type) {
		LAZYENGINE_PROFILE_FUNCTION();

		if (type == "vertex") {
			return GL_VERTEX_SHADER;
		}
		else if (type == "fragment" || type == "pixel") {
			return GL_FRAGMENT_SHADER;
		}
		else if (type == "geometry") {
			return GL_GEOMETRY_SHADER;
		}
		else {
			LAZYENGINE_CORE_ERROR("Shader Type unknown in OpenGL: '{0}'", type);
			return 0;
		}
	}

	// converts from a gl type to a string representation of that type
	static inline std::string shaderStringFromType(const GLenum& type) {
		LAZYENGINE_PROFILE_FUNCTION();

		if (type == GL_VERTEX_SHADER) {
			return "Vertex Shader";
		}
		else if (type == GL_FRAGMENT_SHADER) {
			return "Fragment Shader";
		}
		else if (type == GL_GEOMETRY_SHADER) {
			return "Geometry Shader";
		}
		else {
			LAZYENGINE_CORE_ERROR("Shader Type Unknown: '{0}'", type);
			return "<Shader not Supported>";
		}
	}

	// helper function for the shader constructor
	static inline GLuint createShader(const std::string& source, const GLenum& shaderType) {
		LAZYENGINE_PROFILE_FUNCTION();

		// create an empty shader handle
		GLuint shader = glCreateShader(shaderType);

		// send shader source code to GL
		const GLchar *sourceCode = (const GLchar *)source.c_str();
		glShaderSource(shader, 1, &sourceCode, 0);

		// compile shader
		glCompileShader(shader);

		// check if the compilation was successful
		GLint isCompiled = 0;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
		if (isCompiled == GL_FALSE) {
			// shader compilation failed, print error message to console
			// first, get the length of the error message
			GLint maxLength = 0;
			glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

			// then query the actual message
			std::vector<GLchar> message(maxLength);
			glGetShaderInfoLog(shader, maxLength, &maxLength, &message[0]);

			// delete shader as it is not needed in an invalid state
			glDeleteShader(shader);

			// print the message as error in the console
			LAZYENGINE_CORE_ERROR("{0}", message.data());
			std::string shaderTypeName = shaderStringFromType(shaderType);
			LAZYENGINE_CORE_ERROR("{0} Compilation failed!", shaderTypeName);
			LAZYENGINE_CORE_ASSERT(false, "Compilation failed!");
			return -1;
		}

		return shader;
	}

	// constructor
	OpenGLShader::OpenGLShader(const std::string& name, const std::string& vertexSrc, const std::string& fragmentSrc) 
		: m_programID(0)
		, m_name(name)
		, m_filepath("")
	{
		LAZYENGINE_PROFILE_FUNCTION();

		std::unordered_map<GLenum, std::string> shaderSources;
		shaderSources[GL_VERTEX_SHADER] = vertexSrc;
		shaderSources[GL_FRAGMENT_SHADER] = fragmentSrc;
		compile(shaderSources);
	}

	OpenGLShader::OpenGLShader(const std::string& name, const std::unordered_map<ShaderRoutine, std::string>& sources)
		: m_programID(0)
		, m_name(name)
		, m_filepath("")
	{
		LAZYENGINE_PROFILE_FUNCTION();
		std::unordered_map<GLenum, std::string> shaderSources;
		shaderSources[GL_VERTEX_SHADER] = sources.at(ShaderRoutine::VertexShader);
		shaderSources[GL_FRAGMENT_SHADER] = sources.at(ShaderRoutine::FragmentShader);
		
		if (sources.find(ShaderRoutine::GeometryShader) != sources.end()) {
			shaderSources[GL_GEOMETRY_SHADER] = sources.at(ShaderRoutine::GeometryShader);
		}
		// TODO: extend me in the future (e.g. for Tesselation Shaders, etc)

		compile(shaderSources);
	}


	OpenGLShader::OpenGLShader(const std::string& filepath) 
		: m_programID(0)
		, m_name()
		, m_filepath(filepath)
	{
		LAZYENGINE_PROFILE_FUNCTION();

		// extract name from filename without extension and previous folders:
		// eg. assets/shaders/Texture.glsl --> Texture
		
		// if there is no slash, start from the first character, otherwise move one after the last slash
		auto lastSlash = filepath.find_last_of("/\\");
		lastSlash = (lastSlash == std::string::npos) ? 0 : (lastSlash + 1);
		
		// find last dot that marks the file extension
		auto lastDot = filepath.rfind(".");
		std::string::size_type charCount;
		if (lastDot == std::string::npos || lastDot < lastSlash) {
			// there is no file extension or a folder contains a dot that will be removed anyways
			charCount = filepath.size() - lastSlash;
		}
		else {
			// there is a file extension which should be removed
			charCount = lastDot - lastSlash;
		}
		m_name = filepath.substr(lastSlash, charCount);

		std::string fileContents = readFile(filepath);
		auto shaderSources = preprocess(fileContents);
		compile(shaderSources);
	}

	OpenGLShader::OpenGLShader(const std::string& name, const std::string& filepath)
		: m_programID(0)
		, m_name(name)
		, m_filepath(filepath)
	{
		LAZYENGINE_PROFILE_FUNCTION();

		std::string fileContents = readFile(filepath);
		if (fileContents.size() > 0) {
			auto shaderSources = preprocess(fileContents);
			compile(shaderSources);
		}
	}

	OpenGLShader::~OpenGLShader() {
		LAZYENGINE_PROFILE_FUNCTION();

		glDeleteProgram(m_programID);
	}

	void OpenGLShader::bind() const {
		LAZYENGINE_PROFILE_FUNCTION();

		glUseProgram(m_programID);
	}

	void OpenGLShader::unbind() const {
		LAZYENGINE_PROFILE_FUNCTION();

		glUseProgram(0);
	}

	// uniform methods
	// TODO: in the future, when all uniforms and types are known, do type checks
	void OpenGLShader::uniformFloat(const std::string& name, float data) {
		LAZYENGINE_PROFILE_FUNCTION();

		GLint location = glGetUniformLocation(m_programID, name.c_str());
		glUniform1f(location, data);
	}
	void OpenGLShader::uniformFloat2(const std::string& name, const glm::vec2& data) {
		LAZYENGINE_PROFILE_FUNCTION();

		GLint location = glGetUniformLocation(m_programID, name.c_str());
		glUniform2fv(location, 1, glm::value_ptr(data));
	}
	void OpenGLShader::uniformFloat3(const std::string& name, const glm::vec3& data) {
		LAZYENGINE_PROFILE_FUNCTION();

		GLint location = glGetUniformLocation(m_programID, name.c_str());
		glUniform3fv(location, 1, glm::value_ptr(data));
	}
	void OpenGLShader::uniformFloat4(const std::string& name, const glm::vec4& data) {
		LAZYENGINE_PROFILE_FUNCTION();

		GLint location = glGetUniformLocation(m_programID, name.c_str());
		glUniform4fv(location, 1, glm::value_ptr(data));
	}
	void OpenGLShader::uniformMat3(const std::string& name, const glm::mat3& data) {
		LAZYENGINE_PROFILE_FUNCTION();

		GLint location = glGetUniformLocation(m_programID, name.c_str());
		glUniformMatrix3fv(location, 1, GL_FALSE, glm::value_ptr(data));
	}
	void OpenGLShader::uniformMat4(const std::string& name, const glm::mat4& data) {
		LAZYENGINE_PROFILE_FUNCTION();

		GLint location = glGetUniformLocation(m_programID, name.c_str());
		glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(data));
	}
	void OpenGLShader::uniformUInt(const std::string& name, unsigned int data) {
		LAZYENGINE_PROFILE_FUNCTION();

		GLint location = glGetUniformLocation(m_programID, name.c_str());
		glUniform1ui(location, data);
	}
	void OpenGLShader::uniformInt(const std::string& name, int data) {
		LAZYENGINE_PROFILE_FUNCTION();

		GLint location = glGetUniformLocation(m_programID, name.c_str());
		glUniform1i(location, data);
	}
	void OpenGLShader::uniformIntArray(const std::string& name, int *data, uint32_t count) {
		LAZYENGINE_PROFILE_FUNCTION();

		GLint location = glGetUniformLocation(m_programID, name.c_str());
		glUniform1iv(location, count, data);
	}
	void OpenGLShader::uniformInt2(const std::string& name, const glm::ivec2& data) {
		LAZYENGINE_PROFILE_FUNCTION();

		GLint location = glGetUniformLocation(m_programID, name.c_str());
		glUniform4iv(location, 1, glm::value_ptr(data));
	}
	void OpenGLShader::uniformInt3(const std::string& name, const glm::ivec3& data) {
		LAZYENGINE_PROFILE_FUNCTION();

		GLint location = glGetUniformLocation(m_programID, name.c_str());
		glUniform4iv(location, 1, glm::value_ptr(data));
	}
	void OpenGLShader::uniformInt4(const std::string& name, const glm::ivec4& data) {
		LAZYENGINE_PROFILE_FUNCTION();

		GLint location = glGetUniformLocation(m_programID, name.c_str());
		glUniform4iv(location, 1, glm::value_ptr(data));
	}
	void OpenGLShader::uniformBool(const std::string& name, bool data) {
		LAZYENGINE_PROFILE_FUNCTION();

		GLint location = glGetUniformLocation(m_programID, name.c_str());
		glUniform1i(location, static_cast<int>(data));
	}

	bool OpenGLShader::reload() {
		LAZYENGINE_PROFILE_FUNCTION();

		if (m_filepath != "") {
			std::string fileContents = readFile(m_filepath);
			auto shaderSources = preprocess(fileContents);
			return compile(shaderSources);
		}
		return false;
	}

	// ##################################################################### //
	// ### private Functions ############################################### //

	std::string OpenGLShader::readFile(const std::string& filepath) {
		LAZYENGINE_PROFILE_FUNCTION();

		std::string fileContents;
		// FIXME: create a file management system to be platform independent!
		// this part will be different e.g. on android or linux!
		std::ifstream input(filepath, std::ios::in | std::ios::binary);
		if (input) {
			// go to the end of the file to get its size
			input.seekg(0, std::ios::end);
			// resize fileContents to match the file's size
			fileContents.resize(input.tellg());
			// jump back to the beginning of the file
			input.seekg(0, std::ios::beg);
			// read in the file
			input.read(&fileContents[0], fileContents.size());
			// close file stram
			input.close();
		}
		else {
			LAZYENGINE_CORE_ERROR("Could not open file \"{0}\"", filepath);
		}
		return fileContents;
	}

	// handles all special metadata in a shader file written for this engine
	std::unordered_map<GLenum, std::string> OpenGLShader::preprocess(const std::string& source) {
		LAZYENGINE_PROFILE_FUNCTION();

		std::unordered_map<GLenum, std::string> shaderSources;

		// this string will be stripped of '#comment' parts
		std::string preppedSource = source;

		// remove #comment from source string
		{
			const char *commentToken = "#comment";
			size_t commentTokenLength = strlen(commentToken);
			size_t commentPosition = source.find(commentToken, 0);

			bool inCommentArea = false;
			size_t commentBeginPosition = 0;

			while (commentPosition != std::string::npos) {
				// find end of line
				size_t eol = preppedSource.find_first_of("\r\n", commentPosition);
				// move to the word after '#comment ' 
				size_t begin = commentPosition + commentTokenLength + 1;
				// get the word after '#comment'
				std::string commentCommand = preppedSource.substr(begin, eol - begin);
				LAZYENGINE_CORE_ASSERT(commentCommand == "begin" || commentCommand == "end", "Syntax Error: #comment must be used like either '#comment begin' or '#comment end'");

				// this is the beginning of a comment block
				if (commentCommand == "begin") {
					if (!inCommentArea) {
						commentBeginPosition = commentPosition;
						inCommentArea = true;
					}
					else {
						LAZYENGINE_CORE_ASSERT(false, "Syntax Error: '#comment' cannot be nested!");
					}
				}
				// this is the end of a comment block, erase it
				if (commentCommand == "end") {
					if (inCommentArea) {
						size_t beginNextLine = preppedSource.find_first_not_of("\r\n", eol);
						preppedSource.erase(commentBeginPosition, beginNextLine-1 - commentBeginPosition);
						inCommentArea = false;
						commentPosition = commentBeginPosition;
					}
					else {
						LAZYENGINE_CORE_ASSERT(false, "Syntax Error: '#comment' ends without beginning!");
					}
				}
				
				// find the next '#comment' token
				size_t nextLinePosition = preppedSource.find_first_not_of("\r\n", eol);
				commentPosition = preppedSource.find(commentToken, nextLinePosition);

			}
		}

		// extract shaders by type
		{
			const char *typeToken = "#type";
			size_t typeTokenLength = strlen(typeToken);
			size_t typePosition = preppedSource.find(typeToken, 0);

			while (typePosition != std::string::npos) {
				// find end of line
				size_t eol = preppedSource.find_first_of("\r\n", typePosition);
				LAZYENGINE_CORE_ASSERT(eol != std::string::npos, "Syntax Error: there must be some lines of code after #type")
				// move to the word after '#comment ' 
				size_t begin = typePosition + typeTokenLength + 1;
				// get the word after '#comment'
				std::string type = preppedSource.substr(begin, eol - begin);
				GLenum shaderType = shaderTypeFromString(type);
				LAZYENGINE_CORE_ASSERT(shaderType, "Syntax Error: Only the types 'vertex', 'geometry', 'fragment' and its alternative 'pixel' are allowed.");

				// find the next '#comment' token
				// get position of next line
				size_t nextLinePosition = preppedSource.find_first_not_of("\r\n", eol);
				typePosition = preppedSource.find(typeToken, nextLinePosition);
				shaderSources[shaderType] = preppedSource.substr(nextLinePosition, typePosition - (nextLinePosition == std::string::npos ? source.size()-1 : nextLinePosition));
			}
		}

		return shaderSources;
	}

	// compiles a bunch of shader sources
	bool OpenGLShader::compile(const std::unordered_map<GLenum, std::string>& shaderSources) {
		LAZYENGINE_PROFILE_FUNCTION();

		// create new program in OpenGL
		GLuint program = glCreateProgram();
		
		// create a list for shaders (vertex, geometry, ...)
		// currently this only allows up to 10 shaders at once
		LAZYENGINE_CORE_ASSERT(shaderSources.size() <= 10, "Only up to 10 shaders at once are supported");
		std::array<GLuint, 10> glShaderIDs;

		// create the individual shaders and add them to the list
		int index = 0;
		for (auto& keyVal : shaderSources) {
			GLenum shaderType = keyVal.first;
			const std::string& source = keyVal.second;

			GLuint shader = createShader(source, shaderType);

			glAttachShader(program, shader);
			glShaderIDs[index++] = (shader);
		}

		// link the program
		glLinkProgram(program);

		// check if linking was successful
		GLint isLinked = 0;
		glGetProgramiv(program, GL_LINK_STATUS, (int *)&isLinked);
		if (isLinked == GL_FALSE) {
			// Linking error occured

			// get length of error message
			GLint maxLength = 0;
			glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

			// get the actual error message
			std::vector<GLchar> message(maxLength);
			glGetProgramInfoLog(program, maxLength, &maxLength, &message[0]);

			// delete the invalid m_programID
			glDeleteProgram(program);

			// delete shaders aswell
			for (auto shaderID : glShaderIDs) {
				glDeleteShader(shaderID);
			}

			// print error message to the console
			LAZYENGINE_CORE_ASSERT(false, "Shader Linking failed: {0}", message.data());

			// return failure of compilation process
			return false;
		}

		// detach shaders after successful linking
		for (auto shaderID : glShaderIDs) {
			glDetachShader(program, shaderID);
			glDeleteShader(shaderID);
		}

		// only override m_programID if the shader is valid in its entirety
		m_programID = program;

		// compilation succeded
		return true;
	}
}