// ######################################################################### //
// ### OpenGLVertexArray.cpp ############################################### //
// ### implements OpenGLVertexArray.h                                    ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"

#include "OpenGLVertexArray.h"

#include <glad/glad.h>

#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	static GLenum ShaderDataTypeToOpenGLBaseType(ShaderDataType type) {
		LAZYENGINE_PROFILE_FUNCTION();

		switch (type) {
		case ShaderDataType::Float:
		case ShaderDataType::Float2:
		case ShaderDataType::Float3:
		case ShaderDataType::Float4:
		case ShaderDataType::Mat3:
		case ShaderDataType::Mat4:
			return GL_FLOAT;
		case ShaderDataType::Int:
		case ShaderDataType::Int2:
		case ShaderDataType::Int3:
		case ShaderDataType::Int4:
			return GL_INT;
		case ShaderDataType::Bool:
			return GL_BOOL;
		default:
			LAZYENGINE_CORE_ASSERT(false, "Unknown ShaderDataType!");
			return 0;
		}
	}


	OpenGLVertexArray::OpenGLVertexArray()
		: m_rendererID(0)
		, m_vertexAttribIndex(0)
	{
		LAZYENGINE_PROFILE_FUNCTION();

		glCreateVertexArrays(1, &m_rendererID);
	}

	OpenGLVertexArray::~OpenGLVertexArray() {
		LAZYENGINE_PROFILE_FUNCTION();

		m_vertexBuffers.clear();
		m_indexBuffer = nullptr;

		glDeleteVertexArrays(1, &m_rendererID);
	}

	void OpenGLVertexArray::bind() const {
		LAZYENGINE_PROFILE_FUNCTION();

		glBindVertexArray(m_rendererID);
	}

	void OpenGLVertexArray::unbind() const {
		LAZYENGINE_PROFILE_FUNCTION();

		glBindVertexArray(0);
	}

	void OpenGLVertexArray::addVertexBuffer(const Ref<VertexBuffer>& vertexBuffer) {
		LAZYENGINE_PROFILE_FUNCTION();

		LAZYENGINE_CORE_ASSERT(vertexBuffer->getLayout().getElements().size() != 0, "Vertex Buffer has no Layout!");

		bind();
		vertexBuffer->bind();

		// --- set up vertexbuffer layout --- //
		auto& layout = vertexBuffer->getLayout();
		for (auto& element : layout) {
			glEnableVertexAttribArray(m_vertexAttribIndex);
			glVertexAttribPointer(m_vertexAttribIndex,
				element.getComponentCount(),
				ShaderDataTypeToOpenGLBaseType(element.type),
				element.normalized ? GL_TRUE : GL_FALSE,
				layout.getStride(),
				(const void *)(intptr_t)element.offset);

			// ###################################################################### //
			// ### Experimental ##################################################### //
			element.location = m_vertexAttribIndex;
			// ###################################################################### //
			
			++m_vertexAttribIndex;
		}
		// ---------------------------------- //

		m_vertexBuffers.push_back(vertexBuffer);
	}

	void OpenGLVertexArray::setIndexBuffer(const Ref<IndexBuffer>& indexBuffer) {
		LAZYENGINE_PROFILE_FUNCTION();

		glBindVertexArray(m_rendererID);
		indexBuffer->bind();

		m_indexBuffer = indexBuffer;
	}

	// ##################################################################### //
	// ### Experimental #################################################### //
	void OpenGLVertexArray::setVertexBufferAsInstanced(const Ref<VertexBuffer>& vertexBuffer) {
		LAZYENGINE_PROFILE_FUNCTION();

		glBindVertexArray(m_rendererID);
		auto& layout = vertexBuffer->getLayout();

		for (const auto& element : layout) {
			glVertexAttribDivisor(element.location, 1);
		}
	}

	void OpenGLVertexArray::updateReferences() {
		LAZYENGINE_PROFILE_FUNCTION();

		bind();

		// update vertex buffers
		for (auto& vertexBuffer: m_vertexBuffers) {
			vertexBuffer->bind();
			// --- set up vertexbuffer layout --- //
			auto& layout = vertexBuffer->getLayout();
			for (auto& element : layout) {
				glEnableVertexAttribArray(m_vertexAttribIndex);
				glVertexAttribPointer(m_vertexAttribIndex,
					element.getComponentCount(),
					ShaderDataTypeToOpenGLBaseType(element.type),
					element.normalized ? GL_TRUE : GL_FALSE,
					layout.getStride(),
					(const void*)(intptr_t)element.offset);

				// ###################################################################### //
				// ### Experimental ##################################################### //
				element.location = m_vertexAttribIndex;
				// ###################################################################### //

				++m_vertexAttribIndex;
			}
		}

		m_indexBuffer->bind();
	}

	// ##################################################################### //

}