// ######################################################################### //
// ### OpenGLBuffer.cpp #################################################### //
// ### Implements OpenGLBuffer.h ########################################### //
// ######################################################################### //

#include "LazyEngine/gepch.h"

#include "OpenGLBuffer.h"

#include <glad/glad.h>

#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	static GLenum bufferUsageToGL(BufferUsage usage) {
		LAZYENGINE_PROFILE_FUNCTION();

		switch (usage) {
		case BufferUsage::StaticDraw:
			return GL_STATIC_DRAW;
		case BufferUsage::StaticRead:
			return GL_STATIC_READ;
		case BufferUsage::StaticCopy:
			return GL_STATIC_COPY;
		case BufferUsage::DynamicDraw:
			return GL_DYNAMIC_DRAW;
		case BufferUsage::DynamicRead:
			return GL_DYNAMIC_READ;
		case BufferUsage::DynamicCopy:
			return GL_DYNAMIC_COPY;
		case BufferUsage::StreamDraw:
			return GL_STREAM_DRAW;
		case BufferUsage::StreamRead:
			return GL_STREAM_READ;
		case BufferUsage::StreamCopy:
			return GL_STREAM_COPY;
		default:
			LAZYENGINE_CORE_ERROR("BufferUsage unknown!");
			return -1;
		}
	}

	// ##################################################################### //
	// ### OpenGLVertexBuffer ############################################## //
	// ##################################################################### //

	OpenGLVertexBuffer::OpenGLVertexBuffer(uint32_t size, BufferUsage usage) 
		: m_rendererID(0)
		, m_size(size)
		, m_layout()
		, m_usage(bufferUsageToGL(usage))
	{
		LAZYENGINE_PROFILE_FUNCTION();

		glCreateBuffers(1, &m_rendererID);
		glBindBuffer(GL_ARRAY_BUFFER, m_rendererID);
		if (size > 0) {
			glBufferData(GL_ARRAY_BUFFER, size, nullptr, m_usage);
		}
		else {
			LAZYENGINE_CORE_ERROR("Trying to create a Vertexbuffer with size {0}", size);
		}
	}

	OpenGLVertexBuffer::OpenGLVertexBuffer(float *vertices, uint32_t size, BufferUsage usage)
		: m_rendererID(0)
		, m_size(size)
		, m_layout()
		, m_usage(bufferUsageToGL(usage))
	{
		LAZYENGINE_PROFILE_FUNCTION();

		glCreateBuffers(1, &m_rendererID);
		glBindBuffer(GL_ARRAY_BUFFER, m_rendererID);
		if (size > 0) {
			glBufferData(GL_ARRAY_BUFFER, size, vertices, m_usage);
		}
		else {
			LAZYENGINE_CORE_ERROR("Trying to create a Vertexbuffer with size {0}", size);
		}
	}

	OpenGLVertexBuffer::~OpenGLVertexBuffer() {
		LAZYENGINE_PROFILE_FUNCTION();

		glDeleteBuffers(1, &m_rendererID);
	}

	void OpenGLVertexBuffer::bind() const {
		LAZYENGINE_PROFILE_FUNCTION();

		glBindBuffer(GL_ARRAY_BUFFER, m_rendererID);
	}

	void OpenGLVertexBuffer::unbind() const {
		LAZYENGINE_PROFILE_FUNCTION();

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	uint32_t OpenGLVertexBuffer::getSize() const {
		LAZYENGINE_PROFILE_FUNCTION();

		return m_size;
	}

	void OpenGLVertexBuffer::resize(uint32_t size) {
		LAZYENGINE_PROFILE_FUNCTION();

		glBindBuffer(GL_ARRAY_BUFFER, m_rendererID);
		glBufferData(GL_ARRAY_BUFFER, size, nullptr, m_usage);

		m_size = size;
	}

	void OpenGLVertexBuffer::setBufferUsage(BufferUsage usage) {
		m_usage = bufferUsageToGL(usage);
	}

	void OpenGLVertexBuffer::uploadData(float *data, uint32_t size, uint32_t offset) {
		LAZYENGINE_PROFILE_FUNCTION();

		// bind this buffer
		glBindBuffer(GL_ARRAY_BUFFER, m_rendererID);

		// check if the size of the data exceeds the allocated memory
		if (size + offset > getSize()) {
			// increase the size of this vertexbuffer
			glBufferData(GL_ARRAY_BUFFER, static_cast<size_t>(size)+static_cast<size_t>(offset), nullptr, m_usage);
			// upload the data
			glBufferSubData(GL_ARRAY_BUFFER, (GLintptr)offset, size, data);
			// update size of the buffer
			m_size = size + offset;
		}
		else {
			// just upload the data without allocating memory
			glBufferSubData(GL_ARRAY_BUFFER, (GLintptr)offset, size, data);
		}
	}


	// ##################################################################### //
	// ### OpenGLIndexBuffer ############################################### //
	// ##################################################################### //

	OpenGLIndexBuffer::OpenGLIndexBuffer(uint32_t *indices, uint32_t count, BufferUsage usage)
		: m_rendererID(0)
		, m_count(count)
		, m_usage(bufferUsageToGL(usage))
	{
		LAZYENGINE_PROFILE_FUNCTION();

		glCreateBuffers(1, &m_rendererID);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_rendererID);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(uint32_t), indices, m_usage);
	}

	OpenGLIndexBuffer::~OpenGLIndexBuffer() {
		LAZYENGINE_PROFILE_FUNCTION();

		glDeleteBuffers(1, &m_rendererID);
	}

	void OpenGLIndexBuffer::bind() const {
		LAZYENGINE_PROFILE_FUNCTION();

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_rendererID);
	}

	void OpenGLIndexBuffer::unbind() const {
		LAZYENGINE_PROFILE_FUNCTION();

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	uint32_t OpenGLIndexBuffer::getCount() const {
		LAZYENGINE_PROFILE_FUNCTION();

		return m_count;
	}

	void OpenGLIndexBuffer::setBufferUsage(BufferUsage usage) {
		m_usage = bufferUsageToGL(usage);
	}

	void OpenGLIndexBuffer::resize(uint32_t count) {
		LAZYENGINE_PROFILE_FUNCTION();

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_rendererID);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(uint32_t), nullptr, m_usage);
	
		m_count = count;
	}

	void OpenGLIndexBuffer::uploadData(uint32_t *data, uint32_t count, uint32_t offset) {
		LAZYENGINE_PROFILE_FUNCTION();

		// bind this buffer
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_rendererID);

		// check if the size of the data exceeds the allocated memory
		if (count+offset > getCount()) {
			// increase the size of this indexbuffer
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, (static_cast<size_t>(count)+static_cast<size_t>(offset))*sizeof(uint32_t), nullptr, m_usage);
			// upload the data
			glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, (GLintptr)(offset*sizeof(uint32_t)), count*sizeof(uint32_t), data);
			// update size of the buffer
			m_count = (count + offset);
		}
		else {
			// just upload the data without allocating memory
			glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, (GLintptr)(offset+sizeof(uint32_t)), count*sizeof(uint32_t), data);
		}
	}
}