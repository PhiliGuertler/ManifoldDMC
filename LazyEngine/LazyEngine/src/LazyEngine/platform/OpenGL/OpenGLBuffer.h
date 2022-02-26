#pragma once

// ######################################################################### //
// ### OpenGLBuffer.cpp #################################################### //
// ### Implements LazyEngine/Renderer/Buffer.h for OpenGL ################## //
// ######################################################################### //

#include "LazyEngine/Renderer/Buffer.h"

namespace LazyEngine {

	class OpenGLVertexBuffer : public VertexBuffer {
	public:

		/**
		 *	constructor
		 *	creates a vertexbuffer without any data but with allocated memory
		 *  @param size: The size of the resulting buffer in bytes
		 *  @param usage: The intended usage of the buffer
		 */
		OpenGLVertexBuffer(uint32_t size, BufferUsage usage);
		
		/**
		 *	constructor
		 *	creates a vertexbuffer using OpenGL
		 *  @param vertices: A pointer to an array of floats that define vertices
		 *  @param size: The size of the vertex-data in bytes
		 *  @param usage: The intended usage of the buffer
		 */
		OpenGLVertexBuffer(float *vertices, uint32_t size, BufferUsage usage);


		/**
		 *	destructor
		 */
		virtual ~OpenGLVertexBuffer();

		/**
		 *	binds the buffer
		 */
		virtual void bind() const override;
		/**
		 *	unbinds the buffer
		 */
		virtual void unbind() const override;

		/**
		 *	set the layout of this specific vertexbuffer, e.g. 3 floats position, then 4 floats color.
		 *	@param layout: the actual layout
		 */
		virtual void setLayout(const BufferLayout& layout) override { m_layout = layout; }
		//virtual const BufferLayout& getLayout() const override { return m_layout; }

		/**
		 *	returns the size of this buffer's content in bytes
		 */
		virtual uint32_t getSize() const override;

		/**
		 *	returns the buffer layout that can be modified afterwards
		 */
		virtual BufferLayout& getLayout() override { return m_layout; }

		/**
		 *	sets the buffer usage of this buffer
		 *	@param usage: the new usage of this buffer
		 */
		virtual void setBufferUsage(BufferUsage usage) override;

		/**
		 *	resizes the buffer on the gpu
		 *	@param size: the new size of the buffer in bytes
		 */
		virtual void resize(uint32_t size) override;

		/**
		 *	uploads data to the gpu.
		 *	@param data: a pointer to the data to be uploaded
		 *	@param size: the size of the data in bytes
		 *	@param offset: an offset (in elements) at which the data should be inserted
		 *	@param usage: the usage of the buffer, e.g. for meshes this might be
		 *		StaticDraw, but particles will be better off with DynamicDraw.
		 */
		virtual void uploadData(float *data, uint32_t size, uint32_t offset = 0) override;

	protected:
		// HACK: should be RendererType
		uint32_t m_rendererID;
		// KCAH

		// the size of the buffer on the gpu in bytes
		uint32_t m_size;
		// the layout of the buffer
		BufferLayout m_layout;
		// usage of this buffer
		uint32_t m_usage;
	};

	/**
	 *	Wraps an OpenGL specific IndexBuffer 
	 */
	class OpenGLIndexBuffer : public IndexBuffer {
	public:
		/**
		 *	constructor
		 *  @param vertices: A pointer to an array of uint32_t's that define indices
		 *  @param count: The amount of uint32_t that should be uploaded into the buffer
		 *  @param usage: The intended usage of the buffer
		 */
		OpenGLIndexBuffer(uint32_t *indices, uint32_t size, BufferUsage usage);
		/**
		 *	destructor
		 */
		virtual ~OpenGLIndexBuffer();

		/**
		 *	binds the index buffer
		 */
		virtual void bind() const override;
		/**
		 *	unbinds the index buffer
		 */
		virtual void unbind() const override;

		/**
		 *	returns the amount of indices that are stored in this buffer
		 */
		virtual uint32_t getCount() const override;

		/**
		 *	sets the buffer usage of this buffer
		 *	@param usage: the new usage of this buffer
		 */
		virtual void setBufferUsage(BufferUsage usage) override;

		/**
		 *	resizes the buffer on the gpu
		 *	@param count: the new count of indices of the buffer in #indices
		 */
		virtual void resize(uint32_t count) override;

		/**
		 *	uploads data to the gpu.
		 *	@param data: a pointer to the data to be uploaded
		 *	@param count: the amount of elements
		 *	@param offset: an offset (in bytes) at which the data should be inserted
		 *	@param usage: the usage of the buffer, e.g. for meshes this might be
		 *		StaticDraw, but particles will be better off with DynamicDraw.
		 */
		virtual void uploadData(uint32_t *data, uint32_t count, uint32_t offset = 0) override;

	protected:
		// HACK: should be RendererType
		uint32_t m_rendererID;
		// KCAH

		// the amount of incides stored in this buffer
		uint32_t m_count;
		// Usage of this buffer
		uint32_t m_usage;
	};

}