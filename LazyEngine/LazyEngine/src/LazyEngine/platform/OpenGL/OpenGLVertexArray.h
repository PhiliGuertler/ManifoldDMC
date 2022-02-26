#pragma once

// ######################################################################### //
// ### OpenGLVertexArray.h ################################################# //
// ### Implements VertexArray.h for OpenGL                               ### //
// ######################################################################### //

#include "LazyEngine/Renderer/VertexArray.h"

namespace LazyEngine {

	/**
	 *	A Vertex Array is responsible of saving the states of multiple
	 *	Vertex Buffers plus their layouts and an Index Buffer.
	 */
	class OpenGLVertexArray : public VertexArray {
	public:
		/**
		 *	constructor
		 */
		OpenGLVertexArray();
		/**
		 *	destructor
		 */
		virtual ~OpenGLVertexArray();

		/**
		 *	binds the vertex array
		 */
		virtual void bind() const override;
		/**
		 *	unbinds the vertex array
		 */
		virtual void unbind() const override;

		/**
		 *	Adds a vertex buffer to the vertex array.
		 *	@param vertexBuffer: A reference to the vertex buffer to be added
		 */
		virtual void addVertexBuffer(const Ref<VertexBuffer>& vertexBuffer) override;
		/**
		 *	Sets the index buffer of the vertex array.
		 *	@param indexBuffer: A reference to the index buffer to be added
		 */
		virtual void setIndexBuffer(const Ref<IndexBuffer>& indexBuffer) override;

		/**
		 *	Returns a std::vector of all vertex buffers that are
		 *	registered in this vertex array
		 */
		virtual const std::vector<Ref<VertexBuffer>>& getVertexBuffers() const override { return m_vertexBuffers; }
		/**
		 *	Returns a reference to the indexbuffer.
		 */
		virtual const Ref<IndexBuffer>& getIndexBuffer() const override { return m_indexBuffer; }

		// ################################################################# //
		// ### Experimental ################################################ //

		/**
		 *	adds a vertex buffer and sets it up to be used with instanced drawing
		 */
		virtual void setVertexBufferAsInstanced(const Ref<VertexBuffer>& vertexBuffer) override;

		/**
		 *	updates the references to registered intex- and vertexbuffers.
		 *  This is necessary after resizes!
		 */
		virtual void updateReferences() override;

		// ################################################################# //
	private:
		// the opengl handle to the vertex array
		uint32_t m_rendererID;

		// a list of registered vertex buffers
		std::vector<Ref<VertexBuffer>> m_vertexBuffers;
		// counts the amount of input locations
		uint32_t m_vertexAttribIndex;
		// a reference to the index buffer
		Ref<IndexBuffer> m_indexBuffer;

	};

}