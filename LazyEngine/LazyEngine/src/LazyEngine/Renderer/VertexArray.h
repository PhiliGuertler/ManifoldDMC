#pragma once

// ######################################################################### //
// ### VertexArray.h ####################################################### //
// ### Defines a non-RenderingAPI specific Superclass for VertexArrays   ### //
// ######################################################################### //

#include <memory>

#include "Buffer.h"

namespace LazyEngine {

	/**
	 *	A Vertex Array is responsible of saving the states of multiple
	 *	Vertex Buffers plus their layouts and an Index Buffer.
	 */
	class VertexArray {
	public:
		/**
		 *	factory method for creating a VertexArray
		 */
		static Ref<VertexArray> create();

	public:
		/**
		 *	default destructor
		 */
		virtual ~VertexArray() = default;

		/**
		 *	binds the vertex array
		 */
		virtual void bind() const = 0;
		/**
		 *	unbinds the vertex array
		 */
		virtual void unbind() const = 0;

		/**
		 *	Adds a vertex buffer to the vertex array.
		 *	@param vertexBuffer: A reference to the vertex buffer to be added
		 */
		virtual void addVertexBuffer(const Ref<VertexBuffer>& vertexBuffer) = 0;
		/**
		 *	Sets the index buffer of the vertex array.
		 *	@param indexBuffer: A reference to the index buffer to be added
		 */
		virtual void setIndexBuffer(const Ref<IndexBuffer>& indexBuffer) = 0;

		/**
		 *	Returns a std::vector of all vertex buffers that are
		 *	registered in this vertex array
		 */
		virtual const std::vector<Ref<VertexBuffer>>& getVertexBuffers() const = 0;
		/**
		 *	Returns a reference to the indexbuffer.
		 */
		virtual const Ref<IndexBuffer>& getIndexBuffer() const = 0;

		// ################################################################# //
		// ### Experimental ################################################ //
		
		/**
		 *	adds a vertex buffer and sets it up to be used with instanced drawing
		 */
		virtual void setVertexBufferAsInstanced(const Ref<VertexBuffer>& vertexBuffer) = 0;
		
		/**
		 *	updates the references to registered intex- and vertexbuffers.
		 *  This is necessary after resizes!
		 */
		virtual void updateReferences() = 0;

		// ################################################################# //

	};

}