#pragma once

// ######################################################################### //
// ### OpenGLRendererAPI.cpp ############################################### //
// ### implements RendererAPI for OpenGL.                                ### //
// ######################################################################### //

#include "LazyEngine/Renderer/RendererAPI.h"

namespace LazyEngine {

	/**
	 *	A collection of opengl specific functions
	 */
	class OpenGLRendererAPI : public RendererAPI {
	public:
		/**
		 *	constructor
		 */
		OpenGLRendererAPI() = default;
		/**
		 *	destructor
		 */
		virtual ~OpenGLRendererAPI() = default;

		/**
		 *	initializes the rendererAPI
		 */
		virtual void init() override;

		/**
		 *	sets the clear color that will fill the window on 'clear()'
		 *	@param color: the new clear color
		 */
		virtual void setClearColor(const glm::vec4& color) override;
		/**
		 *	clears the content of the window framebuffer by setting it to the clear color
		 */
		virtual void clear() override;
		/**
		 *	clears the content of the window framebuffer by setting it to the clear color
		 */
		virtual void clearDepth() override;

		/**
		 *	draws a vertex array as points using its vertex buffers and index buffer.
		 *  @param vertexArray: the vertex array to be drawn.
		 */
		virtual void drawIndexedPoints(const Ref<VertexArray>& vertexArray) override;

		/**
		 *	draws a vertex array as wireframe using its vertex buffers and index buffer.
		 *  @param vertexArray: the vertex array to be drawn.
		 */
		virtual void drawIndexedWireframe(const Ref<VertexArray>& vertexArray) override;

		/**
		 *	draws a vertex array as a list of lines using its vertex and index buffer.
		 *	@param vertexArray: the vertex array to be drawn
		 */
		virtual void drawIndexedLines(const Ref<VertexArray>& vertexArray) override;

		/**
		 *	draws a vertex array using its vertex buffers and index buffer.
		 *	@param vertexArray: the vertex array to be drawn.
		 */
		virtual void drawIndexed(const Ref<VertexArray>& vertexArray) override;

		/**
		 *	draws a vertex array using its vertex buffers and index buffer, but it only draws the first elementCount many quads.
		 *	@param vertexArray: the vertex array to be drawn.
		 */
		virtual void drawQuadsIndexed(const Ref<VertexArray>& vertexArray, uint32_t elementCount) override;

		/**
		 *	sets the viewport of the graphics context
		 *	@param x: the x-position of the lower left corner of the viewport
		 *	@param y: the y-position of the lower left corner of the viewport
		 *	@param width: width of the viewport in pixels
		 *	@param height: height of the viewport in pixels
		 */
		virtual void setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) override;

		/**
		 *	enables depth test
		 */
		virtual void enableDepthTest() override;
		/**
		 *	disables depth test
		 */
		virtual void disableDepthTest() override;

		virtual void enableAlphaBlending() override;

		virtual void disableAlphaBlending() override;

		virtual void setLineWidth(float width) override;

		virtual void setPointSize(float radius) override;

	};

}