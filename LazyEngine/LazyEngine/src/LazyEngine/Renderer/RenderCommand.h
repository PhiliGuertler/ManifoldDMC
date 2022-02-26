#pragma once

// ######################################################################### //
// ### RenderCommand.h ##################################################### //
// ### Defines static RenderCommands like drawing, clearing, etc.        ### //
// ######################################################################### //

#include "RendererAPI.h"

namespace LazyEngine {

	// static wrapper of the selected rendererAPI
	class RenderCommand {
	public:
		/**
		 *	initializes the rendererAPI
		 */
		inline static void init() {
			s_rendererAPI->init();
		}

		/**
		 *	draws a vertex array as points using its vertex buffers and index buffer.
		 *  @param vertexArray: the vertex array to be drawn.
		 */
		inline static void drawIndexedPoints(const Ref<VertexArray>& vertexArray) {
			s_rendererAPI->drawIndexedPoints(vertexArray);
		}

		/**
		 *	draws a vertex array as wireframe using its vertex buffers and index buffer.
		 *	@param vertexArray: the vertex array to be drawn.
		 */
		inline static void drawIndexedWireframe(const Ref<VertexArray>& vertexArray) {
			s_rendererAPI->drawIndexedWireframe(vertexArray);
		}

		inline static void drawIndexedLines(const Ref<VertexArray>& vertexArray) {
			s_rendererAPI->drawIndexedLines(vertexArray);
		}

		/**
		 *	draws a vertex array using its vertex buffers and index buffer.
		 *	@param vertexArray: the vertex array to be drawn.
		 */
		inline static void drawIndexed(const Ref<VertexArray>& vertexArray) {
			s_rendererAPI->drawIndexed(vertexArray);
		}

		/**
		 *	draws a vertex array using its vertex buffers and index buffer, but it only draws the first elementCount many quads.
		 *	@param vertexArray: the vertex array to be drawn.
		 */
		inline static void drawQuadsIndexed(const Ref<VertexArray>& vertexArray, uint32_t elementCount) {
			s_rendererAPI->drawQuadsIndexed(vertexArray, elementCount);
		}

		/**
		 *	sets the clear color that will fill the window on 'clear()'
		 *	@param color: the new clear color
		 */
		inline static void setClearColor(const glm::vec4& color) {
			s_rendererAPI->setClearColor(color);
		}
		/**
		 *	clears the content of the window framebuffer by setting it to the clear color
		 */
		inline static void clear() {
			s_rendererAPI->clear();
		}

		inline static void clearDepth() {
			s_rendererAPI->clearDepth();
		}

		/**
		 *	sets the viewport of the graphics context
		 *	@param x: the x-position of the lower left corner of the viewport
		 *	@param y: the y-position of the lower left corner of the viewport
		 *	@param width: width of the viewport in pixels
		 *	@param height: height of the viewport in pixels
		 */
		inline static void setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
			s_rendererAPI->setViewport(x, y, width, height);
		}

		/**
		 *	enables depth test
		 */
		inline static void enableDepthTest() {
			s_rendererAPI->enableDepthTest();
		}
		/**
		 *	disables depth test
		 */
		inline static void disableDepthTest() {
			s_rendererAPI->disableDepthTest();
		}

		inline static void enableAlphaBlending() {
			s_rendererAPI->enableAlphaBlending();
		}

		inline static void disableAlphaBlending() {
			s_rendererAPI->disableAlphaBlending();
		}
		
		inline static void setLineWidth(float width) {
			s_rendererAPI->setLineWidth(width);
		}

		inline static void setPointSize(float radius) {
			s_rendererAPI->setPointSize(radius);
		}


	private:
		// the implementation of the rendererAPI, e.g. an OpenGL or Vulkan implementation
		static Scope<RendererAPI> s_rendererAPI;
	};

}