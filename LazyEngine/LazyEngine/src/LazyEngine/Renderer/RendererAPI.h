#pragma once

// ######################################################################### //
// ### RendererAPI.h ####################################################### //
// ### Defines the API superclass that has to be implemented for every   ### //
// ### supported Graphics-API, as well as an enum listing said APIs.     ### //
// ######################################################################### //

#include <glm/glm.hpp>
#include "VertexArray.h"

namespace LazyEngine {

	/**
	 *	A collection of renderer specific functions
	 */
	class RendererAPI {
	public:
		enum class API {
			None = 0,
			OpenGL = 1,
			DirectX = 2,
			Vulcan = 3
		};

		/**
		 *	Lists all Multi-Sampling Anti-Aliasing options supported by LazyEngine
		 */
		enum class MSAA : int {
			Off = 0,
			x4 = 4,
			x8 = 8,
			x16 = 16,
		};

	public:
		/**
		 *	returns the implementation api
		 */
		inline static API getAPI() { return s_API; }

	public:
		/**
		 *	default destructor
		 */
		virtual ~RendererAPI() = default;

		/**
		 *	initializes the rendererAPI
		 */
		virtual void init() = 0;

		/**
		 *	sets the clear color that will fill the window on 'clear()'
		 *	@param color: the new clear color
		 */
		virtual void setClearColor(const glm::vec4& color) = 0;
		/**
		 *	clears the content of the window framebuffer by setting it to the clear color
		 */
		virtual void clear() = 0;
		/**
		 *	clears the content of the window framebuffer by setting it to the clear color
		 */
		virtual void clearDepth() = 0;

		/**
		 *	draws a vertex array as points using its vertex buffers and index buffer.
		 *  @param vertexArray: the vertex array to be drawn.
		 */
		virtual void drawIndexedPoints(const Ref<VertexArray>& vertexArray) = 0;
		
		/**
		 *	draws a vertex array as wireframe using its vertex buffers and index buffer.
		 *  @param vertexArray: the vertex array to be drawn.
		 */
		virtual void drawIndexedWireframe(const Ref<VertexArray>& vertexArray) = 0;

		/**
		 *	draws a vertex array as lines using its vertex buffers and index buffer.
		 *  @param vertexArray: the vertex array to be drawn.
		 */
		virtual void drawIndexedLines(const Ref<VertexArray>& vertexArray) = 0;

		/**
		 *	draws a vertex array using its vertex buffers and index buffer.
		 *	@param vertexArray: the vertex array to be drawn.
		 */
		virtual void drawIndexed(const Ref<VertexArray>& vertexArray) = 0;

		/**
		 *	draws a vertex array using its vertex buffers and index buffer, but it only draws the first elementCount many quads.
		 *	@param vertexArray: the vertex array to be drawn.
		 */
		virtual void drawQuadsIndexed(const Ref<VertexArray>& vertexArray, uint32_t elementCount) = 0;

		/**
		 *	sets the viewport of the graphics context
		 *	@param x: the x-position of the lower left corner of the viewport
		 *	@param y: the y-position of the lower left corner of the viewport
		 *	@param width: width of the viewport in pixels
		 *	@param height: height of the viewport in pixels
		 */
		virtual void setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) = 0;

		/**
		 *	enables depth test
		 */
		virtual void enableDepthTest() = 0;
		/**
		 *	disables depth test
		 */
		virtual void disableDepthTest() = 0;
		/**
		 *	enables alpha blending
		 */
		virtual void enableAlphaBlending() = 0;
		/**
		 *	disables alpha blending
		 */
		virtual void disableAlphaBlending() = 0;
		/**
		 *	Sets the line width (in pixels) of Renderer::submitLines calls
		 */
		virtual void setLineWidth(float width) = 0;
		/**
		 *	Sets the Point size (in pixels) of Renderer::submitPoints calls
		 */
		virtual void setPointSize(float radius) = 0;

		/**
		 *	TODO: Recreates the Window using the new MSAA settings
		 */
		//virtual void setMSAA(MSAA setting) = 0;

	private:
		// platform specific implementation of the api
		static API s_API;
	};

}
