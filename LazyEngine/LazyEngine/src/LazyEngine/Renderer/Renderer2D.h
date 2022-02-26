#pragma once

// ######################################################################### //
// ### Renderer2D.h ######################################################## //
// ### A Renderer Singleton that batch-renders Quads with and without    ### //
// ### textures, rotations, scales and translations.                     ### //
// ######################################################################### //

#include "Camera.h"
#include "Geometry/GeometricPrimitives.h"

#define LAZYENGINE_ENABLE_RENDERER_STATISTICS 1

namespace LazyEngine {

	/**
	 *	A static wrapper for the 2D renderer singleton
	 */
	class Renderer2D {
	public:

		/**
		 *	begins a scene using a single camera.
		 *	Everything that will be submitted will be batch-rendered using this camera.
		 *	This means that not every submit will result in a draw call, instead submissions
		 *	will be cached and rendered together in batches.
		 *	@param camera: the camera to be used in the scene
		 */
		static void beginScene(const Camera& camera);
		/**
		 *	Ends the scene. After calling this, every quad submitted will have been drawn.
		 */
		static void endScene();
		/**
		 *	Renders the batch. This will be called if the number of pending draw requests exceeds
		 *	the maximum number of quads or if endScene() is called.
		 */
		static void flush();

		// ####################################################################### //
		// ### Primitive Draw Calls ############################################## //
		// ####################################################################### //

		/**
		 *	Draws a Quad
		 *	@param quad: The quad to be drawn
		 */
		static void drawQuad(const Quad& quad);

		/**
		 *	Draws a screen-sized quad with a given texture and color tint
		 *	@param texture: The texture that should be displayed
		 *	@param colorTint [optional]: The color tint that should be applied to the texture.
		 *		By default, this is white, which corresponds to no change in color.
		 */
		static void drawScreenSizedQuad(const Ref<Texture2D>& texture, const glm::vec4& colorTint = glm::vec4(1.f));

#if LAZYENGINE_ENABLE_RENDERER_STATISTICS
		// statistics to measure the performance of this renderer
		struct Statistics {
			uint32_t numDrawCalls = 0;
			uint32_t numQuads = 0;
			uint32_t numTextures = 0;

			uint32_t maxQuadsPerBatch = 0;
			uint32_t maxTexturesPerBatch = 0;

			uint32_t getTotalVertexCount() { return numQuads * 4; }
			uint32_t getTotalIndexCount() { return numQuads * 6; }
		};

		static void resetStatistics();

		static Statistics getStatistics();
#endif
	};
}