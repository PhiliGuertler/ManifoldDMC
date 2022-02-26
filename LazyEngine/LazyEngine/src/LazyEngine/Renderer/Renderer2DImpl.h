#pragma once

// ######################################################################### //
// ### Renderer2DImpl.h #################################################### //
// ### Implements the Renderer2D Implementation that will be wrapped     ### //
// ### statically in Renderer2D.h                                        ### //
// ######################################################################### //

#include <vector>
#include <array>

#include "Camera.h"
#include "Geometry/GeometricPrimitives.h"
#include "Shader.h"
#include "Texture.h"
#include "TextureAtlas.h"
#include "VertexArray.h"

#include "Renderer2D.h"

namespace LazyEngine {

	/**
	 *	A singleton class representing a 2D-Renderer
	 */
	class Renderer2DImpl {
	public:
		/**
		 *	returns the singleton instance of this class
		 */
		static Renderer2DImpl& getInstance();

		// this can be modified to allow for more elements per draw call, the final buffer will contain
		// 2^max(AT_ME, 4) * 4 vertices.
		static constexpr int AT_ME = 11;
		static constexpr int DEFAULT_EXPONENT = std::max(AT_ME, 4);
		// to simplify housekeeping and caching, this renderer only allows 
		// for exactly a power of 2 many elements per batch
		static constexpr int DEFAULT_MAX_BATCH_ELEMENTS = 1 << DEFAULT_EXPONENT;

	public:
		/**
		 *	default destructor
		 */
		~Renderer2DImpl() = default;

		/**
		 *	begins a scene using a single camera.
		 *	Everything that will be submitted will be batch-rendered using this camera.
		 *	This means that not every submit will result in a draw call, instead submissions
		 *	will be cached and rendered together in batches.
		 *	@param camera: the camera to be used in the scene
		 */
		void beginScene(const Camera& camera);

		/**
		 *	Ends the scene. After calling this, every quad submitted will have been drawn.
		 */
		void endScene();

		/**
		 *	Renders the batch. This will be called if the number of pending draw requests exceeds
		 *	the maximum number of quads or if endScene() is called.
		 */
		void flush();


		/**
		 *	Draws a Quad. FIXME: for some reason, the order of quads is backwards
		 *	@param quad: The quad to be drawn
		 */
		void drawQuad(const Quad& quad);

		/**
		 *	Draws a screen-sized quad with a given texture and color tint
		 *	@param texture: The texture that should be displayed
		 *	@param colorTint [optional]: The color tint that should be applied to the texture.
		 *		By default, this is white, which corresponds to no change in color.
		 */
		void drawScreenSizedQuad(const Ref<Texture2D>& texture, const glm::vec4& colorTint = glm::vec4(1.f));

#if LAZYENGINE_ENABLE_RENDERER_STATISTICS
		void resetStatistics();

		Renderer2D::Statistics getStatistics() const;
#endif
	private:
		/**
		 *	private constructor to ensure the singleton nature of this class
		 */
		Renderer2DImpl();

		/**
		 *	delete the copy and move constructors to prevent the user from accidentally copying the singleton
		 */
		Renderer2DImpl(const Renderer2DImpl& copy) = delete;
		Renderer2DImpl(const Renderer2DImpl&& move) = delete;

		/**
		 *	delete the copy and move operator=s to prevent the user from accidentally copying the singleton
		 */
		Renderer2DImpl& operator=(const Renderer2DImpl& copy) = delete;
		Renderer2DImpl& operator=(Renderer2DImpl&& move) = delete;

		/**
		 *	Fills the vector with entries in the scheme {0,1,2, 2,3,0}, to create the index data for quads
		 */
		void fillWithQuadIndexData(std::vector<uint32_t>& output);

		/**
		 *	Performs a lazy flush, which means that the flush only occurs if the current batch would overflow
		 *	by adding a new quad.
		 */
		inline void lazyFlush() {
			if (m_quadCount >= m_maxQuadsPerBatch) {
				flush();
			}
		}

		/**
		 *	creates an array of four quad-vertices from a given transform, color and textureindex.
		 *	@param transform: the model matrix of the quad, that may consist of a combined translation, scale and rotation
		 *	@param color: the color of the quad. if a texture is used, the color will be used as texture tint
		 *	@param textureIndex: the texture index in this current batch. this should be the result of m_textureCache->addTexture() or 0 (no texture).
		 *	@param textureCoordinates: The texture coordinates of the individual vertices
		 */
		std::array<QuadVertex, 4> createQuad(const glm::mat4& transform, const glm::vec4& color, uint32_t textureIndex, const std::array<glm::vec2, 4>& textureCoordinates) const;

	private:
		/**
		 *	A little cache to prevent too many calls of GPU functions.
		 *	It will flush automatically by uploading to the GPU.
		 */
		class QuadCache {
		public:
			/**
			 *	constructor
			 *	@param vertexArray: The VertexArray into which the cached data should be flushed
			 *	@param quadCount: a reference to an external variable that keeps track of
			 *		the number of quads currently stored in the vertexArray. Upon flushing, this variable will be
			 *		updated.
			 */
			QuadCache(Ref<VertexArray> vertexArray, uint32_t& quadCount);

			/**
			 *	default destructor
			 */
			~QuadCache() = default;

			/**
			 *	Adds a quad to the cache. If the cache would overflow, it will first be flushed.
			 *	@param quadVertices: the four vertices that make up the quad that should be added.
			 */
			void addQuad(const std::array<QuadVertex, 4>& quadVertices);

			/**
			 *	Flushes the cache into the VertexArray that has been passed in the constructor.
			 */
			void flush();

			inline size_t size() const { return m_vertexCache.size(); }

		private:
			// The exponent of the cache's size. The caches final size will be 2^CACHE_SIZE
			static constexpr int CACHE_SIZE = Renderer2DImpl::DEFAULT_EXPONENT;

			// The vertex array into which this cache flushes its data
			Ref<VertexArray> m_vertexArray;
			// A reference to Renderer2D's quadCount that will be updated with every flush.
			uint32_t& m_quadCount;

			// The actual cache data structure
			std::array<QuadVertex, (1 << CACHE_SIZE)> m_vertexCache;
			// The index of the vertex that should be inserted next
			std::size_t m_cacheIndex;
		};

		/**
		 *	A simple stack that keeps track of which textures have been
		 *	registered for the current batch and to which slot they will
		 *	be bound.
		 */
		class TextureCache {
		public:
			/**
			 *	constructor
			 *	@param batchShader: The shader that will be used by the batch-renderer.
			 *		This shader needs to contain a uniform array of sampler2D called 'u_textures'.
			 */
			TextureCache(Ref<Shader> batchShader);
			/**
			 *	default destructor
			 */
			~TextureCache() = default;

			/**
			 *	returns the slot of the texture and forces a flush if the cache would overflow
			 */
			uint32_t addTexture(Ref<Texture2D> texture);

			/**
			 *	flushes the cache by resetting the textureSlots
			 */
			void flush();

			/**
			 *	binds all textures that are currently stored to their respective texture slots
			 */
			void bindTextures() const;

			/**
			 *	returns how many different textures are active.
			 *	The white texture in slot 0 is not part of this.
			 */
			inline uint32_t getActiveTextures() const { return m_textureSlotIndex - 1; }

			/**
			 *	Returns the maximum amount of textures that this cache can store.
			 */
			inline size_t size() const { return m_textureSlots.size(); }
		private:

			// TODO: create a class RendererCapabilities, that queries nearly everything the renderer on this system
			//	is capable of on startup and get the max amount of TextureSlots from there
			static const uint32_t MaxTextureSlots = 32;
			// the stack of textures. Slot 0 will always be a white 1x1 texture!
			std::array<Ref<Texture2D>, MaxTextureSlots> m_textureSlots;
			// Kind of the stack-pointer of this texture stack
			uint32_t m_textureSlotIndex = 1; // slot 0 will always be a white 1x1 texture!
		};

	private:
		// internal data
		Ref<VertexArray> m_quadVertexArray;
		// shader library containing all nescessary shaders
		// TODO: the client should enter shaders here
		Scope<ShaderLibrary> m_shaders;

		// Batching Data Members
		// These values can be modified using setMaxBatchElements()
		const uint32_t m_maxQuadsPerBatch = DEFAULT_MAX_BATCH_ELEMENTS;
		const uint32_t m_maxVerticesPerBatch = DEFAULT_MAX_BATCH_ELEMENTS * 4;
		const uint32_t m_maxIndicesPerBatch = DEFAULT_MAX_BATCH_ELEMENTS * 6;

		// the amount of quads that are stored in m_quadVertexArray
		uint32_t m_quadCount = 0;

		// This cache will reduce the amount of data uploads to m_quadVertexArray, which results in a big speedup
		Scope<QuadCache> m_quadCache;

		// store the submitted textures for the current batch
		Scope<TextureCache> m_textureCache;

		glm::mat4 m_currentCamProjView = glm::mat4(1.f);

#if LAZYENGINE_ENABLE_RENDERER_STATISTICS
		// a debugging variable that keeps track of beginScene() and endScene() pairs.
		bool m_sceneIsRunning = false;
		// a struct containing the statistics of the current frame
		Renderer2D::Statistics m_stats;
#endif
	};

}