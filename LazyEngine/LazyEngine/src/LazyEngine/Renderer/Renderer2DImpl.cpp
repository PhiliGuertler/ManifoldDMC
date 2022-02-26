// ######################################################################### //
// ### Renderer2DImpl.cpp ################################################## //
// ### Implements the Renderer2D Implementation that will be wrapped     ### //
// ### statically in Renderer2D.h                                        ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"

#include "Renderer2DImpl.h"

#include <glm/gtc/matrix_transform.hpp>

#include "RenderCommand.h"
#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	// ######################################################################## //
	// ### Helper Functions ################################################### //
	// ######################################################################## //

	static inline glm::vec3 transformVec3(const glm::vec3& vec, const glm::mat4& transform) {
		glm::vec4 result = transform * glm::vec4(vec, 1.f);
		return glm::vec3(result) / result.w;
	}
	

	// ######################################################################## //
	// ### Renderer2D::QuadCache ############################################## //
	// ######################################################################## //

	Renderer2DImpl::QuadCache::QuadCache(Ref<VertexArray> vertexArray, uint32_t& quadCount)
		: m_vertexArray(vertexArray)
		, m_quadCount(quadCount)
		, m_vertexCache()
		, m_cacheIndex(0)
	{
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();
		// empty
	}

	void Renderer2DImpl::QuadCache::addQuad(const std::array<QuadVertex, 4>& quadVertices) {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();

		// add every vertex to the cache and update the cache index
		// This is not in need for a bounds check, because m_vertexCache.size() % 4 == 0
		m_vertexCache[m_cacheIndex++] = quadVertices[0];
		m_vertexCache[m_cacheIndex++] = quadVertices[1];
		m_vertexCache[m_cacheIndex++] = quadVertices[2];
		m_vertexCache[m_cacheIndex++] = quadVertices[3];

		// check if the cache would overflow after adding another quad and if so, flush it
		if (m_cacheIndex >= m_vertexCache.size()) {
			flush();
		}
	}

	void Renderer2DImpl::QuadCache::flush() {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();
		// early out if the cache is empty
		if (m_cacheIndex == 0) return;

		m_vertexArray->bind();
		// the size of the data in bytes is the amount of cached vertices times their size
		const uint32_t dataSize = static_cast<uint32_t>(m_cacheIndex * sizeof(QuadVertex));
		// the offset in the buffer is the amount of existing quads times four vertices times the size of a single vertex
		const uint32_t offset = static_cast<uint32_t>(m_quadCount * 4 * sizeof(QuadVertex));
		// upload the cached vertex data to the gpu
		m_vertexArray->getVertexBuffers()[0]->uploadData((float*)m_vertexCache.data(), dataSize, offset);
		// add the size of the quadcache's amount of elements (the vertex count) divided by 4 to the quadCount
		const uint32_t newQuads = static_cast<uint32_t>(m_cacheIndex / 4);
		// update the quad count
		m_quadCount += newQuads;
		// reset the cacheIndex
		m_cacheIndex = 0;
	}


	// ######################################################################## //
	// ### TextureCache ####################################################### //
	// ######################################################################## //

	Renderer2DImpl::TextureCache::TextureCache(Ref<Shader> batchShader)
		: m_textureSlots()
		, m_textureSlotIndex(1)
	{
		// Initialize the texture slot stack
		for (int i = 0; i < m_textureSlots.size(); ++i) {
			m_textureSlots[i] = nullptr;
		}

		// initialize a 1x1 pixel white texture to be used when drawing vertex colors only
		uint32_t whiteTextureData = 0xffffffff;
		auto whiteTexture = Texture2D::create(1, 1, &whiteTextureData, sizeof(uint32_t));

		// set the first texture to this white texture
		m_textureSlots[0] = whiteTexture;

		// TODO: don't hardcode the amount of textures!
		std::vector<int> u_textures(MaxTextureSlots);
		for (int i = 0; i < u_textures.size(); ++i) {
			u_textures[i] = i;
		}

		// upload the texture slot indices to the shader.
		// this is effectively an array from 0 to size()-1
		batchShader->bind();
		batchShader->uniformIntArray("u_textures", u_textures.data(), (uint32_t)(u_textures.size()));
	}

	uint32_t Renderer2DImpl::TextureCache::addTexture(Ref<Texture2D> texture) {
		if (texture == nullptr) {
			// this object has no texture, so use the white texture in slot 0.
			return 0;
		}

		// Slot 0 is the white texture, it will not be removed!
		// check if this texture is already part of the stack
		for (uint32_t i = 1; i < m_textureSlotIndex; ++i) {
			if (texture == m_textureSlots[i]) {
				return i;
			}
		}

		// this texture is not yet part of the cache.
		// check if it would overflow it.
		if (m_textureSlotIndex == m_textureSlots.size() - 1) {
			// force the renderer to draw the batch.
			// this will also flush this cache.
			Renderer2DImpl::getInstance().flush();
		}

		// add the texture to the cache
		m_textureSlots[m_textureSlotIndex] = texture;
		// update the texture slot index
		m_textureSlotIndex++;
		// return the texture slot index that was used before incrementing it.
		return m_textureSlotIndex - 1;
	}

	void Renderer2DImpl::TextureCache::flush() {
		// Slot 0 is the white texture, it will not be removed!
		// reset the textureSlots to nullptrs ...
		for (uint32_t i = 1; i < m_textureSlotIndex; ++i) {
			m_textureSlots[i] = nullptr;
		}

		// ... and the texture slot index
		m_textureSlotIndex = 1;
	}

	void Renderer2DImpl::TextureCache::bindTextures() const {
		// bind each texture to their corresponding index
		for (uint32_t i = 0; i < m_textureSlotIndex; ++i) {
			m_textureSlots[i]->bind(i);
		}
	}


	// ######################################################################## //
	// ### Renderer2D ######################################################### //
	// ######################################################################## //

	Renderer2DImpl::Renderer2DImpl() {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();

		m_shaders = createScope<ShaderLibrary>();

		m_quadVertexArray = VertexArray::create();

		// create a quad vertex buffer using these vertices and add it to the vertexarray
		Ref<VertexBuffer> quadVertexBuffer = VertexBuffer::create(m_maxVerticesPerBatch * sizeof(QuadVertex), BufferUsage::DynamicDraw);
		quadVertexBuffer->setLayout({
			{ShaderDataType::Float3, "a_position"},
			{ShaderDataType::Float4, "a_color"},
			{ShaderDataType::Float2, "a_textureCoordinates"},
			{ShaderDataType::Float, "a_textureIndex"}
			});
		m_quadVertexArray->addVertexBuffer(quadVertexBuffer);

		// create an index buffer to display m_maxQuadsPerBatch many quads
		std::vector<uint32_t> indices(m_maxIndicesPerBatch);
		fillWithQuadIndexData(indices);
		Ref<IndexBuffer> quadIndexBuffer = IndexBuffer::create(indices.data(), (uint32_t)indices.size());
		m_quadVertexArray->setIndexBuffer(quadIndexBuffer);

		// TODO: this shader should somehow be part of LazyEngine itself, not part of the client
		auto shader = m_shaders->load("MainShader", "assets/shaders/2DShader.glsl");

		// Initialize the texture slot storage
		m_textureCache = createScope<TextureCache>(shader);

		// initialize the quadcache
		m_quadCache = createScope<QuadCache>(m_quadVertexArray, m_quadCount);

		// Initialization complete
		LAZYENGINE_CORE_INFO("Initialized Renderer2D with {0} Vertex-Slots and {1} Cache-Slots", m_maxVerticesPerBatch, m_quadCache->size());
	}

	void Renderer2DImpl::fillWithQuadIndexData(std::vector<uint32_t>& output) {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();

		uint32_t vertexID = 0;
		for (size_t i = 0; i < output.size(); i += 6) {
			// first triangle
			output[i] = 0 + vertexID;
			output[i + 1] = 1 + vertexID;
			output[i + 2] = 2 + vertexID;

			// second triangle
			output[i + 3] = 2 + vertexID;
			output[i + 4] = 3 + vertexID;
			output[i + 5] = 0 + vertexID;

			vertexID += 4;
		}
	}

	Renderer2DImpl& Renderer2DImpl::getInstance() {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();

		static Renderer2DImpl instance;
		return instance;
	}

	void Renderer2DImpl::beginScene(const Camera& camera) {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();

#if LAZYENGINE_ENABLE_RENDERER_STATISTICS
		if (m_sceneIsRunning) {
			LAZYENGINE_CORE_WARN("The previous scene has not ended. Did you forget to call \"Renderer2D::endScene()\"?");
		}
		m_sceneIsRunning = true;
		resetStatistics();
#endif

		// get the main shader
		// FIXME: this shader is currently in a directory of the client app.
		//	It should be part of this Engine.
		auto mainShader = m_shaders->get("MainShader");
		mainShader->bind();

		// set camera matrix
		m_currentCamProjView = camera.getProjectionViewMatrix();
		mainShader->uniformMat4("u_projView", m_currentCamProjView);

		// disable the depth test as only 2D quads will be drawn.
		RenderCommand::disableDepthTest();
	}

	void Renderer2DImpl::endScene() {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();

#if LAZYENGINE_ENABLE_RENDERER_STATISTICS
		m_sceneIsRunning = false;
#endif

		flush();

		// re-enable the depth test
		RenderCommand::enableDepthTest();
	}

	void Renderer2DImpl::flush() {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();

		// flush the quad cache to ensure that all quads submitted so far
		// will be drawn
		m_quadCache->flush();

		// early out if there are no quads to be drawn
		if (m_quadCount == 0) return;

		// bind all textures
		m_textureCache->bindTextures();

		// draw the vertex array
		m_quadVertexArray->bind();
		RenderCommand::drawQuadsIndexed(m_quadVertexArray, m_quadCount);

		// reset the quad count to zero
		m_quadCount = 0;

		// flush the texture cache
		m_textureCache->flush();

#if LAZYENGINE_ENABLE_RENDERER_STATISTICS
		m_stats.numDrawCalls++;
#endif
	}


	std::array<QuadVertex, 4> Renderer2DImpl::createQuad(const glm::mat4& transform, const glm::vec4& color, uint32_t textureIndex, const std::array<glm::vec2, 4>& textureCoordinates) const {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();

		// create quad vertices
		std::array<QuadVertex, 4> quadVertices;

		for (int i = 0; i < 4; ++i) {
			quadVertices[i].position = transformVec3(Quad::s_quadPositions[i], transform);
			quadVertices[i].color = color;
			quadVertices[i].textureCoordinates = textureCoordinates[i];
			quadVertices[i].textureIndex = (float)textureIndex;
		}

		return quadVertices;
	}

	void Renderer2DImpl::drawQuad(const Quad& quad) {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();
		lazyFlush();

		uint32_t activeTexturesBefore = m_textureCache->getActiveTextures();
		uint32_t slot = m_textureCache->addTexture(quad.getTexture());
		uint32_t activeTexturesAfter = m_textureCache->getActiveTextures();

		std::array<QuadVertex, 4> quadVertices = createQuad(quad.computeModelMatrix(), quad.getColor(), slot, quad.getTextureCoordinates());
		m_quadCache->addQuad(quadVertices);
		
#if LAZYENGINE_ENABLE_RENDERER_STATISTICS
		m_stats.numQuads++;
		if (activeTexturesAfter != activeTexturesBefore) {
			m_stats.numTextures++;
		}
#endif
	}
	
	void Renderer2DImpl::drawScreenSizedQuad(const Ref<Texture2D>& texture, const glm::vec4& colorTint) {
		// make sure that every quad that is waiting to be rendered
		// will be rendered immediately, as the camera will have to change
		flush();

		// bind the main shader
		auto mainShader = m_shaders->get("MainShader");
		mainShader->bind();

		// set up an orthographic camera that displays a quad with a width and height of 1
		// as big as the entire viewport
		static const glm::mat4 screenOrtho = glm::ortho(-0.5f, 0.5f, -0.5f, 0.5f);
		mainShader->uniformMat4("u_projView", screenOrtho);

		// set up a quad that fills the entire viewport and set its texture and color
		Quad quad;
		quad.setTexture(texture);
		quad.setColor(colorTint);

		// draw the quad 
		drawQuad(quad);
		// make sure to render this quad with the correct camera
		flush();

		// restore the camera matrix of this scene
		mainShader->uniformMat4("u_projView", m_currentCamProjView);
	}


	// ####################################################################### //
	// ### Statistics of the Renderer ######################################## //

#if LAZYENGINE_ENABLE_RENDERER_STATISTICS
	void Renderer2DImpl::resetStatistics() {
		m_stats.numDrawCalls = 0;
		m_stats.numQuads = 0;
		m_stats.numTextures = 0;
		
		m_stats.maxQuadsPerBatch = m_maxQuadsPerBatch;
		m_stats.maxTexturesPerBatch = (uint32_t)m_textureCache->size();
	}

	Renderer2D::Statistics Renderer2DImpl::getStatistics() const {
		return m_stats;
	}
#endif
}