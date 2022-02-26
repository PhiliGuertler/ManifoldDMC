#include "LazyEngine/gepch.h"

// ######################################################################### //
// ### TextureAtlas.cpp #################################################### //
// ### Implements TextureAtlas.h                                         ### //
// ######################################################################### //

#include "TextureAtlas.h"

namespace LazyEngine {	

	// ######################################################################## //
	// ### TextureAtlas ####################################################### //
	// ######################################################################## //

	TextureAtlas::TextureAtlas(Ref<Texture2D> texture, const glm::ivec2& subTextureResolution)
		: m_texture(texture)
		, m_subTextureResolution(subTextureResolution)
		, m_spriteFactor(glm::vec2(1.f))
	{
		if (texture == nullptr) return;

		// compute the sprite factor
		glm::ivec2 textureResolution = texture->getSize();
		m_spriteFactor = glm::vec2(subTextureResolution) / glm::vec2(textureResolution);
	}

	const Ref<SubTexture2D> TextureAtlas::createSubTexture(const glm::ivec2& index2D, const glm::vec2& spriteSize) const {
		std::array<glm::vec2, 4> result;
		result[0] = (glm::vec2(index2D) + glm::vec2(0.f, 0.f) * spriteSize) * m_spriteFactor;	// bottom-left
		result[1] = (glm::vec2(index2D) + glm::vec2(1.f, 0.f) * spriteSize) * m_spriteFactor;	// bottom-right
		result[2] = (glm::vec2(index2D) + glm::vec2(1.f, 1.f) * spriteSize) * m_spriteFactor;	// top-right
		result[3] = (glm::vec2(index2D) + glm::vec2(0.f, 1.f) * spriteSize) * m_spriteFactor;	// top-left

		return createRef<SubTexture2D>(m_texture, result);
	}

	const glm::ivec2 TextureAtlas::getNumberOfSprites() const {
		return m_texture->getSize()/m_subTextureResolution;
	}

}