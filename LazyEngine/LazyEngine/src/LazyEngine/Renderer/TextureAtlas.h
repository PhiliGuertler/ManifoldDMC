#pragma once

// ######################################################################### //
// ### TextureAtlas.h ###################################################### //
// ### Defines the class TextureAtlas, that allows simple creation of    ### //
// ### Sub-Textures using a bigger texture.                              ### //
// ######################################################################### //

#include "Texture.h"

namespace LazyEngine {

	/**
	 *	A simple Texture Atlas, that can be uses as a 2D table of
	 *	multiple sub-textures inside a bigger texture.
	 */
	class TextureAtlas {
	public:
		/**
		 *	Constructor for Textures that consist of sub-textures that are all the same size
		 *	@param texture: the texture that contains multiple smaller sprites. Undefined for nullptr.
		 *	@param subTextureResolution: the resolution of the smallest sprite in pixels
		 */
		TextureAtlas(Ref<Texture2D> texture, const glm::ivec2& subTextureResolution);

		/**
		 *	default destructor
		 */
		virtual ~TextureAtlas() = default;

		/**
		 *	returns a subtexture for the sprite at location [xIndex, yIndex].
		 *	@param index2D: The 2 dimensional index of the desired sprite.
		 *	@param spriteSize: The size of a sprite.
		 *		Examples: Assuming the subTextureResolution on construction was {16,16}
		 *		{1,1} means a sprite has the resolution of the subtexture resolution: {16,16}
		 *		{2,1} means the sprite is twice as wide as it is high: {32,16}
		 */
		const Ref<SubTexture2D> createSubTexture(const glm::ivec2& index2D, const glm::vec2& spriteSize = {1.f,1.f}) const;

		/**
		 *	returns how many sprites there are in each dimension
		 */
		const glm::ivec2 getNumberOfSprites() const;

		/**
		 *	returns the texture of this atlas.
		 */
		inline const Ref<Texture2D> getTexture() const { return m_texture; }
	private:
		// a reference to the texture itself
		Ref<Texture2D> m_texture;
		// resolution of one sprite in the atlas (in whole pixels)
		glm::ivec2 m_subTextureResolution;

		// stores the factor of subTexture to full texture resolution (0 < factor <= 1)
		glm::vec2 m_spriteFactor;
	};

}