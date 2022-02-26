#pragma once

// ######################################################################### //
// ### GeometricPrimitives.h ############################################### //
// ### Defines geometric primitives like quads and vertices              ### //
// ######################################################################### //

#include "glm/glm.hpp"

#include "LazyEngine/Core/Core.h"
#include "../Texture.h"


namespace LazyEngine {

	/**
	 *	A simple data structure that represents a single vertex
	 */
	struct QuadVertex {
		glm::vec3 position = glm::vec3(0.f);
		glm::vec4 color = glm::vec4(1.f, 0.f, 1.f, 1.f);
		glm::vec2 textureCoordinates = glm::vec2(0.f);
		// the texture ID of this vertex' texture (which should be casted to an int in the shader)
		float textureIndex = 0;
		// TODO: extend me (maskID, etc)
	};

	/**
	 *	represents a Quad with a position, size, color and rotation
	 */
	class Quad {
	public:
		// defines the vertices of a 1x1 quad, that is centered around the origin in the order bottom-left, bottom-right, top-right, top-left
		static const glm::vec3 s_quadPositions[4];
		// defines the texture coordinates of a quad's vertices in the order bottom-left, bottom-right, top-right, top-left
		static const std::array<glm::vec2, 4> s_textureCoordinates;

	public:
		/**
		 *	constructor
		 *	@param position: The position of the quad's center
		 *	@param size: The size of the quad in x and y direction
		 *	@param color: The color of the quad. If also a texture is added, this color will be the tint
		 *	@param angleRadians: The rotation in radians (counter clock-wise)
		 *	@param texture: The texture of this quad. nullptr corresponds to no texture.
		 */
		Quad(glm::vec3 position = glm::vec3(0.f, 0.f, 0.f), 
			glm::vec2 size = glm::vec2(1.f, 1.f), 
			glm::vec4 color = glm::vec4(1.f, 1.f, 1.f, 1.f), 
			float angleRadians = 0.f, 
			Ref<Texture2D> texture = nullptr);

		Quad(glm::vec3 position,
			glm::vec2 size,
			glm::vec4 color,
			float angleRadians,
			Ref<SubTexture2D> subtexture);

		~Quad() = default;

		/**
		 *	sets the position
		 */
		inline void setPosition(const glm::vec2& position) { m_position = glm::vec3(position,0.f); }
		/**
		 *	sets the size
		 */
		inline void setSize(const glm::vec2& size) { m_size = size; }
		/**
		 *	sets the color
		 */
		inline void setColor(const glm::vec4& color) { m_color = color; }
		/**
		 *	sets the angle in radians
		 */
		inline void setAngleRadians(float angle) { m_angleRadians = angle; }
		/**
		 *	sets the texture
		 */
		inline void setTexture(Ref<Texture2D> texture) { m_texture = texture; }
		/**
		 *	sets the texture
		 */
		inline void setTextureCoordinates(const std::array<glm::vec2, 4>& textureCoordinates) {	m_textureCoordinates = textureCoordinates; }
		
		/**
		 *	returns the position of the quad center
		 */
		inline const glm::vec2 getPosition() const { return glm::vec2(m_position); }
		/**
		 *	returns the size of the quad
		 */
		inline const glm::vec2& getSize() const { return m_size; }
		/**
		 *	returns the color of the quad
		 */
		inline const glm::vec4& getColor() const { return m_color; }
		/**
		 *	returns the angle of rotation of the quad in radians
		 */
		inline float getAngleRadians() const { return m_angleRadians; }
		/**
		 *	returns the texture of this quad
		 */
		inline Ref<Texture2D> getTexture() const { return m_texture; }
		/**
		 *	returns the texturecoordinates of this quad
		 */
		inline const std::array<glm::vec2, 4>& getTextureCoordinates() const { return m_textureCoordinates; }


		/**
		 *	computes the model matrix depending on position, size and angle
		 */
		const glm::mat4 computeModelMatrix() const;

	private:
		// the position of the quad
		glm::vec3 m_position;
		// the size of the quad
		glm::vec2 m_size;
		// the color of the quad
		glm::vec4 m_color;
		// the angle of the quad's rotation around its center (counter-clockwise) in radians
		float m_angleRadians;
		// the texture of this quad (nullptr if this quad has none)
		Ref<Texture2D> m_texture;
		// the texture coordinates for each vertex of the quad
		std::array<glm::vec2, 4> m_textureCoordinates;

	};

}