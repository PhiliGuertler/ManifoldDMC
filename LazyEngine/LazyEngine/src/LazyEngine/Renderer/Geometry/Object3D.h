#pragma once

// ######################################################################### //
// ### Object3D.h ########################################################## //
// ### Defines a class for 3D objects.                                   ### //
// ######################################################################### //

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace LazyEngine {

	/**
	 *	This class represents an object in 3D space with a given
	 *	position, rotation and scaling.
	 */
	class Object3D {
	public:
		/**
		 *	constructor
		 */
		Object3D();
		/**
		 *	destructor
		 */
		virtual ~Object3D();

		/**
		 *	returns a model-to-world matrix that transforms from model space to world space.
		 */
		virtual const glm::mat4 getModelToWorld() const;
		/**
		 *	returns a world-to-model matrix that transforms from world space to model space.
		 */
		virtual const glm::mat4 getWorldToModel() const;
		/**
		 *	returns the orientation of this object
		 */
		virtual const glm::quat getOrientation() const;
		/**
		 *	returns the position of this object
		 */
		virtual const glm::vec3 getPosition() const;
		/**
		 *	returns the scale of this object in x,y,z directions
		 */
		virtual const glm::vec3 getScale() const;
		/**
		 *	returns the current normalized forward direction in world space
		 */
		virtual const glm::vec3 getForwardDirection() const;

		/**
		 *	sets the position of this object
		 *	@param position: the new position
		 */
		virtual void setPosition(const glm::vec3& position);
		/**
		 *	sets the orientation of this object
		 *	@param orientation: the new orientation
		 */
		virtual void setOrientation(const glm::quat& orientation);
		/**
		 *	sets the orientation of this object
		 *	@param eulerAngles: the angles in radians in the order:
		 *		pitch, yaw, roll
		 */
		virtual void setOrientation(const glm::vec3& eulerAngles);
		/**
		 *	sets the scale of this object in x,y,z directions
		 *	@param scale: the new scale in x,y,z directions
		 */
		virtual void setScale(const glm::vec3& scale);

		/**
		 *	rotates the object relative around an axis in world space
		 *	@param axis: the normalized axis around which the object should be rotated (in world space)
		 *	@param angle: the angle in radians
		 */
		virtual void rotateBy(const glm::vec3& axis, float angle);
		/**
		 *	rotates the object relative around an axis in model space
		 *	@param localAxis: the normalized axis around which the object should be rotated (in model space)
		 *	@param angle: the angle in radians
		 */
		virtual void rotateLocallyBy(const glm::vec3& localAxis, float angle);
		/**
		 *	translates the object relative to its current position
		 *	@param translation: the change in position that should be applied to the object
		 */
		virtual void translateBy(const glm::vec3& translation);
		/**
		 *	scales the object relative to its current scaling
		 *	@param factors: the factors in x,y,z that should be applied to this object's scale
		 */
		virtual void scaleBy(const glm::vec3& factors);
		/**
		 *	Orients the object to have its forward direction (-z) point to the target point
		 *  @param target: the target at which to point
		 *  @param upVector: a vector that defines the up direction
		 */
		virtual void lookAt(const glm::vec3& target, const glm::vec3& upVector = glm::vec3(0.f, 1.f, 0.f));
	protected:
		/**
		 *	updates the cached model to world matrix
		 */
		virtual void updateModelToWorld();

	protected:
		glm::quat m_orientation;
		glm::vec3 m_position;
		glm::vec3 m_scale;

		glm::mat4 m_cachedModelToWorld;
	};

}