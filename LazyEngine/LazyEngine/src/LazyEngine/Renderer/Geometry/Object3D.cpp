#include "LazyEngine/gepch.h"

// ######################################################################### //
// ### Object3D.cpp ######################################################## //
// ### Implements Object3D.h                                             ### //
// ######################################################################### //

#include "Object3D.h"

namespace LazyEngine {

	Object3D::Object3D()
		: m_orientation(glm::vec3(0.f))	// this represents no rotation at all
		, m_position(0.f)
		, m_scale(1.f)
	{
		updateModelToWorld();
	}

	Object3D::~Object3D() {
		// empty
	}

	const glm::mat4 Object3D::getModelToWorld() const {
		return m_cachedModelToWorld;
	}
	
	const glm::mat4 Object3D::getWorldToModel() const {
		return glm::inverse(m_cachedModelToWorld);
	}

	const glm::quat Object3D::getOrientation() const {
		return m_orientation;
	}

	const glm::vec3 Object3D::getPosition() const {
		return m_position;
	}

	const glm::vec3 Object3D::getScale() const {
		return m_scale;
	}
	
	const glm::vec3 Object3D::getForwardDirection() const {
		return glm::normalize(getOrientation() * glm::vec3(0.f, 0.f, -1.f));
	}
	
	void Object3D::setOrientation(const glm::vec3& eulerAngles) {
		m_orientation = glm::quat(eulerAngles);

		updateModelToWorld();
	}

	void Object3D::setScale(const glm::vec3& scale) { 
		m_scale = scale;

		updateModelToWorld();
	}

	void Object3D::setPosition(const glm::vec3& position) {
		m_position = position;

		updateModelToWorld();
	}

	void Object3D::setOrientation(const glm::quat& orientation) {
		m_orientation = orientation;

		updateModelToWorld();
	}

	void Object3D::rotateBy(const glm::vec3& axis, float angle) {
		// create a quaternion from the input rotation
		glm::quat rotation = glm::angleAxis(angle, axis);
		m_orientation = glm::normalize(rotation * m_orientation);

		updateModelToWorld();
	}

	void Object3D::rotateLocallyBy(const glm::vec3& localAxis, float angle) {
		// create a quaternion from the input rotation
		glm::quat rotation = glm::angleAxis(angle, localAxis);
		m_orientation = glm::normalize(m_orientation * rotation);

		updateModelToWorld();
	}


	void Object3D::translateBy(const glm::vec3& translation) {
		m_position += translation;

		updateModelToWorld();
	}

	void Object3D::scaleBy(const glm::vec3& factors) {
		m_scale *= factors;

		updateModelToWorld();
	}

	void Object3D::lookAt(const glm::vec3& target, const glm::vec3& upVector) {
		m_orientation = glm::lookAt(m_position, target, upVector);

		updateModelToWorld();
	}


	void Object3D::updateModelToWorld() {
		// create the identity matrix
		glm::mat4 result = glm::mat4(1.f);
		// the matrices must be created in the inverse order, so the first 
		// operation is the last one that will be applied to the vertices.
		// third, translate the model
		result = glm::translate(result, m_position);
		// second, rotate the model
		result *= glm::mat4_cast(m_orientation);
		// first, scale the model
		result = glm::scale(result, m_scale);
		// update the cached modelToWorld matrix
		m_cachedModelToWorld = result;
	}


}