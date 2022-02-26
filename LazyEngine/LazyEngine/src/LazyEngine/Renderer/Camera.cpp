// ######################################################################### //
// ### Camera.cpp ########################################################## //
// ### Implements Camera.h                                               ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"
#include "Camera.h"

#include "LazyEngine/Core/Util.h"

#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	// ##################################################################### //
	// ### Orthographic Camera ############################################# //
	// ##################################################################### //

	OrthographicCamera::OrthographicCamera(float left, float right, float bottom, float top)
		: m_projectionMatrix(glm::ortho(left, right, bottom, top, -1.f, 1.f))
		, m_viewMatrix(1.f)
		, m_projectionViewMatrix()
	{
		LAZYENGINE_PROFILE_FUNCTION();

		setPositionAndRotation({0.f,0.f,0.f}, 0.f);
	}

	void OrthographicCamera::setProjection(float left, float right, float bottom, float top) {
		LAZYENGINE_PROFILE_FUNCTION();

		m_projectionMatrix = glm::ortho(left, right, bottom, top, -1.f, 1.f);
		m_projectionViewMatrix = m_projectionMatrix * m_viewMatrix;
	}


	void OrthographicCamera::setPositionAndRotation(const glm::vec3& position, float rotation) {
		LAZYENGINE_PROFILE_FUNCTION();

		glm::mat4 transform = glm::translate(glm::mat4(1.f), position);
		transform = glm::rotate(transform, rotation, glm::vec3(0.f, 0.f, 1.f));

		m_viewMatrix = glm::inverse(transform);
		m_projectionViewMatrix = m_projectionMatrix * m_viewMatrix;
	}

	void OrthographicCamera::recomputeMatrices() {
		LAZYENGINE_PROFILE_FUNCTION();
		m_projectionViewMatrix = m_projectionMatrix * m_viewMatrix;
	}


	// ##################################################################### //
	// ### Perspective Camera ############################################## //
	// ##################################################################### //

	PerspectiveCamera::PerspectiveCamera(float aspectRatio, float fovY, float near, float far)
		: Camera()
		, m_projectionMatrix(glm::mat4(1.f))
		, m_viewMatrix(glm::mat4(1.f))
		, m_projectionViewMatrix(glm::mat4(1.f))
	{
		LAZYENGINE_PROFILE_FUNCTION();

		setProjection(aspectRatio, fovY, near, far);
		lookAt(glm::vec3(0.f,1.f,0.f), glm::vec3(0.f,1.f,-1.f), glm::vec3(0.f,1.f,0.f));
		recomputeMatrices();
	}

	void PerspectiveCamera::lookAt(const glm::vec3& cameraPosition, const glm::vec3& target, const glm::vec3& up) {
		LAZYENGINE_PROFILE_FUNCTION();
		m_viewMatrix = glm::lookAt(cameraPosition, target, up);
		recomputeMatrices();
	}

	void PerspectiveCamera::setProjection(float aspectRatio, float fovY, float near, float far) {
		LAZYENGINE_PROFILE_FUNCTION();
		m_projectionMatrix = glm::perspective(fovY, aspectRatio, near, far);
		recomputeMatrices();
	}

	void PerspectiveCamera::recomputeMatrices() {
		LAZYENGINE_PROFILE_FUNCTION();
		m_projectionViewMatrix = m_projectionMatrix * m_viewMatrix;
	}
}