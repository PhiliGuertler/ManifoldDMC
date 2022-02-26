#pragma once

// ######################################################################### //
// ### Camera.h ############################################################ //
// ### Defines the Superclass Camera and derivatives like an Orthogra-   ### //
// ### phic Camera and a Perspective Camera.                             ### //
// ######################################################################### //

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// For some reason, windows defined near and far somewhere.
#undef near
#undef far

namespace LazyEngine {

	/**
	 *	The interface of a camera providing access to all of its matrices
	 */
	class Camera {
	public:
		// a constructor is not nescessary
		
		/**
		 *	default destructor
		 */
		virtual ~Camera() = default;

		/**
		 *	returns the projection matrix of this camera.
		 *	For an orthographic camera this will be an orthographic projection.
		 */
		virtual const glm::mat4& getProjectionMatrix() const = 0;
		/**
		 *	returns the view matrix of this camera.
		 */
		virtual const glm::mat4& getViewMatrix() const = 0;
		/**
		 *	returns Projection * View
		 */
		virtual const glm::mat4& getProjectionViewMatrix() const = 0;
	};

	/**
	 *	Orhographic 2D Camera
	 */
	class OrthographicCamera : public Camera {
	public:
		/**
		 *	constructor
		 *	the parameters will determine the aspect ratio of this camera
		 *	@param left: a float representing the left part of the frustum
		 *	@param right: a float representing the right part of the frustum
		 *	@param bottom: a float representing the bottom part of the frustum
		 *	@param top: a float representing the top part of the frustum
		 */
		OrthographicCamera(float left, float right, float bottom, float top);
		/**
		 *	default destructor
		 */
		virtual ~OrthographicCamera() = default;

		/**
		 *	updates the projection of this camera
		 *	the parameters will determine the aspect ratio of this camera
		 *	@param left: a float representing the left part of the frustum
		 *	@param right: a float representing the right part of the frustum
		 *	@param bottom: a float representing the bottom part of the frustum
		 *	@param top: a float representing the top part of the frustum
		 */
		void setProjection(float left, float right, float bottom, float top);

		void setPositionAndRotation(const glm::vec3& position, float rotation);

		/**
		 *	sets the view matrix of this camera directly
		 *	@param viewMatrix: the new view matrix
		 */
		inline void setViewMatrix(const glm::mat4& viewMatrix) { m_viewMatrix = viewMatrix; recomputeMatrices(); }

		/**
		 *	returns the projection matrix of this camera which is orthographic.
		 */
		const glm::mat4& getProjectionMatrix() const override { return m_projectionMatrix; }
		/**
		 *	returns the view matrix of this camera.
		 */
		const glm::mat4& getViewMatrix() const override { return m_viewMatrix; }
		/**
		 *	returns Projection * View
		 */
		const glm::mat4& getProjectionViewMatrix() const override { return m_projectionViewMatrix; }

	private:
		void recomputeMatrices();

	private:
		glm::mat4 m_projectionMatrix;
		glm::mat4 m_viewMatrix;
		// cached viewProjectionMatrix
		glm::mat4 m_projectionViewMatrix;
	};


	class PerspectiveCamera : public Camera {
	public:
		/**
		 *	constructor
		 *	@param aspectRatio: the aspect ratio of the camera's viewport
		 *	@param near: the near plane of the camera's view frustum
		 *	@param far: the far plane of the camera's view frustum
		 *	@param fovY: the field of view 
		 */
		PerspectiveCamera(float aspectRatio, float fovY, float near = 0.05f, float far = 1000.f);

		/**
		 *	Updates the view matrix of this camera to look at a specific point.
		 *	This will consider the current position of the camera
		 *	@param target: The point to look at
		 *	@param up: The up-vector of the viewport (this is [0,1,0] by default)
		 */
		void lookAt(const glm::vec3& cameraPosition, const glm::vec3& target, const glm::vec3& up = glm::vec3(0.f,1.f,0.f));

		/**
		 *	sets the view matrix of this camera directly
		 *	@param viewMatrix: the new view matrix
		 */
		inline void setViewMatrix(const glm::mat4& viewMatrix) { m_viewMatrix = viewMatrix; recomputeMatrices(); }

		/**
		 *	sets the projection of this camera
		 *	@param aspectRatio: the aspect ratio of the camera's viewport
		 *	@param near: the near plane of the camera's view frustum
		 *	@param far: the far plane of the camera's view frustum
		 *	@param fovY: the field of view
		 */
		void setProjection(float aspectRatio, float fovY, float near = 0.05f, float far = 1000.f);

		/**
		 *	sets the projection matrix of this camera directly
		 *	@param projectionMatrix: the new projection matrix
		 */
		inline void setProjectionMatrix(const glm::mat4& projectionMatrix) { m_projectionMatrix = projectionMatrix; }

		/**
		 *	returns the projection matrix of this camera.
		 *	For an orthographic camera this will be an orthographic projection.
		 */
		virtual inline const glm::mat4& getProjectionMatrix() const override { return m_projectionMatrix; }
		/**
		 *	returns the view matrix of this camera.
		 */
		virtual inline const glm::mat4& getViewMatrix() const override { return m_viewMatrix; }
		/**
		 *	returns Projection * View
		 */
		virtual inline const glm::mat4& getProjectionViewMatrix() const override { return m_projectionViewMatrix; }
	private:
		void recomputeMatrices();

	private:
		glm::mat4 m_projectionMatrix;
		glm::mat4 m_viewMatrix;
		// cached viewProjectionMatrix
		glm::mat4 m_projectionViewMatrix;
	};

}