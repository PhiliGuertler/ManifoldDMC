#include "LazyEngine/gepch.h"
#include "OrthographicController.h"

namespace LazyEngine {


	// ##################################################################### //
	// ### OrthographicOrbitalCameraController ############################# //
	// ##################################################################### //

	OrthographicOrbitalCameraController::OrthographicOrbitalCameraController(float left, float right, float bottom, float top)
		: OrbitalCameraController()
		, m_camera(left, right, bottom, top)
		, m_left(left)
		, m_right(right)
		, m_bottom(bottom)
		, m_top(top)
	{
		// empty
	}

	const Camera& OrthographicOrbitalCameraController::getCamera() const {
		return m_camera;
	}

	float OrthographicOrbitalCameraController::getAspectRatio() const {
		return (m_right - m_left) / (m_top - m_bottom);
	}

	void OrthographicOrbitalCameraController::updateViewMatrix() {
		m_camera.setViewMatrix(m_cameraObject3D.getWorldToModel());
	}

	void OrthographicOrbitalCameraController::updateProjection(float aspectRatio, float width, float height) {
		m_camera.setProjection(-width * 0.5f, width * 0.5f, -height * 0.5f, height * 0.5f);
	}

	// ##################################################################### //
	// ### OrthographicFreeCameraController ################################ //
	// ##################################################################### //

	OrthographicFreeCameraController::OrthographicFreeCameraController(float left, float right, float bottom, float top)
		: FreeCameraController()
		, m_camera(left, right, bottom, top)
		, m_left(left)
		, m_right(right)
		, m_bottom(bottom)
		, m_top(top)
	{
		// empty
	}

	const Camera& OrthographicFreeCameraController::getCamera() const {
		return m_camera;
	}

	float OrthographicFreeCameraController::getAspectRatio() const {
		return (m_right - m_left) / (m_top - m_bottom);
	}

	void OrthographicFreeCameraController::updateViewMatrix() {
		m_camera.setViewMatrix(m_cameraObject3D.getWorldToModel());
	}

	void OrthographicFreeCameraController::updateProjection(float aspectRatio, float width, float height) {
		m_camera.setProjection(-width * 0.5f, width * 0.5f, -height * 0.5f, height * 0.5f);
	}

	// ##################################################################### //
	// ### OrthographicGroundedCameraController ############################ //
	// ##################################################################### //

	OrthographicGroundedCameraController::OrthographicGroundedCameraController(float left, float right, float top, float bottom)
		: GroundedCameraController()
		, m_camera(left, right, bottom, top)
		, m_left(left)
		, m_right(right)
		, m_bottom(bottom)
		, m_top(top)
	{
		// empty
	}

	const Camera& OrthographicGroundedCameraController::getCamera() const {
		return m_camera;
	}

	float OrthographicGroundedCameraController::getAspectRatio() const {
		return (m_right - m_left) / (m_top - m_bottom);
	}

	void OrthographicGroundedCameraController::updateViewMatrix() {
		m_camera.setViewMatrix(m_cameraObject3D.getWorldToModel());
	}

	void OrthographicGroundedCameraController::updateProjection(float aspectRatio, float width, float height) {
		m_camera.setProjection(-width*0.5f, width*0.5f, -height*0.5f, height*0.5f);
	}


	// ##################################################################### //
	// ### Orthographic2DCameraController ############################ //
	// ##################################################################### //

	Orthographic2DCameraController::Orthographic2DCameraController(float left, float right, float top, float bottom)
		: PannableCameraController(1.f)
		, m_camera(left, right, bottom, top)
		, m_left(left)
		, m_right(right)
		, m_bottom(bottom)
		, m_top(top)
	{
		// empty
	}

	Orthographic2DCameraController::Orthographic2DCameraController(float aspectRatio)
		: PannableCameraController(1.f)
		, m_camera(-aspectRatio * m_zoomFactor, aspectRatio* m_zoomFactor, -m_zoomFactor, m_zoomFactor)
		, m_left(-aspectRatio * m_zoomFactor)
		, m_right(aspectRatio* m_zoomFactor)
		, m_bottom(m_zoomFactor)
		, m_top(m_zoomFactor)
	{
		// empty
	}


	const Camera& Orthographic2DCameraController::getCamera() const {
		return m_camera;
	}

	float Orthographic2DCameraController::getAspectRatio() const {
		return (m_right - m_left) / (m_top - m_bottom);
	}

	void Orthographic2DCameraController::updateViewMatrix() {
		m_camera.setViewMatrix(m_cameraObject3D.getWorldToModel());
	}

	void Orthographic2DCameraController::updateProjection(float aspectRatio, float width, float height) {
		// Keep the Zoomfactor from before
		//m_zoomFactor = width / aspectRatio;
		m_left = -aspectRatio * m_zoomFactor;
		m_right = aspectRatio * m_zoomFactor;
		m_bottom = -m_zoomFactor;
		m_top = m_zoomFactor;
		m_camera.setProjection(m_left, m_right, m_bottom, m_top);
	}

	void Orthographic2DCameraController::updateProjectionInternal() {
		// update the aspectRatio to match the new window size
		float aspectRatio = getAspectRatio();

		// update the projection matrix of the camera
		m_camera.setProjection(-aspectRatio * m_zoomFactor, aspectRatio * m_zoomFactor, -m_zoomFactor, m_zoomFactor);

	}

}
