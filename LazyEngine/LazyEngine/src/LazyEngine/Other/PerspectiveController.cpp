#include "LazyEngine/gepch.h"
#include "PerspectiveController.h"
#include "LazyEngine/Core/Util.h"

namespace LazyEngine {

	// ##################################################################### //
	// ### PerspectiveQuirks ############################################### //
	// ##################################################################### //

	std::pair<Object3D, PerspectiveQuirks> PerspectiveQuirks::extractInfosFromCamera(const PerspectiveCamera& camera) {

		// extract near, far, fovY, position, orientation and aspect ratio from the camera's projection matrix
		auto projection = camera.getProjectionMatrix();
		float near = projection[3][2] / (projection[2][2] - 1.0f);
		float far = projection[3][2] / (projection[2][2] + 1.0f);
		float aspectRatio = projection[1][1] / projection[0][0];
		float fovY = 2.0f * std::atan2(1.0f, projection[1][1]);

		// create the PerspectiveQuirks object
		PerspectiveQuirks quirks(aspectRatio, fovY, near, far);

		Object3D worldPosition;
		auto modelMatrix = glm::inverse(camera.getViewMatrix());
		worldPosition.setOrientation(glm::normalize(glm::quat_cast(glm::mat3(modelMatrix))));
		worldPosition.setPosition(glm::vec3(modelMatrix[3][0], modelMatrix[3][1], modelMatrix[3][2]));

		return { worldPosition, quirks };
	}

	PerspectiveQuirks::PerspectiveQuirks(float aspectRatio, float fovY, float near, float far)
		: m_camera(aspectRatio, fovY, near, far)
		, m_aspectRatio(aspectRatio)
		, m_fovY(fovY)
		, m_near(near)
		, m_far(far)
	{
		// empty
	}

	PerspectiveQuirks::~PerspectiveQuirks() {
		// empty
	}

	const Camera& PerspectiveQuirks::getCamera() const {
		return m_camera;
	}

	float PerspectiveQuirks::getAspectRatio() const {
		return m_aspectRatio;
	}

	void PerspectiveQuirks::updateViewMatrix(const glm::mat4& viewMatrix) {
		m_camera.setViewMatrix(viewMatrix);
	}

	void PerspectiveQuirks::updateProjection(float aspectRatio, float fovY, float near, float far) {
		m_aspectRatio = aspectRatio;
		m_fovY = fovY < 0.f ? m_fovY : fovY;
		m_near = near < 0.f ? m_near : near;
		m_far = far < 0.f ? m_far : far;
		m_camera.setProjection(m_aspectRatio, m_fovY, m_near, m_far);
	}

	void PerspectiveQuirks::clipOblique(const glm::vec3& position, const glm::vec3& normal) {
		// A whole lot of mathmagics, copied from somewhere on the Internet
		const glm::vec3 clipPosition = glm::vec3(m_camera.getViewMatrix() * glm::vec4(position, 1.f));
		const glm::vec3 clipNormal = glm::vec3(m_camera.getViewMatrix() * glm::vec4(normal, 0.f));
		const glm::vec4 clipPlane = glm::vec4(clipNormal, -glm::dot(clipPosition, clipNormal));

		glm::mat4 projection = m_camera.getProjectionMatrix();
		const glm::vec4 q = glm::inverse(projection) * glm::vec4(
			Util::sgn(clipPlane.x),
			Util::sgn(clipPlane.y),
			1.f,
			1.f
		);

		const glm::vec4 c = clipPlane * (2.f / glm::dot(clipPlane, q));

		projection[0][2] = c.x - projection[0][3];
		projection[1][2] = c.y - projection[1][3];
		projection[2][2] = c.z - projection[2][3];
		projection[3][2] = c.w - projection[3][3];

		m_camera.setProjectionMatrix(projection);
	}


	// ##################################################################### //
	// ### PerspectiveOrbitalCameraController ############################## //
	// ##################################################################### //

	PerspectiveOrbitalCameraController::PerspectiveOrbitalCameraController(float aspectRatio, float fovY, float near, float far)
		: OrbitalCameraController()
		, m_quirks(aspectRatio, fovY, near, far)
	{
		// empty
	}

	PerspectiveOrbitalCameraController::PerspectiveOrbitalCameraController(const PerspectiveCamera& camera)
		: OrbitalCameraController()
		, m_quirks(0.f, 0.f, 0.f, 0.f)
	{
		auto unpacked = PerspectiveQuirks::extractInfosFromCamera(camera);
		m_cameraObject3D = unpacked.first;
		m_quirks = unpacked.second;
	}


	const Camera& PerspectiveOrbitalCameraController::getCamera() const {
		return m_quirks.getCamera();
	}

	float PerspectiveOrbitalCameraController::getAspectRatio() const {
		return m_quirks.getAspectRatio();
	}

	void PerspectiveOrbitalCameraController::updateViewMatrix() {
		m_quirks.updateViewMatrix(m_cameraObject3D.getWorldToModel());
	}

	void PerspectiveOrbitalCameraController::updateProjection(float aspectRatio, float width, float height) {
		m_quirks.updateProjection(aspectRatio);
	}

	void PerspectiveOrbitalCameraController::clipOblique(const glm::vec3& position, const glm::vec3& normal) {
		m_quirks.clipOblique(position, normal);
	}

	// ##################################################################### //
	// ### PerspectiveFreeCameraController ################################# //
	// ##################################################################### //

	PerspectiveFreeCameraController::PerspectiveFreeCameraController(float aspectRatio, float fovY, float near, float far)
		: FreeCameraController()
		, m_quirks(aspectRatio, fovY, near, far)
	{
		// empty
	}

	PerspectiveFreeCameraController::PerspectiveFreeCameraController(const PerspectiveCamera& camera)
		: FreeCameraController()
		, m_quirks(0.f, 0.f, 0.f, 0.f)
	{
		auto unpacked = PerspectiveQuirks::extractInfosFromCamera(camera);
		m_cameraObject3D = unpacked.first;
		m_quirks = unpacked.second;
	}

	const Camera& PerspectiveFreeCameraController::getCamera() const {
		return m_quirks.getCamera();
	}

	float PerspectiveFreeCameraController::getAspectRatio() const {
		return m_quirks.getAspectRatio();
	}

	void PerspectiveFreeCameraController::updateViewMatrix() {
		m_quirks.updateViewMatrix(m_cameraObject3D.getWorldToModel());
	}

	void PerspectiveFreeCameraController::updateProjection(float aspectRatio, float width, float height) {
		m_quirks.updateProjection(aspectRatio);
	}

	void PerspectiveFreeCameraController::clipOblique(const glm::vec3& position, const glm::vec3& normal) {
		m_quirks.clipOblique(position, normal);
	}

	// ##################################################################### //
	// ### PerspectiveGroundedCameraController ############################# //
	// ##################################################################### //

	PerspectiveGroundedCameraController::PerspectiveGroundedCameraController(float aspectRatio, float fovY, float near, float far)
		: GroundedCameraController()
		, m_quirks(aspectRatio, fovY, near, far)
	{
		// empty
	}

	PerspectiveGroundedCameraController::PerspectiveGroundedCameraController(const PerspectiveCamera& camera)
		: GroundedCameraController()
		, m_quirks(0.f, 0.f, 0.f, 0.f)
	{
		auto unpacked = PerspectiveQuirks::extractInfosFromCamera(camera);
		m_cameraObject3D = unpacked.first;
		m_quirks = unpacked.second;
	}

	const Camera& PerspectiveGroundedCameraController::getCamera() const {
		return m_quirks.getCamera();
	}

	float PerspectiveGroundedCameraController::getAspectRatio() const {
		return m_quirks.getAspectRatio();
	}

	void PerspectiveGroundedCameraController::updateViewMatrix() {
		m_quirks.updateViewMatrix(m_cameraObject3D.getWorldToModel());
	}

	void PerspectiveGroundedCameraController::updateProjection(float aspectRatio, float width, float height) {
		m_quirks.updateProjection(aspectRatio);
	}

	void PerspectiveGroundedCameraController::clipOblique(const glm::vec3& position, const glm::vec3& normal) {
		m_quirks.clipOblique(position, normal);
	}
}
