// ######################################################################### //
// ### CameraController.cpp ################################################ //
// ### Implements CameraController.h                                     ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"
#include "CameraController.h"

#include "LazyEngine/Core/Input/Input.h"
#include "LazyEngine/Events/KeyEvent.h"

#include "LazyEngine/Core/Application.h" // used to get the window of this application

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>

namespace LazyEngine {

	__host__
	CameraRay::CameraRay(const CameraController& camera, const glm::vec2& pixelCoordinates)
		: m_direction()
		, m_origin()
	{
		extractRayFromCamera(camera, pixelCoordinates);
	}

	__host__
	void CameraRay::extractRayFromCamera(const CameraController& camera, const glm::vec2& pixelCoordinates) {
		// transform pixel into screen space coordinates
		glm::vec4 screenSpacePixel = transformPixelToScreenSpace(camera.getCamera().getProjectionMatrix(), pixelCoordinates);

		// transform the screen space coordinates into world space coordinates
		glm::mat4 screenToWorld = glm::inverse(camera.getCamera().getViewMatrix());
		m_origin = camera.getPosition();
		m_direction = glm::vec3(screenToWorld * screenSpacePixel) - m_origin;
		m_direction = glm::normalize(m_direction);
	}

	__host__
	glm::vec4 CameraRay::transformPixelToScreenSpace(const glm::mat4& projection, const glm::vec2& pixelCoordinates) {
		glm::ivec2 windowSize = Application::getInstance().getMainViewportSize();
		glm::ivec2 pixelOffset = Application::getInstance().getMainViewport().getOffset();

		glm::vec2 coordinates = glm::ivec2(pixelCoordinates) - pixelOffset;


		// Transform the pixelCoordinates into Screen-Space Coordinates ([-1, 1]^3)
		glm::vec2 screenSpacePixel = 2.f * glm::vec2(coordinates.x, coordinates.y) / glm::vec2(windowSize);
		screenSpacePixel.x -= 1.f;
		screenSpacePixel.y = 1.f - screenSpacePixel.y;
		
		glm::vec4 screenPixel = glm::vec4(screenSpacePixel, 0.f, 1.f);
		screenPixel = glm::inverse(projection) * screenPixel;
		// homogenize the coordinates.
		screenPixel /= screenPixel.w;

		return screenPixel;
	}


	// ##################################################################### //
	// ### CameraController ################################################ //
	// ##################################################################### //

	CameraController::CameraController()
		: m_cameraObject3D()
		, m_cameraSpeed(1.f)
		, m_panningSpeed(1.f)
		, m_lastMousePosition(glm::vec2(0.f, 0.f))
		, m_mouseSensitivity(glm::vec2(1.f, 1.f))
		, m_cursorIsGrabbed(false)
		, m_gamepad(nullptr)
		, m_leftStickSensitivity({ 1.f, 1.f })
		, m_rightStickSensitivity({10.f, 10.f})
	{
		// empty
	}

	CameraController::~CameraController() {
		// empty
	}

	void CameraController::onUpdate(const TimeStep& deltaTime) {
		pollKeyboard(deltaTime);

		if (m_gamepad && m_gamepad->isConnected()) {
			pollGamepad(deltaTime);
		}

		updateViewMatrix();
	}

	void CameraController::onEvent(Event& event) {
		EventDispatcher dispatcher(event);

		// handle WindowResizeEvents in onWindowResized
		dispatcher.dispatch<WindowResizeEvent>(LAZYENGINE_BIND_EVENT_FUNC(CameraController::onWindowResizedEvent));
		dispatcher.dispatch<MouseButtonPressedEvent>(LAZYENGINE_BIND_EVENT_FUNC(CameraController::onMouseButtonPressedEvent));
		dispatcher.dispatch<MouseMovedEvent>(LAZYENGINE_BIND_EVENT_FUNC(CameraController::onMouseMovedEvent));
		dispatcher.dispatch<MouseScrolledEvent>(LAZYENGINE_BIND_EVENT_FUNC(CameraController::onMouseScrolledEvent));

		// These event-functions are not meant for typical movement, as polling is better suited for this.
		// However, they can be used for Sprinting, Flying, Jumping, etc, when extending this class.
		// dispatcher.dispatch<KeyPressedEvent>();
		// dispatcher.dispatch<GamepadButtonPressedEvent>();
	}

	void CameraController::rotateBy(const glm::vec3& axis, float angle) {
		m_cameraObject3D.rotateBy(axis, angle);
		updateViewMatrix();
	}

	void CameraController::rotateLocallyBy(const glm::vec3& axis, float angle) {
		m_cameraObject3D.rotateLocallyBy(axis, angle);
		updateViewMatrix();
	}

	void CameraController::setPosition(const glm::vec3& position) {
		m_cameraObject3D.setPosition(position);
		updateViewMatrix();
	}

	void CameraController::setOrientation(const glm::quat& orientation) {
		m_cameraObject3D.setOrientation(orientation);
		updateViewMatrix();
	}

	void CameraController::setOrientation(const glm::vec3& eulerAngles) {
		m_cameraObject3D.setOrientation(eulerAngles);
		updateViewMatrix();
	}

	void CameraController::lookAt(const glm::vec3& target, const glm::vec3& upVector) {
		m_cameraObject3D.lookAt(target, upVector);
		updateViewMatrix();
	}

	void CameraController::setCameraSpeed(float speed) {
		m_cameraSpeed = speed;
	}

	void CameraController::setPanningSpeed(float speed) {
		m_panningSpeed = speed;
	}

	glm::vec3 CameraController::getPosition() const {
		return m_cameraObject3D.getPosition();
	}

	glm::quat CameraController::getOrientation() const {
		return m_cameraObject3D.getOrientation();
	}

	glm::vec3 CameraController::getForwardDirection() const {
		return m_cameraObject3D.getForwardDirection();
	}

	float CameraController::getCameraSpeed() const {
		return m_cameraSpeed;
	}

	float CameraController::getPanningSpeed() const {
		return m_panningSpeed;
	}

	void CameraController::setGamepad(Ref<Gamepad> gamepad) {
		if (gamepad == nullptr && m_gamepad != nullptr) {
			// don't update the gamepad, as it might be in use right now.
		}
		else {
			m_gamepad = gamepad;
		}
	}

	void CameraController::pollKeyboard(const TimeStep& deltaTime) {
		glm::vec3 moveDirection = glm::vec3(0.f);

		if (Input::isKeyPressed(KeyCode::W)) {
			// move forward
			moveDirection += glm::vec3(0.f, 0.f, -1.f);
		}
		if (Input::isKeyPressed(KeyCode::A)) {
			// move left
			moveDirection += glm::vec3(-1.f, 0.f, 0.f);
		}
		if (Input::isKeyPressed(KeyCode::S)) {
			// move back
			moveDirection += glm::vec3(0.f, 0.f, 1.f);
		}
		if (Input::isKeyPressed(KeyCode::D)) {
			// move right
			moveDirection += glm::vec3(1.f, 0.f, 0.f);
		}
		if (glm::length(moveDirection) > Constants::EPSILON) {
			 moveDirection = glm::normalize(moveDirection) * deltaTime.getSeconds();
			 glm::vec3 transformedMoveDirection = (getOrientation() * moveDirection);

			handleKeyboardMovement(transformedMoveDirection * m_cameraSpeed);
		}
	}

	void CameraController::pollGamepad(const TimeStep& deltaTime) {
		// update the camera's position from the gamepad's left stick
		glm::vec2 movement = m_gamepad->getStickValue(GamepadStickCode::Left_Stick);
		// invert the y-Axis of the movement
		movement.y = -movement.y;
		// apply the stick sensitivity
		movement *= m_leftStickSensitivity * deltaTime.getSeconds();

		// update the camera's view from the gamepad's left stick
		if (!m_gamepad) return;
		glm::vec2 stickValue = m_gamepad->getStickValue(GamepadStickCode::Right_Stick);
		// invert the y-Axis
		stickValue.y = -stickValue.y;
		stickValue = stickValue * m_rightStickSensitivity * deltaTime.getSeconds();
		
		handleGamepadStickMovement(movement, stickValue);

		if (!m_gamepad) return;
		float leftTrigger = m_gamepad->getAxisValue(GamepadAxisCode::Left_Trigger);
		if (!m_gamepad) return;
		float rightTrigger = m_gamepad->getAxisValue(GamepadAxisCode::Right_Trigger);

		handleGamepadTriggerMovement(leftTrigger * deltaTime.getSeconds(), rightTrigger * deltaTime.getSeconds());
	}

	void CameraController::handleGamepadStickMovement(const glm::vec2& leftStick, const glm::vec2& rightStick) {
		glm::vec3 movement3D = glm::vec3(leftStick.x, 0.f, leftStick.y);
		float distance = glm::length(movement3D);
		if (distance > Constants::EPSILON) {
			movement3D = getOrientation() * glm::normalize(movement3D);
			movement3D *= distance;
			// apply the movement as regular keyboard movement
			handleKeyboardMovement(movement3D * m_cameraSpeed);
		}

		handleMouseMovement(rightStick * m_panningSpeed);
	}

	void CameraController::handleGamepadTriggerMovement(float leftTrigger, float rightTrigger) {
		// do nothing by default
	}

	bool CameraController::onMouseMovedEvent(MouseMovedEvent& event) {
		bool consumeEvent = false;
		if (Input::isMouseButtonPressed(MouseButtonCode::Button_Left)) {
			// compute the mouse-movement
			glm::vec2 currentMousePosition = event.getPosition();
			glm::vec2 mouseMovement = currentMousePosition - m_lastMousePosition;
			mouseMovement /= Application::getInstance().getWindow().getDPI();
			mouseMovement *= m_mouseSensitivity;
			m_lastMousePosition = currentMousePosition;

			consumeEvent = handleMouseMovement(mouseMovement);
		}

		return consumeEvent;
	}

	bool CameraController::onMouseButtonPressedEvent(MouseButtonEvent& event) {
		// update the last mouse position to the position of a left click
		if (event.getMouseButton() == MouseButtonCode::Button_Left) {
			m_lastMousePosition = Input::getMousePosition();
		}

		// don't consume the event
		return false;
	}

	bool CameraController::onMouseScrolledEvent(MouseScrolledEvent& event) {
		return handleMouseScoll(glm::vec2(event.getXOffset(), event.getYOffset()));
	}


	bool CameraController::onWindowResizedEvent(WindowResizeEvent& event) {

		// don't touch anything if the window is minimized or any dimension is zero
		if (Application::getInstance().getWindow().isMinimized() || event.getWidth() == 0 || event.getHeight() == 0) {
			return false;
		}

		// update the aspectRatio to match the new window size
		float aspectRatio = static_cast<float>(event.getWidth()) / static_cast<float>(event.getHeight());

		// update the projection matrix of the camera
		updateProjection(aspectRatio, event.getWidth(), event.getHeight());

		// don't consume the event
		return false;
	}

	void CameraController::captureMouse(bool capture) {
		m_cursorIsGrabbed = capture;

		// get the window and tell it to hide the mouse cursor
		auto& window = Application::getInstance().getWindow();
		window.grabMouseCursor(capture);

		// update the last mouse position as it might have been warped to the center of the window
		m_lastMousePosition = Input::getMousePosition();

	}

	CameraRay CameraController::shootPixelRay(const glm::vec2& pixelPosition) const {
		return CameraRay(*this, pixelPosition);
	}


	// ##################################################################### //
	// ### OrbitalCameraController ######################################### //
	// ##################################################################### //

	OrbitalCameraController::OrbitalCameraController(const glm::vec3& focusPoint)
		: CameraController()
		, m_focusPoint(focusPoint)
	{
		// empty
	}

	OrbitalCameraController::~OrbitalCameraController() {
		// empty
	}

	void OrbitalCameraController::setFocusPoint(const glm::vec3& focusPoint) {
		m_focusPoint = focusPoint;
		glm::vec3 upVector = getOrientation() * glm::vec3(0.f, 1.f, 0.f);
		lookAt(focusPoint, upVector);
	}

	void OrbitalCameraController::handleKeyboardMovement(const glm::vec3& movementWorldSpace) {
		// undo the rotation of the movement direction in world space
		glm::vec3 originalMovement = glm::inverse(getOrientation()) * movementWorldSpace;
		handleMouseMovement(glm::vec2(-originalMovement.x, -originalMovement.z));
	}

	bool OrbitalCameraController::handleMouseMovement(const glm::vec2& mouseMovement) {
		glm::vec3 toFocusPoint = m_focusPoint - getPosition();
		float distance = glm::length(toFocusPoint);

		// Rotate the camera orientation like normal
		rotateLocallyBy(glm::vec3(0.f, -1.f, 0.f), mouseMovement.x);
		rotateLocallyBy(glm::vec3(-1.f, 0.f, 0.f), mouseMovement.y);

		// set the camera's position to directly look at the focus point from the new orientation
		toFocusPoint = getForwardDirection() * distance;
		setPosition(m_focusPoint - toFocusPoint);

		// don't consume the event
		return false;
	}

	bool OrbitalCameraController::handleMouseScoll(const glm::vec2& mouseScroll) {
		glm::vec3 viewDirection = m_focusPoint - getPosition();
		float distance = glm::length(viewDirection);
		viewDirection = getForwardDirection();

		if (distance < LazyEngine::Constants::EPSILON) {
			distance = 0.05f;
		}
		distance *= 1.f - (mouseScroll.y * 0.1f);

		setPosition(m_focusPoint - viewDirection * distance);

		// don't consume the event
		return false;
	}

	void OrbitalCameraController::handleGamepadTriggerMovement(float leftTrigger, float rightTrigger) {
		glm::vec3 viewDirection = m_focusPoint - getPosition();
		float distance = glm::length(viewDirection);
		viewDirection = getForwardDirection();

		if (distance < LazyEngine::Constants::EPSILON) {
			distance = 0.05f;
		}
		distance *= 1.f - ((rightTrigger-leftTrigger));

		setPosition(m_focusPoint - viewDirection * distance);
	}



	// ##################################################################### //
	// ### FreeCameraController ############################################ //
	// ##################################################################### //


	void FreeCameraController::handleKeyboardMovement(const glm::vec3& movementWorldSpace) {
		setPosition(getPosition() + movementWorldSpace);
	}

	bool FreeCameraController::handleMouseMovement(const glm::vec2& mouseMovement) {
		// Rotate the camera orientation freely around its model axes
		rotateLocallyBy(glm::vec3(0.f, -1.f, 0.f), mouseMovement.x);
		rotateLocallyBy(glm::vec3(-1.f, 0.f, 0.f), mouseMovement.y);

		// don't consume the event
		return false;
	}

	bool FreeCameraController::handleMouseScoll(const glm::vec2& mouseScroll) {
		m_cameraSpeed *= (1.f + (mouseScroll.y * 0.1f));
		
		// don't consume the event
		return false;
	}


	// ##################################################################### //
	// ### GroundedCameraController ######################################## //
	// ##################################################################### //

	GroundedCameraController::GroundedCameraController()
		: CameraController()
		, m_yaw(0.f)
		, m_pitch(0.f)
		, m_forward({0.f, 0.f, -1.f})
		, m_right({1.f, 0.f, 0.f})
		, m_up({ 0.f, 1.f, 0.f })
	{
		// cast the orientation to euler angles
		auto euler = glm::eulerAngles(getOrientation());
		// looking into -z means a yaw of -PI/2
		yaw(euler.y - static_cast<float>(Constants::PI) * 0.5f);
		pitch(euler.x);
	}

	GroundedCameraController::~GroundedCameraController() {
		// empty
	}

	void GroundedCameraController::handleKeyboardMovement(const glm::vec3& movementWorldSpace) {
		// ignore the y-component of the movement, as this would lead to flying
		//glm::vec3 movement = glm::normalize(glm::vec3(movementWorldSpace.x, 0.f, movementWorldSpace.z)) * glm::length(movementWorldSpace);
		glm::vec3 movement = movementWorldSpace;

		setPosition(getPosition() + movement);
	}

	bool GroundedCameraController::handleMouseMovement(const glm::vec2& mouseMovement) {
		yaw(mouseMovement.x);
		pitch(-mouseMovement.y);

		updateOrientation();

		// don't consume the event
		return false;
	}

	bool GroundedCameraController::handleMouseScoll(const glm::vec2& mouseScroll) {
		// TODO: handle zoom or something similar

		// don't consume the event
		return false;
	}

	void GroundedCameraController::updateOrientation() {
		glm::mat4 newView = glm::lookAt(getPosition(), getPosition() + m_forward, m_up);

		setOrientation(glm::normalize(glm::conjugate(glm::quat_cast(newView))));
	}

	void GroundedCameraController::yaw(float deltaAngle) {
		//if (abs(deltaAngle) < Constants::EPSILON) return;

		m_yaw += deltaAngle;

		// calculate new forward vector
		m_forward = glm::normalize(m_forward * std::cos(deltaAngle) + m_right * std::sin(deltaAngle));

		// calculate new right vector
		m_right = glm::normalize(glm::cross(m_forward, m_up));

		// This part is not suitable for a flying camera
		m_forward.x = std::cos(m_yaw) * std::cos(m_pitch);
		m_forward.y = std::sin(m_pitch);
		m_forward.z = std::sin(m_yaw) * std::cos(m_pitch);
		m_forward = glm::normalize(m_forward);

	}

	void GroundedCameraController::pitch(float deltaAngle) {
		//if (abs(deltaAngle) < Constants::EPSILON) return;
		const float PI_HALF = Constants::PI * 0.5f;

		m_pitch += deltaAngle;

		// clamp pitch to be between -PI/2 and PI/2 to avoid weird loopings that will break the camera
		m_pitch = std::min(PI_HALF - Constants::EPSILON, std::max(-PI_HALF + Constants::EPSILON, m_pitch));

		// calculate new forward vector
		m_forward = glm::normalize(m_forward * std::cos(deltaAngle) + m_up * sin(deltaAngle));

		// calculate new up vector
		//m_up = glm::normalize(glm::cross(m_right, m_forward));

		// This part is not suitable for a flying camera
		m_forward.x = std::cos(m_yaw) * std::cos(m_pitch);
		m_forward.y = std::sin(m_pitch);
		m_forward.z = std::sin(m_yaw) * std::cos(m_pitch);
		m_forward = glm::normalize(m_forward);
	}

	void GroundedCameraController::lookAt(const glm::vec3& target, const glm::vec3& upVector) {
		CameraController::lookAt(target, upVector);

		// update orientation
		m_pitch = 0.f;
		m_yaw = 0.f;
		// cast the orientation to euler angles
		auto euler = glm::eulerAngles(getOrientation());
		// looking into -z means a yaw of -PI/2
		yaw(euler.y - static_cast<float>(Constants::PI) * 0.5f);
		pitch(euler.x);
	}


	// ##################################################################### //
	// ### PannableCameraController ######################################## //
	// ##################################################################### //

	PannableCameraController::PannableCameraController(float zoomFactor)
		: CameraController()
		, m_zoomFactor(zoomFactor)
	{
		// empty
	}


	void PannableCameraController::handleKeyboardMovement(const glm::vec3& movementWorldSpace) {
		// disabled by default
	}

	bool PannableCameraController::handleMouseMovement(const glm::vec2& mouseMovement) {
		// restore the original mouse movement
		glm::vec2 mouseMovementOriginal = mouseMovement / m_mouseSensitivity;
		mouseMovementOriginal *= Application::getInstance().getWindow().getDPI();

		// map the mouse movement to [-1, 1] inside the window
		float aspectRatio = getAspectRatio();
		// Viewport Size is needed, not window size!
		const glm::vec2 viewportSize = glm::vec2(Application::getInstance().getMainViewportSize());
		if (viewportSize.x < 0 || viewportSize.y < 0) {
			// The viewport does not exist
			return false;
		}
		//const glm::vec2 viewportSize = glm::vec2(Application::getInstance().getWindow().getSize());
		glm::vec2 mappedMovement = mouseMovementOriginal / viewportSize;
		mappedMovement = mappedMovement * m_zoomFactor * glm::vec2(2.f * aspectRatio, 2.f);

		setPosition(getPosition() + glm::vec3(-mappedMovement.x, mappedMovement.y, 0.f));

		// don't consume the event
		return false;
	}

	bool PannableCameraController::handleMouseScoll(const glm::vec2& mouseScroll) {
		// modify the zoom level depending on the scroll event
		float deltaZoom = mouseScroll.y * 0.1f * m_zoomFactor;
		m_zoomFactor -= deltaZoom;
		m_zoomFactor = std::max(m_zoomFactor, 0.01f);

		// update the projection matrix
		updateProjectionInternal();

		// don't consume the event
		return false;

	}

	void PannableCameraController::setZoom(float zoomFactor) {
		m_zoomFactor = zoomFactor;

		updateProjectionInternal();
	}

}
