#pragma once

// ######################################################################### //
// ### CameraController.h ################################################## //
// ### Defines wrappers for cameras that will enable user friendly       ### //
// ### control of them while keeping the cameras themselves light weight ### //
// ######################################################################### //

#include "LazyEngine/Renderer/Camera.h"
#include "LazyEngine/Core/Time.h"
#include "LazyEngine/Core/Constants.h"

#include "LazyEngine/Events/ApplicationEvent.h"
#include "LazyEngine/Events/KeyEvent.h"
#include "LazyEngine/Events/MouseEvent.h"

#include "LazyEngine/Core/Input/Gamepad.h"

#include "LazyEngine/Renderer/Geometry/Object3D.h"
#include "LazyEngine/platform/CUDA/CUDAUtils.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace LazyEngine {

	class CameraController;

	/**
	 * Defines a single ray shot from the camera into the scene
	 * TODO: Maybe rename this just "Ray" and move it into Utils
	 */
	class CameraRay {
	public:
		HostDevice
		inline CameraRay(const glm::vec3& direction, const glm::vec3& origin)
			: m_direction(direction)
			, m_origin(origin)
		{ /* empty */ }
		
		HostDevice
		inline CameraRay()
			: m_direction(glm::vec3(NAN))
			, m_origin(glm::vec3(NAN))
		{ /* empty */ }

		__host__
		CameraRay(const CameraController& camera, const glm::vec2& pixelCoordinates);

		HostDevice
		inline glm::vec3 getDirection() const { return m_direction; }

		HostDevice
		inline glm::vec3 getOrigin() const { return m_origin; }

		HostDevice
		inline void setDirection(const glm::vec3& direction) { m_direction = direction; }

		HostDevice
		inline void setOrigin(const glm::vec3& origin) { m_origin = origin; }

		/**
		 *	Computes the signed distance from this ray's origin to a point.
		 *	Negative distances signalize that the ray is pointing away from the point
		 */
		HostDevice
		inline float signedDistanceTo(glm::vec3 point) const {
			glm::vec3 originToIntersection = point - getOrigin();
			float sign = glm::dot(originToIntersection, getDirection()) >= 0.f ? 1.f : -1.f;
			return glm::length(originToIntersection) * sign;
		}

	protected:
		__host__
		void extractRayFromCamera(const CameraController& camera, const glm::vec2& pixelCoordinates);
		__host__
		glm::vec4 transformPixelToScreenSpace(const glm::mat4& projection, const glm::vec2& pixelCoordinates);

	protected:
		glm::vec3 m_direction;
		glm::vec3 m_origin;
	};

	// ##################################################################### //
	// ### CameraController ################################################ //
	// ##################################################################### //

	/**
	 * A Camera Controller that uses an Object 3D as a representation of its wrapped camera in space
	 */
	class CameraController {
	public:
		/**
		 *	constructor
		 */
		CameraController();
		/**
		 *	destructor
		 */
		virtual ~CameraController();

		/**
		 * Handles Camera-Movement in a Polling Fashion by checking the keyboard/Gamepad states
		 * @param deltaTime: The time that passed since the last call (which was probably a frame ago)
		 */
		virtual void onUpdate(const TimeStep& deltaTime);

		/**
		 * handles Event-driven camera movement. This mostly contains Mouse Move and Scroll events or Window resizes
		 * @param event: The (mutable because consumable) event that triggered.
		 */
		virtual void onEvent(Event& event);

		/**
		 * Returns the encapsulated Camera Object to be used in a shader for example
		 */
		virtual const Camera& getCamera() const = 0;

		/**
		 * Rotates the camera clockwise around a World-Space axis.
		 * @param axis: The normalized rotation-axis in world-space
		 * @param angle: The clockwise angle of the rotation
		 */
		virtual void rotateBy(const glm::vec3& axis, float angle);

		/**
		 * Rotates the camera clockwise around a Model-Space axis.
		 * This means, that a rotation around {0,1,0} will always result in a left-right rotation, regardless of the camera's orientation
		 * @param axis: The normalized rotation-axis in world-space
		 * @param angle: The clockwise angle of the rotation
		 */
		virtual void rotateLocallyBy(const glm::vec3& axis, float angle);

		/**
		 * sets the position of the camera
		 * @param position: the new position of the camera in world-space coordinates
		 */
		virtual void setPosition(const glm::vec3& position);

		/**
		 * Sets the orientation of the camera
		 * @param orientation: The new orientation of the camera
		 */
		virtual void setOrientation(const glm::quat& orientation);

		/**
		 * Sets the orientation of the camera
		 * @param eulerAngles: the angles in radians in the order:
		 * 		pitch, yaw, roll
		 */
		virtual void setOrientation(const glm::vec3& eulerAngles);

		/**
		 * Sets a specific target as the center of the Viewport
		 * @param target: the target at which to look
		 * @param upVector: A vector that defines the up-direction
		 */
		virtual void lookAt(const glm::vec3& target, const glm::vec3& upVector = glm::vec3(0.f, 1.f, 0.f));

		/**
		 * Sets the camera's movement speed in units/second.
		 * @param speed: The new movement speed.
		 */
		virtual void setCameraSpeed(float speed);

		/**
		 * Sets the camera's panning speed in no particular unit.
		 * @param speed: the new panning speed.
		 */
		virtual void setPanningSpeed(float speed);

		/**
		 *	Returns the camera's position in world space
		 */
		glm::vec3 getPosition() const;

		/**
		 *	Returns the camera's orientation in world space
		 */
		glm::quat getOrientation() const;

		/**
		 *	Returns a normalized vector in the current view direction of the camera
		 */
		glm::vec3 getForwardDirection() const;

		/**
		 *	Returns the camera's movement speed
		 */
		float getCameraSpeed() const;

		/**
		 *	Returns the camera's panning speed
		 */
		float getPanningSpeed() const;

		/**
		 *	Returns the aspect ratio of the camera
		 */
		virtual float getAspectRatio() const = 0;

		/**
		 *	Sets a Gamepad as control source for this camera controller
		 *  @param gamepad: A Ref to the gamepad that should be used
		 */
		void setGamepad(Ref<Gamepad> gamepad);

		/**
		 *	Captures the mouse, so that every movement of the mouse will be used, without having to left click first
		 */
		virtual void captureMouse(bool capture);

		/**
		 * Shoots a ray into the scene, in the direction of a given pixel.
		 * @param pixelPosition: The pixelPosition (in [0, viewportWidth], [0, viewportHeight])
		 */
		CameraRay shootPixelRay(const glm::vec2& pixelPosition) const;

	protected:
		/**
		 *	Handles keyboard movement with a movement-vector given in world space coordinates
		 */
		virtual void handleKeyboardMovement(const glm::vec3& movementWorldSpace) = 0;
		/**
		 *	Handles mouse movement with a movement-vector given in x- and y-coordinates
		 *  @returns true if the corresponding MouseMovedEvent should be consumed
		 */
		virtual bool handleMouseMovement(const glm::vec2& mouseMovement) = 0;
		/**
		 *	Handles mouse scroll with a scroll-vector given in x- and y-coordinates
		 *  @returns true if the corresponding MouseScrolledEvent should be consumed
		 */
		virtual bool handleMouseScoll(const glm::vec2& mouseScroll) = 0;
		/**
		 *	Handles gamepad stick movements with the time-adjusted values for the left and right stick.
		 *  @param leftStick: The values of the left stick adjusted to frame-time and stick sensitivity
		 *  @param rightStick: The values of the right stick adjusted to frame-time and stick sensitivity
		 */
		virtual void handleGamepadStickMovement(const glm::vec2& leftStick, const glm::vec2& rightStick);
		/**
		 *	Handles gamepad trigger movements with the time-adjusted values for both triggers
		 */
		virtual void handleGamepadTriggerMovement(float leftTrigger, float rightTrigger);
		/**
		 *	Updates the camera's view matrix
		 */
		virtual void updateViewMatrix() = 0;
		/**
		 *	Updates the camera's projection matrix to match the new aspect ratio
		 */
		virtual void updateProjection(float aspectRatio, float width, float height) = 0;
	private:
		/** 
		 *	Handles MouseMovedEvents
		 */
		bool onMouseMovedEvent(MouseMovedEvent& event);
		/**
		 *	Handles MouseScrolledEvents
		 */
		bool onMouseScrolledEvent(MouseScrolledEvent& event);
		/**
		 *	Handles MouseButtonEvents
		 */
		bool onMouseButtonPressedEvent(MouseButtonEvent& event);
		/**
		 *	Handles WindowResizeEvents
		 */
		bool onWindowResizedEvent(WindowResizeEvent& event);

		/**
		 *	Polls the Keyboard by checking its currently pressed keys
		 */
		virtual void pollKeyboard(const TimeStep& deltaTime);

		/**
		 *	Polls the Gamepad by checking its currently pressed keys and axis-states
		 */
		virtual void pollGamepad(const TimeStep& deltaTime);

	protected:
		// The internal representation of the camera's position and orientation in world space
		Object3D m_cameraObject3D;
		
		// The camera's movement speed (in units/second)
		float m_cameraSpeed;
		// The camera's panning speed (in no particular unit)
		float m_panningSpeed;

		// --- Mouse Input --- //
		// The last position of the mouse in pixel coordinates
		glm::vec2 m_lastMousePosition;
		// The sensitivity of the mouse-movement in x- and y-direction in radians per inch (of mouse-cursor movement)
		glm::vec2 m_mouseSensitivity;
		// if true, the mouse will be hidden and kept inside of the window
		bool m_cursorIsGrabbed;

		// --- Gamepad Input --- //
		// FIXME: on disconnections, m_gamepad becomes invalid before it can be unregistered!
		// A reference to the connected gamepad
		Ref<Gamepad> m_gamepad;
		// Stick sensitivities
		glm::vec2 m_rightStickSensitivity;
		glm::vec2 m_leftStickSensitivity;
	};


	// ##################################################################### //
	// ### OrbitalCameraController ######################################### //
	// ##################################################################### //

	/**
	 *	A Camera Controller that orbits around a focus point using mouse panning or WASD-Movement.
	 *  The camera will always look at the given focus point and orbit around it.
	 *  By default the camera spawns at {0,0,0} and looks in {0,0,-1}-Direction, while focusing {0,0,0}
	 */
	class OrbitalCameraController : public CameraController {
	public:
		/**
		 *	constructor
		 */
		OrbitalCameraController(const glm::vec3& focusPoint = glm::vec3(0.f, 0.f, 0.f));
		/**
		 *	destructor
		 */
		virtual ~OrbitalCameraController();

		/**
		 *	Sets the focus point around which to orbit
		 */
		void setFocusPoint(const glm::vec3& focusPoint);

		inline glm::vec3 getFocusPoint() const {
			return m_focusPoint;
		}

	protected:
		/**
		 *	Handles keyboard movement with a movement-vector given in world space coordinates
		 */
		virtual void handleKeyboardMovement(const glm::vec3& movementWorldSpace) override;
		/**
		 *	Handles mouse movement with a movement-vector given in x- and y-coordinates
		 *  @returns true if the corresponding MouseMovedEvent should be consumed
		 */
		virtual bool handleMouseMovement(const glm::vec2& mouseMovement) override;
		/**
		 *	Handles mouse scroll with a scroll-vector given in x- and y-coordinates
		 *  @returns true if the corresponding MouseScrolledEvent should be consumed
		 */
		virtual bool handleMouseScoll(const glm::vec2& mouseScroll) override;

		virtual void handleGamepadTriggerMovement(float leftTrigger, float rightTrigger) override;
		/**
		 *	Updates the camera's view matrix
		 */
		virtual void updateViewMatrix() = 0;
		/**
		 *	Updates the camera's projection matrix to match the new aspect ratio
		 */
		virtual void updateProjection(float aspectRatio, float width, float height) = 0;

	protected:
		glm::vec3 m_focusPoint;
	};


	// ##################################################################### //
	// ### FreeCameraController ############################################ //
	// ##################################################################### //

	/**
	 *	A Camera Controller that controls a gimbal-lock-free camera that can be oriented in any way.
	 *  Most useful in situations at which the up-direction is not important, e.g. when flying a spacecraft.
	 *  By default the camera spawns at {0,0,0} and looks in {0,0,-1}-Direction
	 */
	class FreeCameraController : public CameraController {
	protected:
		/**
		 *	Handles keyboard movement with a movement-vector given in world space coordinates
		 */
		virtual void handleKeyboardMovement(const glm::vec3& movementWorldSpace) override;
		/**
		 *	Handles mouse movement with a movement-vector given in x- and y-coordinates
		 *  @returns true if the corresponding MouseMovedEvent should be consumed
		 */
		virtual bool handleMouseMovement(const glm::vec2& mouseMovement) override;
		/**
		 *	Handles mouse scroll with a scroll-vector given in x- and y-coordinates
		 *  @returns true if the corresponding MouseScrolledEvent should be consumed
		 */
		virtual bool handleMouseScoll(const glm::vec2& mouseScroll) override;

		/**
		 *	Updates the camera's view matrix
		 */
		virtual void updateViewMatrix() = 0;
		/**
		 *	Updates the camera's projection matrix to match the new aspect ratio
		 */
		virtual void updateProjection(float aspectRatio, float width, float height) = 0;
	};


	// ##################################################################### //
	// ### GroundedCameraController ######################################## //
	// ##################################################################### //

	/**
	 *	A Camera Controller that controls a camera that can be oriented in any way while keeping an overall horizontal layout.
	 *  That means, the up-direction will always stay in the positive y-direction.
	 *  When hitting 90 degrees up or down, the camera will not turn further in these directions to avoid overrotations.
	 *  Most useful in Scenes with a well-defined up and down direction, e.g. A typical game using walking or driving.
	 *  By default the camera spawns at {0,0,0} and looks in {0,0,-1}-Direction
	 */
	class GroundedCameraController : public CameraController {
	public:
		GroundedCameraController();
		virtual ~GroundedCameraController();

		virtual void lookAt(const glm::vec3& target, const glm::vec3& upVector = glm::vec3(0.f, 1.f, 0.f)) override;

	protected:
		/**
		 *	Handles keyboard movement with a movement-vector given in world space coordinates
		 */
		virtual void handleKeyboardMovement(const glm::vec3& movementWorldSpace) override;
		/**
		 *	Handles mouse movement with a movement-vector given in x- and y-coordinates
		 *  @returns true if the corresponding MouseMovedEvent should be consumed
		 */
		virtual bool handleMouseMovement(const glm::vec2& mouseMovement) override;
		/**
		 *	Handles mouse scroll with a scroll-vector given in x- and y-coordinates
		 *  @returns true if the corresponding MouseScrolledEvent should be consumed
		 */
		virtual bool handleMouseScoll(const glm::vec2& mouseScroll) override;

		/**
		 *	Updates the camera's view matrix
		 */
		virtual void updateViewMatrix() = 0;
		/**
		 *	Updates the camera's projection matrix to match the new aspect ratio
		 */
		virtual void updateProjection(float aspectRatio, float width, float height) = 0;

		void yaw(float deltaAngle);
		void pitch(float deltaAngle);
		void updateOrientation();

	protected:
		// The pitch and yaw angles of the camera
		float m_yaw;
		float m_pitch;

		glm::vec3 m_forward;
		glm::vec3 m_right;
		glm::vec3 m_up;
	};


	// ##################################################################### //
	// ### PannableCameraController ######################################## //
	// ##################################################################### //

	/**
	 *	A Camera Controller that controls in a 2D-Fashion by panning along the x-y-plane
	 *  Most useful in situations at which the up-direction is not important, e.g. when flying a spacecraft.
	 *  By default the camera spawns at {0,0,0} and looks in {0,0,-1}-Direction
	 */
	class PannableCameraController : public CameraController {
	public:
		PannableCameraController(float zoomFactor = 1.f);

		/**
		 *	Sets the the zoom factor
		 */
		void setZoom(float zoomFactor);
	
		inline float getZoom() {
			return m_zoomFactor;
		}

	protected:
		/**
		 *	Handles keyboard movement with a movement-vector given in world space coordinates
		 */
		virtual void handleKeyboardMovement(const glm::vec3& movementWorldSpace) override;
		/**
		 *	Handles mouse movement with a movement-vector given in x- and y-coordinates
		 *  @returns true if the corresponding MouseMovedEvent should be consumed
		 */
		virtual bool handleMouseMovement(const glm::vec2& mouseMovement) override;
		/**
		 *	Handles mouse scroll with a scroll-vector given in x- and y-coordinates
		 *  @returns true if the corresponding MouseScrolledEvent should be consumed
		 */
		virtual bool handleMouseScoll(const glm::vec2& mouseScroll) override;

		/**
		 *	Updates the camera's view matrix
		 */
		virtual void updateViewMatrix() = 0;
		/**
		 *	Updates the camera's projection matrix to match the new aspect ratio
		 */
		virtual void updateProjection(float aspectRatio, float width, float height) = 0;

		virtual void updateProjectionInternal() {
			// empty by default
		};

	protected:
		float m_zoomFactor;
	};

}
