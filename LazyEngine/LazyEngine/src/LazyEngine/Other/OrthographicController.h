#pragma once

#include "CameraController.h"

namespace LazyEngine {

	// TODO: These Controllers are all intrinsically like their Perspective Counterparts.
	// A Controller using Just Panning to display a 2D-Scene and Zooming is missing
	// Also TODO: Update the CameraControllers' Usages in all test projects

	// ##################################################################### //
	// ### OrthographicOrbitalCameraController ############################# //
	// ##################################################################### //

	/**
	 *	An Orthographic Camera Controller that orbits around a focus point using mouse panning or WASD-Movement.
	 *  The camera will always look at the given focus point and orbit around it.
	 *  By default the camera spawns at {0,0,0} and looks in {0,0,-1}-Direction, while focusing {0,0,0}
	 */
	class OrthographicOrbitalCameraController : public OrbitalCameraController {
	public:
		/**
		 *	constructor
		 */
		OrthographicOrbitalCameraController(float left, float right, float bottom, float top);

		/**
		 *	returns the wrapped camera object
		 */
		virtual const Camera& getCamera() const override;

		/**
		 *	Returns the aspect ratio of the camera
		 */
		virtual float getAspectRatio() const override;
	protected:
		/**
		 *	Updates the camera's view matrix
		 */
		virtual void updateViewMatrix() override;

		/**
		 *	Updates the camera's projection matrix to match the new aspect ratio
		 */
		virtual void updateProjection(float aspectRatio, float width, float height) override;

	private:
		OrthographicCamera m_camera;
		float m_left;
		float m_right;
		float m_bottom;
		float m_top;
	};

	// ##################################################################### //
	// ### OrthographicFreeCameraController ################################ //
	// ##################################################################### //

	/**
	 *	An Orthographic Camera Controller that controls a gimbal-lock-free camera that can be oriented in any way.
	 *  Most useful in situations at which the up-direction is not important, e.g. when flying a spacecraft.
	 *  By default the camera spawns at {0,0,0} and looks in {0,0,-1}-Direction
	 */
	class OrthographicFreeCameraController : public FreeCameraController {
	public:
		/**
		 *	constructor
		 */
		OrthographicFreeCameraController(float left, float right, float bottom, float top);

		/**
		 *	returns the wrapped camera object
		 */
		const Camera& getCamera() const override;

		/**
		 *	Returns the aspect ratio of the camera
		 */
		virtual float getAspectRatio() const override;
	protected:
		/**
		 *	Updates the camera's view matrix
		 */
		virtual void updateViewMatrix() override;

		/**
		 *	Updates the camera's projection matrix to match the new aspect ratio
		 */
		virtual void updateProjection(float aspectRatio, float width, float height) override;

	private:
		OrthographicCamera m_camera;
		float m_left;
		float m_right;
		float m_bottom;
		float m_top;
	};


	// ##################################################################### //
	// ### OrthographicGroundedCameraController ############################ //
	// ##################################################################### //

	/**
	 *	An Orthographic Camera Controller that controls a camera that can be oriented in any way while keeping an overall horizontal layout.
	 *  That means, the up-direction will always stay in the positive y-direction.
	 *  When hitting 90 degrees up or down, the camera will not turn further in these directions to avoid overrotations.
	 *  Most useful in Scenes with a well-defined up and down direction, e.g. A typical game using walking or driving.
	 *  By default the camera spawns at {0,0,0} and looks in {0,0,-1}-Direction
	 */
	class OrthographicGroundedCameraController : public GroundedCameraController {
	public:
		/**
		 *	constructor
		 */
		OrthographicGroundedCameraController(float left, float right, float bottom, float top);

		/**
		 *	returns the wrapped camera object
		 */
		virtual const Camera& getCamera() const override;

		/**
		 *	Returns the aspect ratio of the camera
		 */
		virtual float getAspectRatio() const override;
	protected:
		/**
		 *	Updates the camera's view matrix
		 */
		virtual void updateViewMatrix() override;

		/**
		 *	Updates the camera's projection matrix to match the new aspect ratio
		 */
		virtual void updateProjection(float aspectRatio, float width, float height) override;

	private:
		OrthographicCamera m_camera;
		float m_left;
		float m_right;
		float m_bottom;
		float m_top;
	};


	// ##################################################################### //
	// ### Orthographic2DCameraController ################################## //
	// ##################################################################### //

	class Orthographic2DCameraController : public PannableCameraController {
	public:
		/**
		 *	constructor
		 *  @param left: The left plane of the orthographic camera
		 *  @param right: The right plane of the orthographic camera
		 *  @param bottom: The bottom plane of the orthographic camera
		 *  @param top: The top plane of the orthographic camera
		 */
		Orthographic2DCameraController(float left, float right, float bottom, float top);

		/**
		 *	constructor
		 *  @param aspectRatio: The aspect-ratio of the orthographic camera
		 */
		Orthographic2DCameraController(float aspectRatio);

		/**
		 *	returns the wrapped camera object
		 */
		virtual const Camera& getCamera() const override;

		/**
		 *	Returns the aspect ratio of the camera
		 */
		virtual float getAspectRatio() const override;
	protected:
		/**
		 *	Updates the camera's view matrix
		 */
		virtual void updateViewMatrix() override;

		/**
		 *	Updates the camera's projection matrix to match the new aspect ratio
		 */
		virtual void updateProjection(float aspectRatio, float width, float height) override;

		virtual void updateProjectionInternal() override;
	private:
		OrthographicCamera m_camera;
		float m_left;
		float m_right;
		float m_bottom;
		float m_top;
	};

	typedef Orthographic2DCameraController OrthographicCameraController;
}
