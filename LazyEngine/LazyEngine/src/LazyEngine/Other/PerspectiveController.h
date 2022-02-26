#pragma once

#include "CameraController.h"

namespace LazyEngine {

	/**
	 *	A wrapper of a PerspectiveCamera that stores a few extra variables
	 *  and adds a few more specific methods to modify the camera.
	 */
	class PerspectiveQuirks {
	public:
		/**
		 *	outputs an Object3D, that represents the camera's position and orientation in world space
		 *  and a projection matrix.
		 */
		static std::pair<Object3D, PerspectiveQuirks> extractInfosFromCamera(const PerspectiveCamera& camera);

	public:
		/**
		 *	constructor
		 *  @param aspecRatio: The aspect ratio of the camera
		 *  @param fovY: The field of view in the vertical (y) direction. 45 degrees is reasonable.
		 *  @param near: the near plane distance
		 *  @param far: the far plane distance
		 */
		PerspectiveQuirks(float aspectRatio, float fovY, float near = 0.05f, float far = 1000.f);
		/**
		 *	destructor
		 */
		virtual ~PerspectiveQuirks();

		/**
		 *	returns the wrapped camera object
		 */
		const Camera& getCamera() const;
		
		/**
		 *	returns the camera's aspect ratio
		 */
		float getAspectRatio() const;

		/**
		 *	Updates the view matrix (which transforms the world space to the camera's model space)
		 */
		void updateViewMatrix(const glm::mat4& viewMatrix);

		/**
		 *  Updates the projection matrix
		 *  @param aspecRatio: The aspect ratio of the camera
		 *  @param fovY: The field of view in the vertical (y) direction. 45 degrees is reasonable. If this is negative, the last fovY will be used.
		 *  @param near: the near plane distance. If this is negative, the last near value will be used.
		 *  @param far: the far plane distance. If this is negative, the last far value will be used.
		 */
		void updateProjection(float aspectRatio, float fovY = -1.f, float near = -1.f, float far = -1.f);

		/**
		 *	Creates a near plane that is not perpendicular to the look direction (useful for portals/mirrors),
		 *  but instead has a near plane that is defined by a given position and normal
		 *  @param position: a point on the new near plane
		 *  @param normal: the normal of the new near plane 
		 */
		void clipOblique(const glm::vec3& position, const glm::vec3& normal);

	protected:
		PerspectiveCamera m_camera;
		float m_aspectRatio;
		float m_fovY;
		float m_near;
		float m_far;
	};

	// ##################################################################### //
	// ### PerspectiveOrbitalCameraController ############################## //
	// ##################################################################### //

	/**
	 *	A Perspective Camera Controller that orbits around a focus point using mouse panning or WASD-Movement.
	 *  The camera will always look at the given focus point and orbit around it.
	 *  By default the camera spawns at {0,0,0} and looks in {0,0,-1}-Direction, while focusing {0,0,0}
	 */
	class PerspectiveOrbitalCameraController : public OrbitalCameraController {
	public:
		/**
		 *	constructor
		 */
		PerspectiveOrbitalCameraController(float aspectRatio, float fovY, float near = 0.05f, float far = 1000.f);
		PerspectiveOrbitalCameraController(const PerspectiveCamera& camera);
		/**
		 *	returns the wrapped camera object
		 */
		virtual const Camera& getCamera() const override;

		/**
		 *	Returns the aspect ratio of the camera
		 */
		virtual float getAspectRatio() const override;

		/**
		 *	Creates a near plane that is not perpendicular to the look direction (useful for portals/mirrors),
		 *  but instead has a near plane that is defined by a given position and normal
		 *  @param position: a point on the new near plane
		 *  @param normal: the normal of the new near plane
		 */
		void clipOblique(const glm::vec3& position, const glm::vec3& normal);
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
		PerspectiveQuirks m_quirks;
	};

	// ##################################################################### //
	// ### PerspectiveFreeCameraController ################################# //
	// ##################################################################### //

	/**
	 *	A Perspective Camera Controller that controls a gimbal-lock-free camera that can be oriented in any way.
	 *  Most useful in situations at which the up-direction is not important, e.g. when flying a spacecraft.
	 *  By default the camera spawns at {0,0,0} and looks in {0,0,-1}-Direction
	 */
	class PerspectiveFreeCameraController : public FreeCameraController {
	public:
		/**
		 *	constructor
		 */
		PerspectiveFreeCameraController(float aspectRatio, float fovY, float near = 0.05f, float far = 1000.f);
		PerspectiveFreeCameraController(const PerspectiveCamera& camera);

		/**
		 *	returns the wrapped camera object
		 */
		const Camera& getCamera() const override;

		/**
		 *	Returns the aspect ratio of the camera
		 */
		virtual float getAspectRatio() const override;

		/**
		 *	Creates a near plane that is not perpendicular to the look direction (useful for portals/mirrors),
		 *  but instead has a near plane that is defined by a given position and normal
		 *  @param position: a point on the new near plane
		 *  @param normal: the normal of the new near plane
		 */
		void clipOblique(const glm::vec3& position, const glm::vec3& normal);
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
		PerspectiveQuirks m_quirks;
	};


	// ##################################################################### //
	// ### PerspectiveGroundedCameraController ############################# //
	// ##################################################################### //

	/**
	 *	A Perspective Camera Controller that controls a camera that can be oriented in any way while keeping an overall horizontal layout.
	 *  That means, the up-direction will always stay in the positive y-direction.
	 *  When hitting 90 degrees up or down, the camera will not turn further in these directions to avoid overrotations.
	 *  Most useful in Scenes with a well-defined up and down direction, e.g. A typical game using walking or driving.
	 *  By default the camera spawns at {0,0,0} and looks in {0,0,-1}-Direction
	 */
	class PerspectiveGroundedCameraController : public GroundedCameraController {
	public:
		/**
		 *	constructor
		 */
		PerspectiveGroundedCameraController(float aspectRatio, float fovY, float near = 0.05f, float far = 1000.f);
		PerspectiveGroundedCameraController(const PerspectiveCamera& camera);

		/**
		 *	returns the wrapped camera object
		 */
		virtual const Camera& getCamera() const override;

		/**
		 *	Returns the aspect ratio of the camera
		 */
		virtual float getAspectRatio() const override;

		/**
		 *	Creates a near plane that is not perpendicular to the look direction (useful for portals/mirrors),
		 *  but instead has a near plane that is defined by a given position and normal
		 *  @param position: a point on the new near plane
		 *  @param normal: the normal of the new near plane
		 */
		void clipOblique(const glm::vec3& position, const glm::vec3& normal);
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
		PerspectiveQuirks m_quirks;
	};

	typedef PerspectiveFreeCameraController PerspectiveCameraController;


	

}
