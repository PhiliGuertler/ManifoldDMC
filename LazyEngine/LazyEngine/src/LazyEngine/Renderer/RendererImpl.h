#pragma once

// ######################################################################### //
// ### Renderer.h ########################################################## //
// ### A Representation of general Renderer functionality like Clearing  ### //
// ### and the RendererAPI that is currently in use like OpenGL, Vulcan, ### //
// ### etc.                                                              ### //
// ######################################################################### //

#include "RendererAPI.h"

#include "Camera.h"
#include "Shader.h"
#include "../Core/Core.h"

namespace LazyEngine {

	/**
	 *	A general purpose 3D Renderer implemented as a singleton
	 */
	class RendererImpl {
	public:
		/**
		 *	returns the singleton instance of this class
		 */
		static RendererImpl& getInstance();

		/**
		 *	returns the API used by RendererAPI
		 */
		inline static RendererAPI::API getAPI() { return RendererAPI::getAPI(); }

	public:
		/**
		 *	default destructor
		 */
		~RendererImpl() = default;

		/**
		 *	begins a scene using a single camera
		 *	@param camera: the camera to be used in the scene
		 */
		void beginScene(const Camera& camera);
		/**
		 *	Ends the scene.
		 *	FIXME: Currently nothing happens here.
		 */
		void endScene();
		/**
		 *	Renders a vertex array wíth a given shader and model matrix.
		 *	This must be called after beginScene.
		 *	@param vertexArray: the vertex array to be rendered
		 *	@param shader: the shader to be used to render the vertex array
		 *	@param modelTransform: the model matrix of the object
		 */
		void submit(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform = glm::mat4(1.f));
		/**
		 *	Renders a vertex array wíth a given shader and model matrix as a wireframe.
		 *	This must be called after beginScene.
		 *	@param vertexArray: the vertex array to be rendered
		 *	@param shader: the shader to be used to render the vertex array
		 *	@param modelTransform: the model matrix of the object
		 */
		void submitWireframe(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform = glm::mat4(1.f));
		/**
		 *	Renders a vertex array's vertices as points.
		 *  This must be called after beginScene.
		 *  @param vertexArray: the vertex array to be rendered
		 *  @param shader: the shader to be used to render the vertex array
		 *  @param modelTransform: the model matrix of the object
		 */
		void submitPoints(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform = glm::mat4(1.f));
		/**
		 *	Renders a vertex array's vertices as lines.
		 *  This must be called after beginScene.
		 *  @param vertexArray: the vertex array to be rendered
		 *  @param shader: the shader to be used to render the vertex array
		 *  @param modelTransform: the model matrix of the object
		 */
		void submitLines(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform = glm::mat4(1.f));
		/**
		 *	Renders a subset of a vertex array's vertices as points.
		 *	This must be called after beginScene.
		 *	The regular index-buffer will not be altered.
		 *	@param vertexArray: the vertex array of which a subset should be rendered
		 *	@param shader: the shader to be used to render the vertex array
		 *	@param vertexIndices: the indices of the single vertices that should be rendered
		 *	@param modelTransform: the model matrix of the object
		void submitSinglePoints(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const std::vector<uint32_t> vertexIndices, const glm::mat4& modelTransform = glm::mat4(1.f));
		 */
		/**
		 *	updates the viewport size.
		 *	@param width: width of the new viewport size
		 *	@param height: height of the new viewport size
		 */
		void onWindowResize(uint32_t width, uint32_t height);

	private:
		/**
		 *	private constructor to ensure the singleton nature of this class
		 */
		RendererImpl();

		/**
		 *	delete the copy and move constructors to prevent the user from accidentally copying the singleton instance
		 */
		RendererImpl(const RendererImpl& copy) = delete;
		RendererImpl(const RendererImpl&& move) = delete;

		/**
		 *	delete copy and move operators= to prevent the user from accidentally copying the singleton instance
		 */
		RendererImpl& operator=(const RendererImpl& copy) = delete;
		RendererImpl& operator=(RendererImpl&& move) = delete;

		/**
		 *	Updates the camera uniforms u_projView, u_worldToView, u_viewToProjection, u_normalMatrix and u_model in the given shader.
		 *  @param shader: The shader whose uniforms should be updated
		 *  @param modelTransform: The transform of the model that will be rendered using the shader
		 */
		void updateShaderCameraUniforms(const Ref<Shader>& shader, const glm::mat4& modelTransform);

		// the cached camera matrix
		glm::mat4 m_projViewMatrix;
		glm::mat4 m_worldToView;
		glm::mat4 m_viewToProjection;
		glm::vec3 m_cameraPosition;
		float m_near;
		float m_far;
	};
}