#pragma once

// ######################################################################### //
// ### Renderer.h ########################################################## //
// ### A Representation of general Renderer functionality like Clearing  ### //
// ### and the RendererAPI that is currently in use like OpenGL, Vulcan, ### //
// ### etc.                                                              ### //
// ######################################################################### //

#include "Camera.h"
#include "RendererAPI.h"
#include "Shader.h"

namespace LazyEngine {

	class Renderer {
	public:

		/**
		 *	begins a scene using a single camera
		 *	@param camera: the camera to be used in the scene
		 */
		static void beginScene(const Camera& camera);

		/**
		 *	Ends the scene.
		 *	FIXME: Currently nothing happens here.
		 */
		static void endScene();

		/**
		 *	Renders a vertex array wíth a given shader and model matrix.
		 *	This must be called after beginScene.
		 *	@param vertexArray: the vertex array to be rendered
		 *	@param shader: the shader to be used to render the vertex array
		 *	@param modelTransform: the model matrix of the object
		 */
		static void submit(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform = glm::mat4(1.f));

		/**
		 *	Renders a vertex array wíth a given shader and model matrix as a wireframe.
		 *	This must be called after beginScene.
		 *	@param vertexArray: the vertex array to be rendered
		 *	@param shader: the shader to be used to render the vertex array
		 *	@param modelTransform: the model matrix of the object
		 */
		static void submitWireframe(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform = glm::mat4(1.f));

		/**
		 *	Renders a vertex array's vertices as points.
		 *  This must be called after beginScene.
		 *  @param vertexArray: the vertex array to be rendered
		 *  @param shader: the shader to be used to render the vertex array
		 *  @param modelTransform: the model matrix of the object
		 */
		static void submitPoints(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform = glm::mat4(1.f));

		/**
		 *	Renders a vertex array's vertices as lines.
		 *  This must be called after beginScene.
		 *  @param vertexArray: the vertex array to be rendered
		 *  @param shader: the shader to be used to render the vertex array
		 *  @param modelTransform: the model matrix of the object
		 */
		static void submitLines(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform = glm::mat4(1.f));

		/**
		 *	Renders a subset of a vertex array's vertices as points.
		 *	This must be called after beginScene.
		 *	The regular index-buffer will not be altered.
		 *	@param vertexArray: the vertex array of which a subset should be rendered
		 *	@param shader: the shader to be used to render the vertex array
		 *	@param vertexIndices: the indices of the single vertices that should be rendered
		 *	@param modelTransform: the model matrix of the object
		static void submitSinglePoints(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const std::vector<uint32_t> vertexIndices, const glm::mat4& modelTransform = glm::mat4(1.f));
		 */

		/**
		 *	returns the API used by RendererAPI
		 */
		static RendererAPI::API getAPI();

	};

}