#include "LazyEngine/gepch.h"

#include "Renderer.h"
#include "RendererImpl.h"

namespace LazyEngine {

	void Renderer::beginScene(const Camera& camera) {
		RendererImpl::getInstance().beginScene(camera);
	}

	void Renderer::endScene() {
		RendererImpl::getInstance().endScene();
	}

	void Renderer::submit(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform) {
		RendererImpl::getInstance().submit(vertexArray, shader, modelTransform);
	}

	void Renderer::submitWireframe(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform) {
		RendererImpl::getInstance().submitWireframe(vertexArray, shader, modelTransform);
	}

	void Renderer::submitPoints(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform) {
		RendererImpl::getInstance().submitPoints(vertexArray, shader, modelTransform);
	}

	void Renderer::submitLines(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform) {
		RendererImpl::getInstance().submitLines(vertexArray, shader, modelTransform);
	}


	/*
	void Renderer::submitSinglePoints(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const std::vector<uint32_t> vertexIndices, const glm::mat4& modelTransform) {
		RendererImpl::getInstance().submitSinglePoints(vertexArray, shader, vertexIndices, modelTransform);
	}
	*/

	RendererAPI::API Renderer::getAPI() {
		return RendererImpl::getAPI();
	}

}