// ######################################################################### //
// ### Renderer.cpp ######################################################## //
// ### Implements Renderer.h ############################################### //
// ######################################################################### //

#include "LazyEngine/gepch.h"

#include "RendererImpl.h"

#include <glm/gtc/type_ptr.hpp>

#include "RenderCommand.h"
#include "../Profiling/Profiler.h"

namespace LazyEngine {

	RendererImpl& RendererImpl::getInstance() {
		static RendererImpl instance;
		return instance;
	}

	RendererImpl::RendererImpl()
		: m_projViewMatrix(glm::mat4(1.f))
		, m_worldToView(glm::mat4(1.f))
		, m_viewToProjection(glm::mat4(1.f))
		, m_cameraPosition(glm::vec3(0.f))
		, m_near(0.05)
		, m_far(1000.f)
	{
		// empty
	}

	void RendererImpl::beginScene(const Camera& camera) {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();

		// FIXME: do things in the future
		m_projViewMatrix = camera.getProjectionViewMatrix();
		m_worldToView = camera.getViewMatrix();
		m_viewToProjection = camera.getProjectionMatrix();
		m_cameraPosition = glm::vec3(camera.getViewMatrix() * glm::vec4(0.f, 0.f, 0.f, 1.f));
		m_near = m_viewToProjection[3][2] / (m_viewToProjection[2][2] - 1.f);
		m_far = m_viewToProjection[3][2] / (m_viewToProjection[2][2] + 1.f);
	}

	void RendererImpl::endScene() {
		// FIXME: do things in the future
	}

	void RendererImpl::submit(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform) {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();

		vertexArray->bind();
		shader->bind();

		updateShaderCameraUniforms(shader, modelTransform);

		RenderCommand::drawIndexed(vertexArray);
	}

	void RendererImpl::submitWireframe(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform) {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();

		vertexArray->bind();
		shader->bind();

		updateShaderCameraUniforms(shader, modelTransform);

		RenderCommand::drawIndexedWireframe(vertexArray);
	}

	void RendererImpl::submitPoints(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform) {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();

		vertexArray->bind();
		shader->bind();

		updateShaderCameraUniforms(shader, modelTransform);

		RenderCommand::drawIndexedPoints(vertexArray);
	}

	void RendererImpl::submitLines(const Ref<VertexArray>& vertexArray, const Ref<Shader>& shader, const glm::mat4& modelTransform) {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();

		vertexArray->bind();
		shader->bind();

		updateShaderCameraUniforms(shader, modelTransform);

		RenderCommand::drawIndexedLines(vertexArray);
	}


	void RendererImpl::onWindowResize(uint32_t width, uint32_t height) {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();

		RenderCommand::setViewport(0, 0, width, height);
	}

	void RendererImpl::updateShaderCameraUniforms(const Ref<Shader>& shader, const glm::mat4& modelTransform) {
		// FIXME: this does not need to be updated every time if a global state is being saved
		shader->uniformMat4("u_projView", m_projViewMatrix);
		shader->uniformMat4("u_worldToView", m_worldToView);
		shader->uniformMat4("u_viewToProjection", m_viewToProjection);
		shader->uniformFloat("u_near", m_near);
		shader->uniformFloat("u_far", m_far);

		// compute the normal matrix, which is the transpose inverse of the modelView.
		glm::mat4 modelView = m_worldToView * modelTransform;
		glm::mat3 normalMatrix = glm::mat3(modelView);
		normalMatrix = glm::transpose(glm::inverse(normalMatrix));
		shader->uniformMat3("u_normalMatrix", normalMatrix);
		// :EMXIF

		shader->uniformMat4("u_model", modelTransform);
	}


}