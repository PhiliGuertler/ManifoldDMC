// ######################################################################### //
// ### OpenGLRendererAPI.cpp ############################################### //
// ### implements OpenGLRendererAPI.h                                    ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"
#include "OpenGLRendererAPI.h"

#include <glad/glad.h>

#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	void OpenGLRendererAPI::init() {
		LAZYENGINE_PROFILE_FUNCTION();

		// enable Multisampling
		glEnable(GL_MULTISAMPLE);

		// enable alpha blending
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		// enable depth testing
		glEnable(GL_DEPTH_TEST);

		// TODO: further initialization code here
	}

	void OpenGLRendererAPI::setClearColor(const glm::vec4& color) {
		LAZYENGINE_PROFILE_FUNCTION();

		glClearColor(color.r, color.g, color.b, color.a);
	}

	void OpenGLRendererAPI::clear() {
		LAZYENGINE_PROFILE_FUNCTION();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	void OpenGLRendererAPI::clearDepth() {
		LAZYENGINE_PROFILE_FUNCTION();

		glClear(GL_DEPTH_BUFFER_BIT);
	}


	void OpenGLRendererAPI::drawIndexed(const Ref<VertexArray>& vertexArray) {
		LAZYENGINE_PROFILE_FUNCTION();

		glDrawElements(GL_TRIANGLES, vertexArray->getIndexBuffer()->getCount(), GL_UNSIGNED_INT, nullptr);
	}

	void OpenGLRendererAPI::drawIndexedWireframe(const Ref<VertexArray>& vertexArray) {
		LAZYENGINE_PROFILE_FUNCTION();

		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDrawElements(GL_TRIANGLES, vertexArray->getIndexBuffer()->getCount(), GL_UNSIGNED_INT, nullptr);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}

	void OpenGLRendererAPI::drawIndexedPoints(const Ref<VertexArray>& vertexArray) {
		LAZYENGINE_PROFILE_FUNCTION();
		glDrawElements(GL_POINTS, vertexArray->getIndexBuffer()->getCount(), GL_UNSIGNED_INT, nullptr);
	}

	void OpenGLRendererAPI::drawIndexedLines(const Ref<VertexArray>& vertexArray) {
		LAZYENGINE_PROFILE_FUNCTION();
		glDrawElements(GL_LINES, vertexArray->getIndexBuffer()->getCount(), GL_UNSIGNED_INT, nullptr);
	}

	void OpenGLRendererAPI::drawQuadsIndexed(const Ref<VertexArray>& vertexArray, uint32_t elementCount) {
		LAZYENGINE_PROFILE_FUNCTION();

		glDrawElements(GL_TRIANGLES, elementCount*6, GL_UNSIGNED_INT, nullptr);
	}

	void OpenGLRendererAPI::setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
		LAZYENGINE_PROFILE_FUNCTION();

		glViewport(x, y, width, height);
	}

	void OpenGLRendererAPI::enableDepthTest() {
		LAZYENGINE_PROFILE_FUNCTION();

		glEnable(GL_DEPTH_TEST);
	}

	void OpenGLRendererAPI::disableDepthTest() {
		LAZYENGINE_PROFILE_FUNCTION();

		glDisable(GL_DEPTH_TEST);
	}

	void OpenGLRendererAPI::enableAlphaBlending() {
		LAZYENGINE_PROFILE_FUNCTION();

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

	void OpenGLRendererAPI::disableAlphaBlending() {
		LAZYENGINE_PROFILE_FUNCTION();

		glDisable(GL_BLEND);
	}

	void OpenGLRendererAPI::setLineWidth(float width) {
		LAZYENGINE_PROFILE_FUNCTION();

		glLineWidth(width);
	}

	void OpenGLRendererAPI::setPointSize(float radius) {
		LAZYENGINE_PROFILE_FUNCTION();
		
		glPointSize(radius);
	}

}