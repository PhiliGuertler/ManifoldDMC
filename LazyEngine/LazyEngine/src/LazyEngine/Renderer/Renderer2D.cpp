#include "LazyEngine/gepch.h"

#include "Renderer2D.h"
#include "Renderer2DImpl.h"

namespace LazyEngine {

	void Renderer2D::beginScene(const Camera& camera) {
		Renderer2DImpl::getInstance().beginScene(camera);
	}

	void Renderer2D::endScene() {
		Renderer2DImpl::getInstance().endScene();
	}

	void Renderer2D::flush() {
		Renderer2DImpl::getInstance().flush();
	}

	void Renderer2D::drawQuad(const Quad& quad) {
		Renderer2DImpl::getInstance().drawQuad(quad);
	}
	
	void Renderer2D::drawScreenSizedQuad(const Ref<Texture2D>& texture, const glm::vec4& colorTint) {
		Renderer2DImpl::getInstance().drawScreenSizedQuad(texture, colorTint);
	}

	// ### Statistics ######################################################### //

	void Renderer2D::resetStatistics() {
		Renderer2DImpl::getInstance().resetStatistics();
	}

	Renderer2D::Statistics Renderer2D::getStatistics() {
		return Renderer2DImpl::getInstance().getStatistics();
	}
}