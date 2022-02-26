#pragma once

// ######################################################################### //
// ### Framebuffer.h ####################################################### //
// ### Defines a non-RenderingAPI specific Superclass for Framebuffers   ### //
// ######################################################################### //

#include "LazyEngine/Core/Core.h"
#include "Texture.h"

namespace LazyEngine {

	// TODO: add the option to choose what kind of textures the framebuffer will use
	class Framebuffer {
	public:
		// TODO: create a framebuffer with a given specification instead of just width and height
		struct Specification {
			// the width of the framebuffer (in pixels)
			uint32_t width;
			// the height of the framebuffer (in pixels)
			uint32_t height;

			// this Flag is true, if the render target is the screen (which corresponds to glBindFramebuffer(0)
			bool isSwapChainTarget = false;

			// The number of color attachments for this framebuffer
			uint32_t numColorAttachments = 1;
			// Enables or disables a depth and stencil attachment
			bool enableDepthStencilAttachment = true;
		};

		// FIXME: this might be obsolete, if the renderer takes a framebuffer as source and target (which is not yet implemented)
		enum class FramebufferUsage {
			Read,
			Draw,
			ReadAndDraw
		};

		/**
		 *	factory method for creating a Framebuffer
		 */
		static Ref<Framebuffer> create(uint32_t width, uint32_t height);

	public:
		/**
		 *	default destructor
		 */
		virtual ~Framebuffer() = default;

		/**
		 *	binds the framebuffer as read, draw or read and draw target.
		 *	All draw-calls will now render into this framebuffer until it is unbound or a different framebuffer is bound.
		 *	@param usage: the usage of this buffer. By default this is ReadAndDraw
		 */
		virtual void bind(FramebufferUsage usage = FramebufferUsage::ReadAndDraw) = 0;
		/**
		 *	unbinds the framebuffer
		 */
		virtual void unbind() = 0;

		/**
		 *	returns the color texture into which this framebuffer is rendering
		 */
		virtual Ref<Texture2D> getColorTexture() const = 0;

		/**
		 *	returns the depth-stencil texture into which this framebuffer is rendering
		 */
		virtual Ref<Texture2D> getDepthStencil() const = 0;

		/**
		 *	returns the size of the framebuffer in width and height
		 */
		virtual const glm::ivec2 getSize() const = 0;

		/**
		 *	resizes the attachments of this framebuffer
		 */
		virtual void resize(uint32_t width, uint32_t height) = 0;
		/**
		 *	resizes the attachments of this framebuffer
		 */
		virtual inline void resize(const glm::ivec2& size) { resize(static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)); }
	};

}