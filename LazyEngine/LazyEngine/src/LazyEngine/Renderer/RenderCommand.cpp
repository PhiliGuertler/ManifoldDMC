// ######################################################################### //
// ### RenderCommand.cpp ################################################### //
// ### Creates the rendererAPI implementation object, depending on the   ### //
// ### platform (OpenGL, Vulcan, ...).                                   ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"
#include "RenderCommand.h"

// --- OpenGL implementations --- //
#include "LazyEngine/platform/OpenGL/OpenGLRendererAPI.h"
// --- OpenGL implementations --- //

namespace LazyEngine {

	// FIXME: dynamically create the rendererapi on application start
	Scope<RendererAPI> RenderCommand::s_rendererAPI = createScope<OpenGLRendererAPI>();

}