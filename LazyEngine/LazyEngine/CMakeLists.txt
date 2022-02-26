# add the dependencies of LazyEngine
add_subdirectory("vendor/glad")
add_subdirectory("vendor/glfw")
add_subdirectory("vendor/glm")
add_subdirectory("vendor/imgui")
add_subdirectory("vendor/spdlog-1.x")
add_subdirectory("vendor/stb_image")
set_target_properties(Glad PROPERTIES FOLDER LazyEngine/Dependencies)
set_target_properties(GLFW PROPERTIES FOLDER LazyEngine/Dependencies)
set_target_properties(ImGui PROPERTIES FOLDER LazyEngine/Dependencies)
set_target_properties(stb_image PROPERTIES FOLDER LazyEngine/Dependencies)

# set include directories
set(LazyEngine_INCLUDE_DIRS
	"${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# set library sources
set(LazyEngine_SRC
	"src/LazyEngine/Core/Input/Gamepad.cpp"
	"src/LazyEngine/Core/Input/Gamepad.h"
	"src/LazyEngine/Core/Input/GamepadCodes.h"
	"src/LazyEngine/Core/Input/Input.cpp"
	"src/LazyEngine/Core/Input/Input.h"
	"src/LazyEngine/Core/Input/KeyCodes.h"
	"src/LazyEngine/Core/Input/MouseButtonCodes.h"
	"src/LazyEngine/Core/Application.cpp"
	"src/LazyEngine/Core/Application.h"
	"src/LazyEngine/Core/Constants.h"
	"src/LazyEngine/Core/Core.h"
	"src/LazyEngine/Core/EntryPoint.h"
	"src/LazyEngine/Core/Layer.h"
	"src/LazyEngine/Core/LayerStack.cpp"
	"src/LazyEngine/Core/LayerStack.h"
	"src/LazyEngine/Core/Log.cpp"
	"src/LazyEngine/Core/Log.h"
	"src/LazyEngine/Core/Time.cpp"
	"src/LazyEngine/Core/Time.h"
	"src/LazyEngine/Core/Util.h"
	"src/LazyEngine/Core/Window.cpp"
	"src/LazyEngine/Core/Window.h"

	"src/LazyEngine/Events/ApplicationEvent.h"
	"src/LazyEngine/Events/Event.h"
	"src/LazyEngine/Events/GamepadEvent.h"
	"src/LazyEngine/Events/KeyEvent.h"
	"src/LazyEngine/Events/MouseEvent.h"

	"src/LazyEngine/ImGui/ImGuiBuild.cpp"
	"src/LazyEngine/ImGui/ImGuiLayer.cpp"
	"src/LazyEngine/ImGui/ImGuiLayer.h"
	"src/LazyEngine/ImGui/ImGuiLayer.inl"
	"src/LazyEngine/ImGui/ImGuiViewport.cpp"
	"src/LazyEngine/ImGui/ImGuiViewport.h"

	"src/LazyEngine/Other/CameraController.cpp"
	"src/LazyEngine/Other/CameraController.h"
	"src/LazyEngine/Other/PerspectiveController.h"
	"src/LazyEngine/Other/PerspectiveController.cpp"
	"src/LazyEngine/Other/OrthographicController.h"
	"src/LazyEngine/Other/OrthographicController.cpp"

	"src/LazyEngine/Profiling/Profiler.cpp"
	"src/LazyEngine/Profiling/Profiler.h"
	"src/LazyEngine/Profiling/Timer.cpp"
	"src/LazyEngine/Profiling/Timer.h"

	"src/LazyEngine/Renderer/Geometry/GeometricPrimitives.cpp"
	"src/LazyEngine/Renderer/Geometry/GeometricPrimitives.h"
	"src/LazyEngine/Renderer/Geometry/Object3D.cpp"
	"src/LazyEngine/Renderer/Geometry/Object3D.h"

	"src/LazyEngine/Renderer/Materials/Material.h"
	"src/LazyEngine/Renderer/Materials/Colors.h"
	"src/LazyEngine/Renderer/Materials/PhongMaterial.cpp"
	"src/LazyEngine/Renderer/Materials/PhongMaterial.h"
	"src/LazyEngine/Renderer/Materials/BillboardMaterial.h"
	"src/LazyEngine/Renderer/Materials/BillboardMaterial.cpp"

	"src/LazyEngine/Renderer/Buffer.cpp"
	"src/LazyEngine/Renderer/Buffer.h"
	"src/LazyEngine/Renderer/Camera.cpp"
	"src/LazyEngine/Renderer/Camera.h"
	"src/LazyEngine/Renderer/Framebuffer.cpp"
	"src/LazyEngine/Renderer/Framebuffer.h"
	"src/LazyEngine/Renderer/GraphicsContext.h"
	"src/LazyEngine/Renderer/RenderCommand.cpp"
	"src/LazyEngine/Renderer/RenderCommand.h"
	"src/LazyEngine/Renderer/Renderer.cpp"
	"src/LazyEngine/Renderer/Renderer.h"
	"src/LazyEngine/Renderer/RendererImpl.cpp"
	"src/LazyEngine/Renderer/RendererImpl.h"
	"src/LazyEngine/Renderer/Renderer2D.cpp"
	"src/LazyEngine/Renderer/Renderer2D.h"
	"src/LazyEngine/Renderer/Renderer2DImpl.cpp"
	"src/LazyEngine/Renderer/Renderer2DImpl.h"
	"src/LazyEngine/Renderer/RendererAPI.cpp"
	"src/LazyEngine/Renderer/RendererAPI.h"
	"src/LazyEngine/Renderer/Shader.cpp"
	"src/LazyEngine/Renderer/Shader.h"
	"src/LazyEngine/Renderer/Texture.cpp"
	"src/LazyEngine/Renderer/Texture.h"
	"src/LazyEngine/Renderer/TextureAtlas.cpp"
	"src/LazyEngine/Renderer/TextureAtlas.h"
	"src/LazyEngine/Renderer/VertexArray.cpp"
	"src/LazyEngine/Renderer/VertexArray.h"

	"src/LazyEngine/platform/CUDA/CUDAGLInteroperabilty.h"
	"src/LazyEngine/platform/CUDA/CUDAGLInteroperabilty.cu"
	"src/LazyEngine/platform/CUDA/CUDAUtils.h"
	"src/LazyEngine/platform/CUDA/CUDAUtils.cu"

	"src/LazyEngine/platform/GLFW/GLFWGamepad.cpp"
	"src/LazyEngine/platform/GLFW/GLFWGamepad.h"
	"src/LazyEngine/platform/GLFW/GLFWInput.cpp"
	"src/LazyEngine/platform/GLFW/GLFWInput.h"
	"src/LazyEngine/platform/GLFW/GLFWWindow.cpp"
	"src/LazyEngine/platform/GLFW/GLFWWindow.h"

	"src/LazyEngine/platform/OpenGL/OpenGLBuffer.cpp"
	"src/LazyEngine/platform/OpenGL/OpenGLBuffer.h"
	"src/LazyEngine/platform/OpenGL/OpenGLContext.cpp"
	"src/LazyEngine/platform/OpenGL/OpenGLContext.h"
	"src/LazyEngine/platform/OpenGL/OpenGLFramebuffer.cpp"
	"src/LazyEngine/platform/OpenGL/OpenGLFramebuffer.h"
	"src/LazyEngine/platform/OpenGL/OpenGLRendererAPI.cpp"
	"src/LazyEngine/platform/OpenGL/OpenGLRendererAPI.h"
	"src/LazyEngine/platform/OpenGL/OpenGLShader.cpp"
	"src/LazyEngine/platform/OpenGL/OpenGLShader.h"
	"src/LazyEngine/platform/OpenGL/OpenGLTexture.cpp"
	"src/LazyEngine/platform/OpenGL/OpenGLTexture.h"
	"src/LazyEngine/platform/OpenGL/OpenGLVertexArray.cpp"
	"src/LazyEngine/platform/OpenGL/OpenGLVertexArray.h"
	
	"src/LazyEngine/platform/Windows/WindowsGamepad.cpp"
	"src/LazyEngine/platform/Windows/WindowsGamepad.h"
	"src/LazyEngine/platform/Windows/WindowsInput.cpp"
	"src/LazyEngine/platform/Windows/WindowsInput.h"
	"src/LazyEngine/platform/Windows/WindowsWindow.cpp"
	"src/LazyEngine/platform/Windows/WindowsWindow.h"
	
	"src/LazyEngine/gepch.cpp"
	"src/LazyEngine/gepch.h"
	"src/LazyEngine/LazyEngine.h"
)

add_library(LazyEngine ${LazyEngine_SRC})

target_link_libraries(LazyEngine
	GLFW
	Glad
	ImGui
	spdlog
	glm
	stb_image
)

target_compile_definitions(LazyEngine PUBLIC
	"LAZYENGINE_ENABLE_IMGUI"
	"LAZYENGINE_BUILD_DLL"
	"LAZYENGINE_CUDA"
	"GLFW_INCLUDE_NONE"
)

target_include_directories(LazyEngine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

# group the source files in the same way in Visual Studio as on disk
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${LazyEngine_SRC})