#pragma once

// ######################################################################### //
// ### LazyEngine.h ######################################################## //
// ### This File is to be included in a client project to gain the       ### //
// ### engine's functionality.                                           ### //
// ######################################################################### //

// For use by LazyEngine applications

// Library Includes
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


// Core Includes
#include "LazyEngine/Core/Application.h"
#include "LazyEngine/Core/Layer.h"
#include "LazyEngine/Core/Log.h"
#include "LazyEngine/Core/Util.h"
#include "LazyEngine/Core/Constants.h"
#include "LazyEngine/Core/Time.h"


// Input related Includes
#include "LazyEngine/Core/Input/Gamepad.h"
#include "LazyEngine/Core/Input/GamepadCodes.h"
#include "LazyEngine/Core/Input/Input.h"
#include "LazyEngine/Core/Input/KeyCodes.h"
#include "LazyEngine/Core/Input/MouseButtonCodes.h"

// Predefined Imgui-Layer
#include "LazyEngine/ImGui/ImGuiLayer.h"

// LazyEngine Graphics Abstractions
#include "LazyEngine/Renderer/Buffer.h"
#include "LazyEngine/Renderer/Camera.h"
#include "LazyEngine/Renderer/Framebuffer.h"
#include "LazyEngine/Renderer/Renderer.h"
#include "LazyEngine/Renderer/Renderer2D.h"
#include "LazyEngine/Renderer/RenderCommand.h"
#include "LazyEngine/Renderer/RendererAPI.h"
#include "LazyEngine/Renderer/Shader.h"
#include "LazyEngine/Renderer/Texture.h"
#include "LazyEngine/Renderer/TextureAtlas.h"
#include "LazyEngine/Renderer/VertexArray.h"

// Geometry
#include "LazyEngine/Renderer/Geometry/GeometricPrimitives.h"

// Materials
#include "LazyEngine/Renderer/Materials/Material.h"
#include "LazyEngine/Renderer/Materials/Colors.h"
#include "LazyEngine/Renderer/Materials/PhongMaterial.h"
#include "LazyEngine/Renderer/Materials/BillboardMaterial.h"

// LazyEngine Cameras
#include "LazyEngine/Other/CameraController.h"
#include "LazyEngine/Other/PerspectiveController.h"
#include "LazyEngine/Other/OrthographicController.h"

// Constants (actually only PI)
#include "LazyEngine/Core/Constants.h"


#ifdef LAZYENGINE_CUDA
#include "LazyEngine/platform/CUDA/CUDAUtils.h"
#include "LazyEngine/platform/CUDA/CUDAGLInteroperabilty.h"

#endif

// ## Entry Point ########################################################## //
//#include "LazyEngine/Core/EntryPoint.h"
// ######################################################################### //