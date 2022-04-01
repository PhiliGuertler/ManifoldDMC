cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if( CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR )
    message( FATAL_ERROR "Please select another Build Directory ! (and give it a clever name, like bin_Visual2012_64bits/)" )
endif()
if( CMAKE_SOURCE_DIR MATCHES " " )
	message( "Your Source Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()
if( CMAKE_BINARY_DIR MATCHES " " )
	message( "Your Build Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()

project(ManifoldDMC LANGUAGES CUDA C CXX)

# Source file extensions do not have to be explicit (required for GLFW)
cmake_policy(SET CMP0115 OLD)

# Set build types for multi/single-config generators
set(CMAKE_CONFIGURATION_TYPES "Debug;Release")
set(CMAKE_CONFIGURATION_TYPE "Release")

# Use C++17
if(NOT DEFINED CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

# Use C++17 in CUDA
if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Use the Turing Architecture (Geforce GTX 1660ti)
#######################################################################
#### Adjust these architectures to match the used graphics card    ####
#### Pascal Architecture: 61 (GTX 1080, GTX 1070, ...)             ####
#### Volta Architecture: 70 (GTX 1180, Titan V, ...)               ####
#### Turing Architecture: 75 (GTX 1660Ti, RTX 2080, RTX 2070, ...) ####
#### Ampere Architecture: 86 (RTX 3090, RTX 3080, ...)             ####
set(CMAKE_CUDA_ARCHITECTURES 75)
####                                                               ####
#######################################################################
# Enables Linking of Device-Code, which means that device code does not need to be defined inline in headers
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Use folders in Visual Studio
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(MSVC)
	# Enable multithreaded compilation in Visual Studio
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP") 
endif()

# Add Compiler definitions to all targets:
add_compile_definitions(
	"$<$<CONFIG:Debug>:LAZYENGINE_DEBUG>"
	"$<$<CONFIG:Debug>:LAZYENGINE_ENABLE_ASSERTS>"
	"$<$<CONFIG:Release>:LAZYENGINE_RELEASE>"
	"_USE_MATH_DEFINES"
	"_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING"
	"_CRT_SECURE_NO_WARNINGS"
)

# Suppress warnings caused by glm
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr")
# Add the '-G' Options to Enable correct Debugging using NSight
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr -G")

# LazyEngine projects
add_subdirectory(LazyEngine)

# Marching Cubes Projects
add_subdirectory(MarchingCubes)
