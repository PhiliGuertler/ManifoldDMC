set(VisualDMC_SRC
"src/LayerCUDA.h"
"src/LayerCUDA.cu"
"src/LayerCUDA.cpp"
"src/Selection.h"
"src/Selection.cu"
"src/VisualDMCApp.cpp"

"src/DMC/CellIntersection.h"
"src/DMC/CellIntersection.inl"
"src/DMC/DeviceHashTables.h"
"src/DMC/DeviceHashTables.cu"
"src/DMC/DeviceVectors.h"
"src/DMC/DeviceVectors.cu"
"src/DMC/Mesh.h"
"src/DMC/Mesh.cu"
"src/DMC/DualMarchingCubes.h"
"src/DMC/DualMarchingCubes.cpp"
"src/DMC/DualMarchingCubes.cu"
"src/DMC/UniformGrid.h"
"src/DMC/UniformGrid.inl"

"src/DMC/Halfedges/Halfedge.h"
"src/DMC/Halfedges/HalfedgeHashTables.h"
"src/DMC/Halfedges/HalfedgeHashTables.cu"
"src/DMC/Halfedges/HalfedgeMesh.h"
"src/DMC/Halfedges/HalfedgeMesh.cu"
"src/DMC/Halfedges/HalfedgeVectors.h"
"src/DMC/Halfedges/HalfedgeVectors.cu"

"src/DMC/Utils/Intersections.h"
"src/DMC/Utils/Intersections.cu"
"src/DMC/Utils/Intersections.inl"
"src/DMC/Utils/Utilities.h"
"src/DMC/Utils/Utilities.cpp"
"src/DMC/Utils/AsymptoticDecider.h"
"src/DMC/Utils/AsymptoticDecider.cu"
"src/DMC/Utils/Interpolation.h"
"src/DMC/Utils/Interpolation.inl"

"src/DMC/PostProcesses/ManifoldCreator.h"
"src/DMC/PostProcesses/ManifoldCreator.cpp"
"src/DMC/PostProcesses/ManifoldCreator.cu"
"src/DMC/PostProcesses/ManifoldSplitter.cu"
"src/DMC/PostProcesses/ManifoldSplitter.h"
)

add_executable(VisualDMC ${VisualDMC_SRC})

target_link_libraries(VisualDMC
PUBLIC
	LazyEngine
)

set_target_properties(VisualDMC PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# group the source files in the same way in Visual Studio as on disk
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${VisualDMC_SRC})