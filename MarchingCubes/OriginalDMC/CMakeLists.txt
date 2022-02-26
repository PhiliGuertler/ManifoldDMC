set(OriginalDMC OriginalDMC)

# Set up the source files
set(OriginalDMC_SRC
"src/kernel.cu"
"src/CellIntersection.h"
"src/CTimer.h"
"src/DualMarchingCubes.cu"
"src/DualMarchingCubes.h"
"src/EdgeHashTable.h"
"src/Edges.h"
"src/FaceColoring.cu"
"src/FaceColoring.h"
"src/GaussElimination.h"
"src/HalfedgeFaces.h"
"src/HalfedgeHashTable.h"
"src/HalfedgeMesh.cu"
"src/HalfedgeMesh.h"
"src/helper_cuda.h"
"src/MarchingCubesLookupTables.h"
"src/Mesh.h"
"src/MeshSimplification.cu"
"src/MeshSimplification.h"
"src/QuadrilateralHashTable.h"
"src/QuadrilateralMap.h"
"src/Quadrilaterals.h"
"src/QualityMeasure.h"
"src/Triangles.h"
"src/UniformGrid.cu"
"src/UniformGrid.h"
"src/Vector.h"
"src/VertexHashTable.h"
"src/VertexMap.h"
"src/Vertices.h"
)

# Create an executable
add_executable(${OriginalDMC} ${OriginalDMC_SRC})

# group the source files in the same way in Visual Studio as on disk
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${OriginalDMC_SRC})