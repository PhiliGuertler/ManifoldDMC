#pragma once

// ######################################################################### //
// ### Author: Philipp Gürtler ############################################# //
// ### DualMarchingCubes.h ################################################# //
// ### Defines classes for the DualMarchingCubes Algorithm, that is      ### //
// ### based heavily on the implementation of DMC from                   ### //
// ### https://github.com/rogrosso/tmc                                   ### //
// ######################################################################### //

#include <LazyEngine/LazyEngine.h>

#include "Mesh.h"
#include "UniformGrid.h"
#include "DeviceHashTables.h"
#include "DeviceVectors.h"
#include "Halfedges/HalfedgeMesh.h"
#include "PostProcesses/ManifoldCreator.h"
#include "PostProcesses/ManifoldSplitter.h"

#include "Utils/Utilities.h"

namespace DMC {
	
	// ##################################################################### //
	// ### MarchingCubes ################################################### //
	// ##################################################################### //

	class MarchingCubes {
	public:
		static const char MC_AMBIGUOUS = 105;

		// Encodes the Edges which are intersected by the isosurface
		static const std::array<unsigned short, 256> s_edgePatterns;
		// Encodes how to connect vertices to build triangles
		static const std::array<char, 4096> s_trianglePatterns;
		// Encodes which MC cases are ambiguous and have to be solved with the asymptotic decider
		static const std::array<char, 256> s_triangleAmbiguities;
		// Encodes the MC polygons obtained by intersecting the iso-surface with a cell
		static const std::array<char, 4352> s_polygonPatterns;

	public:
		/**
		 *	constructor
		 */
		MarchingCubes();
		/**
		 *	destructor
		 */
		virtual ~MarchingCubes();

		/**
		 *	Runs the algorithm.
		 *	@param isoValue: The iso-value at which the grid should be evaluated
		 *  @param mesh: The output mesh containing the resulting surface
		 *	@param grid: The uniform grid containing the data points
		 */
		virtual Performance run(float isoValue, Mesh& mesh, const UniformGridHost<float>& grid) = 0;

		virtual void renderImGui() {}
	};


	// ##################################################################### //
	// ### DualMarchingCubes ############################################### //
	// ##################################################################### //

	class DualMarchingCubes : public MarchingCubes {
	public:
		/**
		 *	constructor
		 */
		DualMarchingCubes();
		/**
		 *	destructor
		 */
		virtual ~DualMarchingCubes();

		/**
		 *	Runs the algorithm.
		 *	@param isoValue: The iso-value at which the grid should be evaluated
		 *  @param mesh: The output mesh containing the resulting surface
		 *	@param grid: The uniform grid containing the data points
		 */
		virtual Performance run(float isoValue, Mesh& mesh, const UniformGridHost<float>& grid) override;

		inline size_t getNumVertices() const {
			return m_numVertices;
		}

		inline size_t getNumQuads() const {
			return m_numQuads;
		}

		/**
		 *	Returns the number of edges that connect 4 faces
		 */
		int getNumNonManifoldEdges4() const;
		
		/**
		 *	Returns the number of edges that connect 3 faces
		 */
		int getNumNonManifoldEdges3() const;

		/**
		 *	Returns the total number of edges
		 */
		int getNumEdges() const;

		/**
		 *	Returns the number of vertices that are not part of exactly one face-fan
		 */
		int getNumNonManifoldVertices() const;

		/**
		 *	Returns the number of quads that have exactly numManifoldEdges many non-manifold edges
		 */
		int getNumQuadsWithNonManifoldEdges(int numManifoldEdges) const;

		virtual void renderImGui() override;

		/**
		 *	Runs post processes
		 *	@param performances: The previous performances to which this method appends its times
		 */
		virtual void runPostProcesses(Performance& performances, const UniformGridHost<float>& grid, float isoValue, Mesh& debugMesh);

		inline HalfedgeMesh& getHalfedgeMesh() {
			return *m_halfedgeMesh;
		}

		Performance splitNonManifoldHalfedges(const UniformGridHost<float>& grid, float isoValue, Mesh& debugMesh);

		/**
		 *	Updates a mesh by mesh-ifying the halfedge-mesh
		 */
		void updateMesh(Mesh& mesh, const UniformGridHost<float>& grid);

		inline VertexHostVector& getVertices() {
			return *m_vertices;
		}

		inline void setSplittingStep(SplittingStep step) {
			m_programmaticSplittingStep = step;
		}

		inline bool& getSplitNonManifoldHalfedges() {
			return m_splitNonManifoldHalfedges;
		}

	protected:
		Performance initializeBuffers();
		
		// DMC Steps in separate functions
		virtual void initHashtable(Performance& performances);
		virtual void initSharedVertexList(Performance& performances);
		virtual void initQuads(Performance& performances);
		virtual void computeIsoSurface(Performance& performances, const UniformGridHost<float>& grid, float isoValue);
		virtual void mapQuads(Performance& performances);
		virtual void createMesh(Performance& performances, Mesh& mesh, const UniformGridHost<float>& grid);

	protected:
		// Counters
		size_t m_numVertices;
		size_t m_numQuads;

		// DMC Options
		bool m_enableSurfaceToVertexIteration;

		// CUDA Buffers (for DMC)
		std::unique_ptr<QuadrilateralHostHashTable> m_hashTable;
		std::unique_ptr<VertexHostVector> m_vertices;
		std::unique_ptr<QuadrilateralHostVector> m_quads;

		// Post-Process Datastructures
		std::unique_ptr<HalfedgeMesh> m_halfedgeMesh;
		std::unique_ptr<ManifoldCreator> m_manifoldCreator;
		SplittingStep m_programmaticSplittingStep;

		// Post-Processing Options
		bool m_simplifyP3x3yColor;
		bool m_simplifyP3x3yOld;
		bool m_simplifyP3333;
		bool m_splitNonManifoldHalfedges;
		bool m_detectNonManifoldVertices;
	};

}