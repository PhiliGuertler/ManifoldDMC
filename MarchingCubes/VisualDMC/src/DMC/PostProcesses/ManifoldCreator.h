#pragma once

#include <LazyEngine/LazyEngine.h>
#include "../Halfedges/HalfedgeMesh.h"
#include "../DeviceVectors.h"
#include "ManifoldSplitter.h"


namespace DMC {

	/**
	 *	Bundles functionalities for resolving non-manifold cases
	 */
	class ManifoldCreator {
	public:
		/**
		 *	constructor
		 *	@param mesh: The halfedge-mesh on which to operate. nullptr by default
		 */
		ManifoldCreator(HalfedgeMesh *mesh = nullptr);
		/**
		 *	destructor
		 */
		~ManifoldCreator();

		inline void setHalfedgeMesh(HalfedgeMesh* mesh) {
			m_halfedgeMesh = mesh;
		}

		Performance splitNonManifoldHalfedgesVersion3(VertexHostVector& vertices, const UniformGridHost<float>& grid, float isoValue, Mesh& debugMesh, SplittingStep enabledSteps = SplittingStep::Ignore);

		/**
		 *	Changes up the twins for manifold edges using an Asymptotic Decider
		 */
		void fixManifoldTwins(VertexHostVector& vertices, const UniformGridHost<float>& grid, float isoValue);

		/**
		 *	Meant to test fixManifoldTwins.
		 *	This function will render the cell-walls that are intersected by non-manifold edges.
		 *	outVertices must be at least numNonManifoldEdges*4 and outIndices must be at least numNonManifoldEdges*6 big
		 */
		void renderManifoldCellWalls(VertexHostVector& vertices, const UniformGridHost<float>& grid, LazyEngine::DataView<Vertex>& outVertices, LazyEngine::DataView<uint32_t>& outIndices);

		Performance detectNonManifoldVertices();

		int getNumNonManifoldVertices();

		void imguiOptions();
		void imguiDebug();

		bool rendersInterestingAreas() const;

	protected:
		// The halfedge-mesh on which the Manifold Creator will perform its actions
		HalfedgeMesh *m_halfedgeMesh;
	};

}