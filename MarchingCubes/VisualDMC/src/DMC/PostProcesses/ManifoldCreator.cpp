#include "ManifoldCreator.h"
#include "ManifoldSplitter.h"

namespace DMC {

	ManifoldCreator::ManifoldCreator(HalfedgeMesh *mesh)
		: m_halfedgeMesh(mesh)
	{
		// empty
	}

	ManifoldCreator::~ManifoldCreator() {
		// empty
	}

	bool ManifoldCreator::rendersInterestingAreas() const {
		return ManifoldSplitter::rendersInterestingAreas();
	}


	void ManifoldCreator::imguiOptions() {
		ManifoldSplitter::imguiOptions();
	}

	void ManifoldCreator::imguiDebug() {
		ManifoldSplitter::imguiDebug();
	}

	Performance ManifoldCreator::splitNonManifoldHalfedgesVersion3(VertexHostVector& vertices, const UniformGridHost<float>& grid, float isoValue, Mesh& debugMesh, SplittingStep step) {
		ManifoldSplitter splitter(grid, *m_halfedgeMesh, vertices);
		SplittingStep currentSplittingStep = ManifoldSplitter::getCurrentSplittingStep();
		if (step != SplittingStep::Ignore) {
			ManifoldSplitter::enableUntilStep(step);
		}

		auto result = splitter.run(isoValue, debugMesh);
		
		if (step != SplittingStep::Ignore) {
			ManifoldSplitter::enableUntilStep(currentSplittingStep);
		}
		return result;

	}
}