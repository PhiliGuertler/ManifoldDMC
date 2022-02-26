// ######################################################################### //
// ### Author: Philipp G�rtler ############################################# //
// ### UniformGrid.h ####################################################### //
// ### Implements a Host- and Device-Side UniformGrid of values.         ### //
// ### This implementation is based heavily on the implementation of     ### //
// ### uniform grids in https://github.com/rogrosso/tmc                  ### //
// ######################################################################### //

#include "UniformGrid.h"

#include <algorithm>
#include <execution>
#include <fstream>

#include <LazyEngine/platform/CUDA/CUDAGLInteroperabilty.h>

#include "./Utils/Interpolation.h"

namespace DMC {

	// ##################################################################### //
	// ### Implicit Surfaces ############################################### //
	// ##################################################################### //

	__device__ inline float sq(float x) { return x * x; }
	__device__ inline float qu(float x) { return x * x * x * x; }

	__device__ inline float torusH(const glm::vec3& pos, const glm::vec3& center, const glm::vec2& param) {
		const float c = sq(param.x);
		const float a = sq(param.y);
		const float x = sq(pos.x - center.x);
		const float y = sq(pos.y - center.y);
		const float z = sq(pos.z - center.z);
		return sq(x + y + z + c - a) - 4 * c * (x + y);
	}

	__device__ inline float torusV(const glm::vec3& pos, const glm::vec3& center, const glm::vec2& param) {
		const float c = sq(param.x);
		const float a = sq(param.y);
		const float x = sq(pos.x - center.x);
		const float y = sq(pos.y - center.y);
		const float z = sq(pos.z - center.z);
		return sq(x + y + z + c - a) - 4 * c * (x + z);
	}

	__device__ inline float genusTwo(const glm::vec3& pos) {
		float alpha = 1.0;
		float x = (pos.x + 1.0f) / 2.0f;
		float y = (pos.y + 1.0f) / 2.0f;
		float z = (pos.z + 1.0f) / 2.0f;
		x = alpha * (4.0f * x - 2.0f);
		y = alpha * (4.0f * y - 2.0f);
		z = alpha * (4.0f * z - 2.0f);
		float t = 2 * y * (y * y - 3 * x * x) * (1 - z * z);
		t += (x * x + y * y) * (x * x + y * y);
		t -= (9 * z * z - 1) * (1 - z * z);
		return t;
	}

	__device__ inline float iWP(const glm::vec3& p) {
		const float alpha = 5.01;
		const float x = alpha * (p.x + 1) * LazyEngine::Constants::PI;
		const float y = alpha * (p.y + 1) * LazyEngine::Constants::PI;
		const float z = alpha * (p.z + 1) * LazyEngine::Constants::PI;
		return cos(x) * cos(y) + cos(y) * cos(z) + cos(z) * cos(x) - cos(x) * cos(y) * cos(z); // iso-value = 0
	}

	__device__ inline float pwHybrid(const glm::vec3& p) {
		const float alpha = 1.01;
		const float x = alpha * (p.x + 1) * LazyEngine::Constants::PI;
		const float y = alpha * (p.y + 1) * LazyEngine::Constants::PI;
		const float z = alpha * (p.z + 1) * LazyEngine::Constants::PI;
		return 4.0f * (cosf(x) * cosf(y) + cosf(y) * cosf(z) + cosf(z) * cosf(x)) - 3 * cosf(x) * cosf(y) * cosf(z) + 0.8f; // iso-value = 0
	}

	__device__ inline float neovius(const glm::vec3& p) {
		const float alpha = 1;
		const float x = alpha * (p.x + 1) * LazyEngine::Constants::PI;
		const float y = alpha * (p.y + 1) * LazyEngine::Constants::PI;
		const float z = alpha * (p.z + 1) * LazyEngine::Constants::PI;
		return 3 * (cos(x) + cos(y) + cos(z)) + 4 * cos(x) * cos(y) * cos(z); // iso_value = 0.0
	}

	__device__ inline float goursat(const glm::vec3& p) {
		const float a = -1.0f;
		const float b = 0.0f;
		const float c = 0.5f;
		return qu(p.x) + qu(p.y) + qu(p.z) + a * (sq(p.x) + sq(p.y) + sq(p.z)) + b * (sq(p.x) + sq(p.y) + sq(p.z)) + c;
	};

	__device__ inline float steinerRoman(const glm::vec3& p) {
		const float alpha = 1.5f;
		const float x = alpha * p.x;
		const float y = alpha * p.y;
		const float z = alpha * p.z;
		return sq(x * x + y * y + z * z - 1.0f) - (sq(z - 1) - 2.0f * x * x) * (sq(z + 1) - 2 * y * y);
	}

	__device__ inline float sphere(const glm::vec3& vertex) {
		return vertex.x * vertex.x + vertex.y * vertex.y + vertex.z * vertex.z - 0.16;
	}

	__device__ inline float torus(const glm::vec3& vertex) {
		const glm::vec2 param{ 0.3,0.15 };
		const glm::vec3 center{ 0,0,0 };
		return torusH(vertex, center, param);
	}

	__device__ inline float twoHoledTorus(const glm::vec3& vertex) {
		const glm::vec2 p1{ 0.3,0.15 };
		const float t1 = 0.38;
		const float t2 = 0.2;
		const float delta = 0.38;
		const float vt1 = torusH(vertex, glm::vec3(-t1, 0, 0), p1);
		const float vt2 = torusH(vertex, glm::vec3(t2, delta, 0), p1);
		return fminf(vt1, vt2);
	}

	__device__ inline float fourHoledTorus(const glm::vec3& vertex) {
		const glm::vec2 p2{ 0.3,0.15 };
		const float t = 0.38;
		const float v1 = torusH(vertex, glm::vec3(-t, 0, 0), p2);
		const float v2 = torusH(vertex, glm::vec3(t, 0, 0), p2);
		const float v3 = torusV(vertex, glm::vec3(0, 0, -t), p2);
		const float v4 = torusV(vertex, glm::vec3(0, 0, t), p2);
		float val = fminf(v1, v2);
		val = fminf(val, v3);
		return fminf(val, v4);
	}


	// ##################################################################### //
	// ### UniformGrid ##################################################### //
	// ##################################################################### //

	template <typename T>
	__global__ void fillVolume(UniformGrid<T> grid, SurfaceCase sc)
	{
		LazyEngine::CUDAInfo info;

		LazyEngine::DataView<T> dataView = grid.getDataView();

		// early out if this thread has nothing to do
		if (info.getGlobalThreadId() >= grid.size()) return;

		const glm::ivec3 index3D = grid.get3DIndexFromGlobalIndex(info.getGlobalThreadId());
		const glm::ivec3 dimensions = grid.getDimensions();
		if (index3D.x >= dimensions.x || index3D.y >= dimensions.y || index3D.z >= dimensions.z) return;

		float val = 0.0f;
		glm::vec3 vertex = grid.getCellVertex(index3D);

		switch (sc) {
		case SurfaceCase::Sphere:
			val = sphere(vertex);
			break;
		case SurfaceCase::Torus:
			val = torus(vertex);
			break;
		case SurfaceCase::TwoHoledTorus:
			val = twoHoledTorus(vertex);
			break;
		case SurfaceCase::FourHoledTorus:
			val = fourHoledTorus(vertex);
			break;
		case SurfaceCase::GenusTwo:
			val = genusTwo(vertex);
			break;
		case SurfaceCase::Goursat:
			val = goursat(vertex);
			break;
		case SurfaceCase::iWP:
			val = iWP(vertex);
			break;
		case SurfaceCase::pwHybrid:
			val = pwHybrid(vertex);
			break;
		case SurfaceCase::neovius:
			val = neovius(vertex);
			break;
		case SurfaceCase::SteinerRoman:
			val = steinerRoman(vertex);
			break;
		default:
			val = genusTwo(vertex);
		}

		dataView[info.getGlobalThreadId()] = val;
	}

	constexpr float hi = 950.f;
	constexpr float lo = 850.f;

	__device__ inline float nonManifoldVertex0Values(glm::ivec3 index3D) {
		const float values[3][4][4] = {
			{
				{lo, lo, lo, lo},
				{lo, lo, hi, lo},
				{lo, hi, lo, lo},
				{lo, lo, lo, lo},
			},
			{
				{lo, lo, lo, lo},
				{lo, lo, lo, lo},
				{lo, lo, lo, lo},
				{lo, lo, lo, lo},
			},
			{
				{lo, lo, lo, lo},
				{lo, lo, lo, lo},
				{lo, lo, lo, lo},
				{lo, lo, lo, lo},
			},
		};

		return values[index3D.z][index3D.y][index3D.x];
	}

	__device__ inline float nonManifoldVertex1Values(glm::ivec3 index3D) {
		const float values[3][4][4] = {
			{
				{lo, lo, lo, lo},
				{lo, lo, hi, lo},
				{lo, hi, lo, lo},
				{lo, lo, lo, lo},
			},
			{
				{lo, lo, lo, lo},
				{lo, lo, lo, lo},
				{lo, hi, lo, lo},
				{lo, lo, lo, lo},
			},
			{
				{lo, lo, lo, lo},
				{lo, lo, lo, lo},
				{lo, lo, lo, lo},
				{lo, lo, lo, lo},
			},
		};

		return values[index3D.z][index3D.y][index3D.x];
	}

	template <typename T>
	__global__ void fillEdgeCase(UniformGrid<T> grid, EdgeCase edgeCase) {
		LazyEngine::CUDAInfo info;

		LazyEngine::DataView<T> dataView = grid.getDataView();

		// early out if this thread has nothing to do
		if (info.getGlobalThreadId() >= grid.size()) return;

		const glm::ivec3 index3D = grid.get3DIndexFromGlobalIndex(info.getGlobalThreadId());
		const glm::ivec3 dimensions = grid.getDimensions();
		if (index3D.x >= dimensions.x || index3D.y >= dimensions.y || index3D.z >= dimensions.z) return;

		float val = 0.f;

		switch (edgeCase) {
		case EdgeCase::NonManifoldVertex0:
			val = nonManifoldVertex0Values(index3D);
			break;
		case EdgeCase::NonManifoldVertex1:
			val = nonManifoldVertex1Values(index3D);
			break;
		}

		dataView[info.getGlobalThreadId()] = val;
	}
	

	/**
	 *	Refines the grid and/or adds a boundary of zeros.
	 *	@param input: The input grid that should be refined and/or zero-padded
	 *	@param output: The output grid that is refined and/or zero-padded
	 *	@param refinement: The refinement-factor. Each cell will be refined into factor^3 sub-cells.
	 *	@param addZeroBoundary: If true, a boundary of 0-values will be written in the output
	 */
	template <typename T>
	__global__ void refineGrid(UniformGrid<T> input, UniformGrid<T> output, int refinement, bool addZeroBoundary) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		auto dataView = output.getDataView();
		if (threadId >= output.size()) return;

		// compute the indices of the current value
		// The index in the output grid
		glm::ivec3 index3D = output.get3DIndexFromGlobalIndex(threadId);

		if (addZeroBoundary) {
			// clamp the index3D to the input mesh, while it is offset by [1,1,1].
			// This way, the boundary values are duplicated (instead of zero).
			index3D = glm::clamp(index3D, glm::ivec3(1), output.getDimensions() - 2);
			index3D -= 1;
		}

		// The index of the low xyz-corner in the input-grid (floored)
		glm::ivec3 minIndex3DInput = glm::ivec3(glm::floor(glm::vec3(index3D) / static_cast<float>(refinement + 1)));

		glm::vec3 difference = glm::vec3(index3D);
		difference /= static_cast<float>(refinement + 1);
		difference = difference - glm::vec3(minIndex3DInput);
		// difference should now be between [0]^3 and [1]^3

		T cellValues[8];
		input.evaluateCellCornerValues(cellValues, minIndex3DInput);
		auto interpolationResult = Interpolation<T>::interpolateTrilinearly(cellValues, difference);

#ifdef LAZYENGINE_DEBUG
		for (int i = 0; i < 8; ++i) {
			if (cellValues[i] != cellValues[i]) {
				printf("%d: CellValue %d is NAN; index3D: [%d,%d,%d], minIndex3DInput: [%d,%d,%d]\n",
					threadId, i, VECCC(index3D), VECCC(minIndex3DInput));
			}
		}
#endif

		dataView[threadId] = interpolationResult;
	}

	// ##################################################################### //
	// ### UniformGrid ##################################################### //
	// ##################################################################### //

	template <typename T>
	__device__ inline void UniformGrid<T>::computeGradients(GradientArray& output, const T scalars[8], const glm::ivec3& index3D) const {
		auto index = [](int dim, int ii) { return (ii < 0 ? 0 : ii >= dim ? (dim - 1) : ii); };

		const glm::ivec3 dimensions = getDimensions();
		const glm::vec3 deltas = getDeltas();

		// vertex 0
		output[0].x = (scalars[1] - evaluate({ index(dimensions.x, index3D.x - 1), index3D.y, index3D.z }));
		output[0].y = (scalars[2] - evaluate({ index3D.x, index(dimensions.y, index3D.y - 1), index3D.z }));
		output[0].z = (scalars[4] - evaluate({ index3D.x, index3D.y, index(dimensions.z, index3D.z - 1) }));
		output[0] = 0.5f * output[0] / deltas;

		// vertex 1
		output[1].x = (evaluate({ index(dimensions.x, index3D.x + 2), index3D.y, index3D.z }) - scalars[0]);
		output[1].y = (scalars[3] - evaluate({ index3D.x + 1, index(dimensions.y, index3D.y - 1), index3D.z }));
		output[1].z = (scalars[5] - evaluate({ index3D.x + 1, index3D.y, index(dimensions.z, index3D.z - 1) }));
		output[1] = 0.5f * output[1] / deltas;

		// vertex 2
		output[2].x = (scalars[3] - evaluate({ index(dimensions.x, index3D.x - 1), index3D.y + 1, index3D.z }));
		output[2].y = (evaluate({ index3D.x, index(dimensions.y, index3D.y + 2), index3D.z }) - scalars[0]);
		output[2].z = (scalars[6] - evaluate({ index3D.x, index3D.y + 1, index(dimensions.z, index3D.z - 1) }));
		output[2] = 0.5f * output[2] / deltas;

		// vertex 3
		output[3].x = (evaluate({ index(dimensions.x, index3D.x + 2), index3D.y + 1, index3D.z }) - scalars[2]);
		output[3].y = (evaluate({ index3D.x + 1, index(dimensions.y, index3D.y + 2), index3D.z }) - scalars[1]);
		output[3].z = (scalars[7] - evaluate({ index3D.x + 1, index3D.y + 1, index(dimensions.z, index3D.z - 1) }));
		output[3] = 0.5f * output[3] / deltas;

		// vertex 4
		output[4].x = (scalars[5] - evaluate({ index(dimensions.x, index3D.x - 1), index3D.y, index3D.z + 1 }));
		output[4].y = (scalars[6] - evaluate({ index3D.x, index(dimensions.y, index3D.y - 1), index3D.z + 1 }));
		output[4].z = (evaluate({ index3D.x, index3D.y, index(dimensions.z, index3D.z + 2) }) - scalars[0]);
		output[4] = 0.5f * output[4] / deltas;

		// vertex 5
		output[5].x = (evaluate({ index(dimensions.x, index3D.x + 2), index3D.y, index3D.z + 1 }) - scalars[4]);
		output[5].y = (scalars[7] - evaluate({ index3D.x + 1, index(dimensions.y, index3D.y - 1), index3D.z + 1 }));
		output[5].z = (evaluate({ index3D.x + 1, index3D.y, index(dimensions.z, index3D.z + 2) }) - scalars[1]);
		output[5] = 0.5f * output[5] / deltas;

		// vertex 6
		output[6].x = (scalars[7] - evaluate({ index(dimensions.x, index3D.x - 1), index3D.y + 1, index3D.z + 1 }));
		output[6].y = (evaluate({ index3D.x, index(dimensions.y, index3D.y + 2), index3D.z + 1 }) - scalars[4]);
		output[6].z = (evaluate({ index3D.x, index3D.y + 1, index(dimensions.z, index3D.z + 2) }) - scalars[2]);
		output[6] = 0.5f * output[6] / deltas;

		// vertex 7
		output[7].x = (evaluate({ index(dimensions.x, index3D.x + 2), index3D.y + 1, index3D.z + 1 }) - scalars[6]);
		output[7].y = (evaluate({ index3D.x + 1, index(dimensions.y, index3D.y + 2), index3D.z + 1 }) - scalars[5]);
		output[7].z = (evaluate({ index3D.x + 1, index3D.y + 1, index(dimensions.z, index3D.z + 2) }) - scalars[3]);
		output[7] = 0.5f * output[7] / deltas;
	}

	template <typename T>
	__device__ inline void UniformGrid<T>::computeCellVertices(VertexArray& output, const glm::ivec3& index3D) const {
		const glm::vec3 deltas = getDeltas();

		output[0] = getOrigin() + index3D * deltas;

		output[1].x = output[0].x + deltas.x;
		output[1].y = output[0].y;
		output[1].z = output[0].z;

		output[2].x = output[0].x;
		output[2].y = output[0].y + deltas.y;
		output[2].z = output[0].z;

		output[3].x = output[0].x + deltas.x;
		output[3].y = output[0].y + deltas.y;
		output[3].z = output[0].z;

		output[4].x = output[0].x;
		output[4].y = output[0].y;
		output[4].z = output[0].z + deltas.z;

		output[5].x = output[0].x + deltas.x;
		output[5].y = output[0].y;
		output[5].z = output[0].z + deltas.z;

		output[6].x = output[0].x;
		output[6].y = output[0].y + deltas.y;
		output[6].z = output[0].z + deltas.z;

		output[7] = output[0] + deltas;
	}

	template <typename T>
	__device__ inline void UniformGrid<T>::evaluateCellCornerValues(ValueArray& output, const glm::ivec3& index3D) const {
		// To avoid out-of-bounds evaluations, clamp the index to its closest boundary-case.
		glm::ivec3 maxIndex = getDimensions() - 1;
		glm::ivec3 minIndex = glm::ivec3(0, 0, 0);
		output[0] = evaluate(glm::clamp(index3D + glm::ivec3(0, 0, 0), minIndex, maxIndex));
		output[1] = evaluate(glm::clamp(index3D + glm::ivec3(1, 0, 0), minIndex, maxIndex));
		output[2] = evaluate(glm::clamp(index3D + glm::ivec3(0, 1, 0), minIndex, maxIndex));
		output[3] = evaluate(glm::clamp(index3D + glm::ivec3(1, 1, 0), minIndex, maxIndex));
		output[4] = evaluate(glm::clamp(index3D + glm::ivec3(0, 0, 1), minIndex, maxIndex));
		output[5] = evaluate(glm::clamp(index3D + glm::ivec3(1, 0, 1), minIndex, maxIndex));
		output[6] = evaluate(glm::clamp(index3D + glm::ivec3(0, 1, 1), minIndex, maxIndex));
		output[7] = evaluate(glm::clamp(index3D + glm::ivec3(1, 1, 1), minIndex, maxIndex));
	}

	template <typename T>
	inline void UniformGrid<T>::updateDataView(const thrust::device_vector<T>& data) {
		d_data = data;
	}

	template <typename T>
	inline void UniformGrid<T>::updateGridStats(const thrust::device_vector<UniformGridStats>& stats) {
		assert(stats.size() == 1);
		d_gridStats = stats;
	}

	template <typename T>
	__device__ glm::ivec3 UniformGrid<T>::getIndex3DFromPosition(const glm::vec3& position) const {
		// compute the offset from the min corner
		glm::vec3 offset = position - getOrigin();
		// divide the offset by the deltas
		offset /= getDeltas();
		// now floor the result
		return glm::ivec3(glm::floor(offset));
	}

	// ##################################################################### //
	// ### UniformGridHost ################################################# //
	// ##################################################################### //

	template <typename T>
	inline void UniformGridHost<T>::setGridData(const std::vector<T>& gridData) {
		d_data.reset(nullptr);
		d_data = std::make_unique<MonitoredThrustBuffer<T>>(gridData.size(), "UniformGridHost-Data");
		*d_data = gridData;
		m_grid->updateDataView(d_data);
	}

	template <typename T>
	inline void UniformGridHost<T>::generateVolume(const glm::ivec3& dimensions, const SurfaceCase& surfaceCase) {
		// set the volume size
		m_gridStats.m_dimensions = dimensions;

		// define the domain and grid size
		glm::vec3 min = glm::vec3(-1.f);
		glm::vec3 max = glm::vec3(1.f);

		m_gridStats.m_deltas = (max - min) / (glm::vec3(m_gridStats.getDimensions()) - 1.f);
		m_gridStats.m_origin = min;

		d_data.reset(nullptr);
		d_data = std::make_unique<MonitoredThrustBuffer<T>>(getCellCount(), "UniforemGridHost-Data");

		//d_data.resize(getCellCount() * sizeof(T));

		m_grid = std::make_unique<UniformGrid<T>>(*d_data, d_gridStats);
		m_grid->updateDataView(*d_data);
		updateGridStats();

		LazyEngine::DataView<T> dataView = m_grid->getDataView();

		fillVolume ToCudaArgs(dataView) (*m_grid, surfaceCase);
	}

	template <typename T>
	inline void UniformGridHost<T>::generateEdgeCase(const EdgeCase& edgeCase) {
		m_gridStats.m_dimensions = { 4,4,3 };

		// define the domain and grid size
		glm::vec3 min = glm::vec3(-1.f);
		glm::vec3 max = glm::vec3(1.f);

		m_gridStats.m_deltas = (max - min) / (glm::vec3(m_gridStats.getDimensions()) - 1.f);
		m_gridStats.m_origin = min;

		d_data.reset(nullptr);
		d_data = std::make_unique<MonitoredThrustBuffer<T>>(getCellCount(), "UniforemGridHost-Data");

		m_grid = std::make_unique<UniformGrid<T>>(*d_data, d_gridStats);
		m_grid->updateDataView(*d_data);
		updateGridStats();

		LazyEngine::DataView<T> dataView = m_grid->getDataView();

		fillEdgeCase ToCudaArgs(dataView) (*m_grid, edgeCase);
	}
	

	template <typename T>
	template <typename K>
	inline void UniformGridHost<T>::readFromFile(const std::string& file) {
		// Open file
		std::ifstream input;
		input.open(file, std::ios::binary);

		if (!input.is_open()) {
			LAZYENGINE_ERROR("File \"{0}\" does not exist!", file);
			throw std::exception("File does not exist");
		}

		// read the dimensions
		glm::ivec3 dimensions = { 0,0,0 };
		input.read(reinterpret_cast<char*>(&dimensions.x), sizeof(int));
		input.read(reinterpret_cast<char*>(&dimensions.y), sizeof(int));
		input.read(reinterpret_cast<char*>(&dimensions.z), sizeof(int));

		// read the deltas
		glm::vec3 deltas = { 0.f,0.f,0.f };
		input.read(reinterpret_cast<char*>(&deltas.x), sizeof(float));
		input.read(reinterpret_cast<char*>(&deltas.y), sizeof(float));
		input.read(reinterpret_cast<char*>(&deltas.z), sizeof(float));

		m_gridStats.m_deltas = deltas;
		m_gridStats.m_origin = -0.5f * glm::vec3(dimensions - 1) * deltas;


		m_gridStats.m_dimensions = dimensions;
		//m_gridStats.m_deltas = deltas;
		updateGridStats();

		// create a buffer into which the vertex-data is read
		size_t currentSize = getCellCount();
		std::vector<K> readingBuffer(currentSize);

		// read the vertex-data
		input.read(reinterpret_cast<char*>(readingBuffer.data()), readingBuffer.size() * sizeof(K));
		// close the input file
		input.close();

		std::vector<T> castedBuffer(currentSize);
		for (int i = 0; i < castedBuffer.size(); ++i) {
			castedBuffer[i] = static_cast<T>(readingBuffer[i]);
			if (castedBuffer[i] != castedBuffer[i]) {
				LAZYENGINE_INFO("NAN encountered: {0}", i);
				castedBuffer[i] = 0.f;
			}
		}

#ifdef LAZYENGINE_DEBUG
		LazyEngine::printCudaMemory();
#endif

		d_data.reset(nullptr);
		d_data = std::make_unique<MonitoredThrustBuffer<T>>(castedBuffer.size(), "UniformGridHost-Data");

		thrust::copy(castedBuffer.begin(), castedBuffer.end(), d_data->begin());
		m_grid->updateDataView(*d_data);
	}

	template <typename T>
	template <typename K>
	inline void UniformGridHost<T>::readFromFile(const std::string& file, int refinement, bool addZeroBoundary) {
		// First, read the file regularly into a temporary UniformGridHost
		UniformGridHost<T> tmpGrid;
		try {
			tmpGrid.readFromFile<K>(file);
		}
		catch (std::exception& e) {
			throw e;
		}

		// Compute the dimensions of the refined uniform grid
		glm::ivec3 dimensions = tmpGrid.getDimensions();
		glm::ivec3 refinedDimensions = dimensions + (dimensions - 1) * refinement;
		if (addZeroBoundary) {
			refinedDimensions += 2;
		}

		// define the domain and grid size
		m_gridStats.m_deltas = tmpGrid.getDeltas() * 1.f/(static_cast<float>(refinement) + 1.f);
		m_gridStats.m_origin = tmpGrid.getOrigin();
		if (addZeroBoundary) {
			m_gridStats.m_origin -= m_gridStats.m_deltas;
		}
		m_gridStats.m_dimensions = refinedDimensions;
		updateGridStats();

		// allocate buffer memory
		d_data.reset(nullptr);
		d_data = std::make_unique<MonitoredThrustBuffer<T>>(getCellCount(), "UniformGridHost-Data");
		m_grid->updateDataView(*d_data);

		// interpolate the refined values in a CUDA kernel
		refineGrid<T> ToCudaArgs(m_grid->getDataView()) (tmpGrid, *this, refinement, addZeroBoundary);
	}

	template <typename T>
	inline glm::ivec3 UniformGridHost<T>::readDimensionsFromFile(const std::string& file) {
		// Open file
		std::ifstream input;
		input.open(file, std::ios::binary);

		if (!input.is_open()) {
			LAZYENGINE_ERROR("File \"{0}\" does not exist!", file);
			return { -1,-1,-1 };
		}

		// read the dimensions
		glm::ivec3 dimensions = { 0,0,0 };
		input.read(reinterpret_cast<char*>(&dimensions.x), sizeof(int));
		input.read(reinterpret_cast<char*>(&dimensions.y), sizeof(int));
		input.read(reinterpret_cast<char*>(&dimensions.z), sizeof(int));

		input.close();

		return dimensions;
	}

	inline bool endsWith(const std::string& string, const std::string& ending) {
		if (string.length() >= ending.length()) {
			return (string.compare(string.length() - ending.length(), ending.length(), ending) == 0);
		}
		return false;
	}

	template <typename T>
	template <typename K>
	inline void UniformGridHost<T>::writeSelectionToFile(const std::string& file, const glm::ivec3& minCorner, const glm::ivec3& maxCorner) {
		// check if an ".obj" is already part of the filepath.
		// if not, append it.
		std::string extension = ".bin";
		std::string finalFilePath = file;
		if (!endsWith(file, extension)) {
			finalFilePath = file + extension;
		}
		
		// open file
		std::ofstream output;
		output.open(finalFilePath, std::ios::binary | std::ios::trunc);

		if (!output.is_open()) {
			LAZYENGINE_ERROR("Could not open \"{0}\" for writing!", file);
			return;
		}

		// write the dimensions
		std::vector<std::vector<std::vector<T>>> vertexData;
		glm::ivec3 dimensions = extractSubData(vertexData, minCorner, maxCorner);

		LAZYENGINE_INFO("Writing Dimensions [{0},{1},{2}]", dimensions.x, dimensions.y, dimensions.z);

		//glm::ivec3 dimensions = maxCorner - minCorner;
		output.write(reinterpret_cast<char*>(&dimensions.x), sizeof(int));
		output.write(reinterpret_cast<char*>(&dimensions.y), sizeof(int));
		output.write(reinterpret_cast<char*>(&dimensions.z), sizeof(int));

		// write deltas
		output.write(reinterpret_cast<char*>(&m_gridStats.m_deltas.x), sizeof(float));
		output.write(reinterpret_cast<char*>(&m_gridStats.m_deltas.y), sizeof(float));
		output.write(reinterpret_cast<char*>(&m_gridStats.m_deltas.z), sizeof(float));

		// write the grid values
		for (int z = 0; z < vertexData.size(); ++z) {
			const auto& currentPlane = vertexData[z];

			for (int y = 0; y < currentPlane.size(); ++y) {
				const auto& currentLine = currentPlane[y];

				for (int x = 0; x < currentLine.size(); ++x) {
					T data = currentLine[x];

					// cast the data to the desired output type
					K outputData = static_cast<K>(data);

					// write it into the output file
					output.write(reinterpret_cast<char*>(&outputData), sizeof(K));
				}
			}
		}

		output.close();
	}

	template <typename T>
	__host__ inline glm::ivec3 UniformGridHost<T>::extractSubData(std::vector<std::vector<std::vector<T>>>& output, const glm::ivec3& minCorner, const glm::ivec3& maxCorner) {
		glm::ivec3 gridDimensions = getDimensions();
		glm::ivec3 clampedMinCorner = glm::clamp(minCorner, glm::ivec3(0, 0, 0), gridDimensions - 1);
		glm::ivec3 clampedMaxCorner = glm::clamp(maxCorner, glm::ivec3(0, 0, 0), gridDimensions - 1);

		glm::ivec3 dimensions = clampedMaxCorner - clampedMinCorner + 1;

		output.clear();

		for (int z = 0; z < dimensions.z; ++z) {
			// create vector for the current plane
			std::vector<std::vector<T>> currentPlane;

			for (int y = 0; y < dimensions.y; ++y) {
				// compute linearized global index
				glm::ivec3 currentIndex3D = clampedMinCorner + glm::ivec3(0, y, z);

				int currentGlobalIndex = m_gridStats.getGlobalIndex(currentIndex3D);
				// extract a whole line of values at once
				std::vector<T> line(dimensions.x);
				thrust::copy(d_data->begin() + currentGlobalIndex, d_data->begin() + currentGlobalIndex + line.size(), line.begin());

				// add line to current plane
				currentPlane.push_back(line);
			}

			output.push_back(currentPlane);
		}

		return dimensions;
	}
}
