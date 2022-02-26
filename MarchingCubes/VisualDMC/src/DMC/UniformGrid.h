#pragma once

// ######################################################################### //
// ### Author: Philipp G�rtler ############################################# //
// ### UniformGrid.h ####################################################### //
// ### Defines a Host- and Device-Side UniformGrid of values.            ### //
// ### This implementation is based heavily on the implementation of     ### //
// ### uniform grids in https://github.com/rogrosso/tmc                  ### //
// ######################################################################### //

#include <LazyEngine/LazyEngine.h>

#include <thrust/device_vector.h>

#include "Utils/Utilities.h"

namespace DMC {

	// Forward declarations
	template <typename T>
	class UniformGrid;

	template <typename T>
	class UniformGridHost;

	enum class SurfaceCase : int {
		Sphere = 0,
		Torus = 1,
		TwoHoledTorus = 2,
		FourHoledTorus = 3,
		GenusTwo = 4,
		iWP = 5,
		pwHybrid = 6,
		neovius = 7,
		Goursat = 8,
		SteinerRoman = 9
	};

	enum class EdgeCase : int {
		NonManifoldVertex0 = 0,
		NonManifoldVertex1 = 1,
	};

	class UniformGridStats {
	public:
		/**
		 *	computes the global index from a 3D Index
		 */
		__host__ __device__
			inline int getGlobalIndex(const glm::ivec3& dimensionIndices) const {
			return dimensionIndices.z * m_dimensions.x * m_dimensions.y + dimensionIndices.y * m_dimensions.x + dimensionIndices.x;
		}

		__host__ __device__
			inline int getXIndexFromGlobalIndex(const int globalIndex) const {
			return globalIndex % m_dimensions.x;
		}

		__host__ __device__
			inline int getYIndexFromGlobalIndex(const int globalIndex) const {
			return (globalIndex / m_dimensions.x) % m_dimensions.y;
		}

		__host__ __device__
			inline int getZIndexFromGlobalIndex(const int globalIndex) const {
			return globalIndex / (m_dimensions.x * m_dimensions.y);
		}

		__host__ __device__
			inline glm::ivec3 get3DIndexFromGlobalIndex(const int globalIndex) const {
			return {
				getXIndexFromGlobalIndex(globalIndex),
				getYIndexFromGlobalIndex(globalIndex),
				getZIndexFromGlobalIndex(globalIndex)
			};
		}

		__host__ __device__
			inline size_t getCellCount() const {
			return static_cast<size_t>(m_dimensions.x) * static_cast<size_t>(m_dimensions.y) * static_cast<size_t>(m_dimensions.z);
		}

		__host__ __device__
			inline glm::vec3 getCellVertex(const glm::ivec3& index3D) const {
			return m_origin + glm::vec3(index3D) * m_deltas;
		}

		/**
		 *	returns the 3D cell index of the cell containing a given position
		 */
		__host__ __device__
			inline glm::ivec3 getCellIndex(const glm::vec3& position) const {
			glm::ivec3 index = glm::ivec3((position - m_origin) / m_deltas);
			return index;
		}

		__host__ inline void setDimensions(const glm::ivec3& dimensions) {
			m_dimensions = dimensions;
		}

		__host__ inline void setOrigin(const glm::vec3& origin) {
			m_origin = origin;
		}

		__host__ inline void setDeltas(const glm::vec3& deltas) {
			m_deltas = deltas;
		}

		__host__ __device__ inline glm::ivec3 getDimensions() const {
			return m_dimensions;
		}

		__host__ __device__ inline glm::vec3 getDeltas() const {
			return m_deltas;
		}

		__host__ __device__ inline glm::vec3 getOrigin() const {
			return m_origin;
		}

	public:
		// defines the size of the uniform grid in each dimension
		glm::ivec3 m_dimensions;
		// defines the step-size of the uniform grid in each dimension
		glm::vec3 m_deltas;
		// defines the value of the 0th elements of each dimension, aka the smallest coordinates
		glm::vec3 m_origin;

		glm::vec3 padding;
	};

	/**
	 *	A Uniform 3D-Grid stored in GPU-Memory that can be passed directly into CUDA kernels.
	 */
	template <typename T>
	class UniformGrid {
	public:
		/**
		 *	constructor
		 */
		inline UniformGrid(const thrust::device_vector<T>& data, const thrust::device_vector<UniformGridStats>& stats)
			: d_gridStats(stats)
			, d_data(data)
		{
			// empty
		}
		/**
		 *	destructor
		 */
		inline ~UniformGrid() = default;

		__device__ inline int getGlobalIndex(const glm::ivec3& dimensionIndices) const {
			return d_gridStats[0].getGlobalIndex(dimensionIndices);
		}

		__device__ inline glm::ivec3 get3DIndexFromGlobalIndex(const int globalIndex) const {
			return d_gridStats[0].get3DIndexFromGlobalIndex(globalIndex);
		}

		__device__ inline glm::vec3 getCellVertex(const glm::ivec3& index3D) const {
			return getOrigin() + glm::vec3(index3D) * getDeltas();
		}

		__device__ inline T operator[](const glm::ivec3& index3D) {
			int globalIndex = getGlobalIndex(index3D);
			return d_data[globalIndex];
		}

		__device__ inline T operator[](const glm::ivec3& index3D) const {
			int globalIndex = getGlobalIndex(index3D);
			return d_data[globalIndex];
		}

		__device__ inline T evaluate(const glm::ivec3& index3D) const {
			return operator[](index3D);
		}

		/**
		 *	returns the allocated GPU memory of this grid in the number of T elements
		 */
		__device__ inline size_t size() const {
			return d_data.size();
		}

		/**
		 *	returns the allocated GPU memory of this grid in bytes
		 */
		__device__ inline size_t getSizeInBytes() const {
			return d_data.getSizeInBytes();
		}

		/**
		 *	returns the dimensions of this uniform grid
		 */
		__device__ inline glm::ivec3 getDimensions() const {
			return d_gridStats[0].getDimensions();
		}

		/**
		 *	returns the deltas of this uniform grid
		 */
		__device__ inline glm::vec3 getDeltas() const {
			return d_gridStats[0].getDeltas();
		}

		/**
		 *	returns the origin of this uniform grid
		 */
		__device__ inline glm::vec3 getOrigin() const {
			return d_gridStats[0].getOrigin();
		}

		/**
		 *	returns the total number of cells of this uniform grid
		 */
		__device__ inline size_t getCellCount() const {
			return d_gridStats[0].getCellCount();
		}

		HostDevice inline LazyEngine::DataView<T> getDataView() const {
			return d_data;
		}

		using VertexArray = glm::vec3[8];

		/**
		 *	Computes the cell-corner vertices in the uniform grid using the 3D index of the cell.
		 *	@param output: A glm::vec3[8] array that will contain the results
		 *  @param index3D: The index of the cell
		 */
		__device__ inline void computeCellVertices(VertexArray& output, const glm::ivec3& index3D) const;

		using GradientArray = glm::vec3[8];

		/**
		 *	Computes the gradients of the scalar field with central difference (?)
		 *  @param output: A glm::vec3[8] array that will contain the results
		 *  @param scalar: The scalar values at the corners of this cell
		 *  @param index3D: The index of the cell
		 */
		__device__ inline void computeGradients(GradientArray& output, const T scalars[8], const glm::ivec3& index3D) const;

		using ValueArray = T[8];

		__device__ inline void evaluateCellCornerValues(ValueArray& output, const glm::ivec3& index3D) const;

		__device__ inline glm::ivec3 getIndex3DFromPosition(const glm::vec3& position) const;

	protected:
		template <typename T>
		friend class UniformGridHost;

		__host__ inline void updateDataView(const thrust::device_vector<T>& data);

		__host__ inline void updateGridStats(const thrust::device_vector<UniformGridStats>& stats);

	protected:
		// The grid-stats in gpu memory
		LazyEngine::DataView<UniformGridStats> d_gridStats;
		// the data of the uniform grid on gpu
		LazyEngine::DataView<T> d_data;
	};

	/**
	 *	The Host part of a uniform grid that is responsible of memory management.
	 *	It can be casted to a UniformGrid<T> that can be used in CUDA kernels.
	 */
	template <typename T>
	class UniformGridHost {
	public:

		UniformGridHost()
			: d_data(nullptr)
			, d_gridStats(1, "UniformGridHost-GridStats")
			, m_gridStats()
			, m_grid(nullptr)
		{
			d_data = std::make_unique<MonitoredThrustBuffer<T>>(0, "UniformGridHost-Data-Init");

			m_gridStats.m_dimensions = glm::ivec3(128);
			m_gridStats.m_origin = glm::vec3(0.f);
			m_gridStats.m_deltas = glm::vec3(1.f);
			
			thrust::copy(&m_gridStats, &m_gridStats + 1, d_gridStats.begin());
			
			m_grid = std::make_unique<UniformGrid<T>>(*d_data, d_gridStats);
		}

		/**
		 *	Uploads the given Data onto the GPU.
		 *  @param gridData: the data that should be uploaded
		 */
		__host__
		inline void setGridData(const std::vector<T>& gridData);

		__host__
		inline void generateVolume(const glm::ivec3& dimensions, const SurfaceCase& surfaceCase);

		__host__
		inline void generateEdgeCase(const EdgeCase& edgeCase);

		__host__ inline UniformGrid<T> getDeviceGrid() const {
			return *m_grid;
		}

		/**
		 *	returns the total number of cells of this uniform grid
		 */
		__host__ inline size_t getCellCount() const {
			return m_gridStats.getCellCount();
		}

		/**
		 *	returns the dimensions of this uniform grid
		 */
		__host__ inline const glm::ivec3 getDimensions() const {
			return m_gridStats.getDimensions();
		}

		/**
		 *	returns the deltas of this uniform grid
		 */
		__host__ inline const glm::vec3 getDeltas() const {
			return m_gridStats.getDeltas();
		}

		/**
		 *	returns the allocated GPU memory of this grid in the number of T elements
		 */
		__host__ inline size_t size() const {
			return d_data->size();
		}

		/**
		 *	returns the allocated GPU memory of this grid in bytes
		 */
		__host__ inline size_t getSizeInBytes() const {
			return d_data->size() * sizeof(T);
		}

		/**
		 *	returns the origin of this uniform grid
		 */
		__host__ inline glm::vec3 getOrigin() const {
			return m_gridStats.getOrigin();
		}

		/**
		 *	cast operator to a UniformGrid<T>
		 */
		__host__ inline operator UniformGrid<T>() const {
			return getDeviceGrid();
		}

		__host__ inline void setDimensions(const glm::ivec3& dimensions) {
			m_gridStats.setDimensions(dimensions);
			updateGridStats();
		}

		__host__ inline void setDeltas(const glm::vec3& deltas) {
			m_gridStats.setDeltas(deltas);
			updateGridStats();
		}

		__host__ inline void setOrigin(const glm::vec3& origin) {
			m_gridStats.setOrigin(origin);
			updateGridStats();
		}

		/**
		 *	Reads in a binary file containing the dimensions, deltas and T-type values
		 *  for the vertices.
		 *  The template typename K describes the types used in the file (e.g. unsigned shorts),
		 *	which will be transformed into T after reading
		 */
		template <typename K = unsigned short>
		__host__ inline void readFromFile(const std::string& file);
		
		/**
		 *	Reads in a binary file containing the dimensions, deltas and T-type values for the vertices
		 *	The template typename K describes the types used in the file which will be transformed into T after reading
		 *	@param file: the source-file that contains the data
		 *	@param refinement: The amount of Refinement-Cuts in between the actual grid-values. This results in bigger grids!
		 *	@param addZeroBoundary: If true, in all positive and negative dimensions, a boundary of 0-values is added.
		 *			This aims to avoid boundary cases, like non-manifold vertices and 3-face-edges
		 */
		template <typename K = unsigned short>
		__host__ inline void readFromFile(const std::string& file, int refinement, bool addZeroBoundary = false);

		/**
		 *	Tries to read the dimensions of the contained volume of a given file
		 */
		__host__ inline glm::ivec3 readDimensionsFromFile(const std::string& file);

		__host__ inline UniformGridStats getGridStats() const {
			UniformGridStats result;
			thrust::copy(d_gridStats.getBuffer().begin(), d_gridStats.getBuffer().begin()+1, &result);
			return result;
		}

		/**
		 *	Writes a sub-section of the grid to a file
		 *	@param file: The name of the output file
		 *	@param minCorner: The index3D of the vertex with the lowest x,y,z coordinates of the selection
		 *	@param maxCorner: The index3D of the vertex with the highest x,y,z coordinates of the selection
		 */
		template <typename K = unsigned short>
		__host__ inline void writeSelectionToFile(const std::string& file, const glm::ivec3& minCorner, const glm::ivec3& maxCorner);

		/**
		 *	Extracts Grid-Data starting at minCorner, up to maxCorner.
		 *	returns the dimensions of the 3D-vector
		 */
		__host__ inline glm::ivec3 extractSubData(std::vector<std::vector<std::vector<T>>>& output, const glm::ivec3& minCorner, const glm::ivec3& maxCorner);

	protected:
		__host__ inline void updateGridStats() {
			// copy the current grid stats onto the gpu
			thrust::copy(&m_gridStats, &m_gridStats + 1, d_gridStats.begin());
			// update the dataview of the device grid
			m_grid->updateGridStats(d_gridStats);
		}

	protected:
		std::unique_ptr<MonitoredThrustBuffer<T>> d_data;
		// This device vector will always contain only one UniformGridStats object.
		MonitoredThrustBuffer<UniformGridStats> d_gridStats;
		UniformGridStats m_gridStats;
		std::unique_ptr<UniformGrid<T>> m_grid;
	};
}

#include "UniformGrid.inl"