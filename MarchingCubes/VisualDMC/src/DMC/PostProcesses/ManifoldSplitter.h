#pragma once

#include <LazyEngine/LazyEngine.h>
#include "../Halfedges/HalfedgeMesh.h"
#include "../DeviceVectors.h"

namespace DMC {

	typedef int LookUpType;
	constexpr LookUpType INVALID = -1;


	// ##################################################################### //
	// ### MergeList ####################################################### //
	// ##################################################################### //
	
	class HostMergeList;

	struct MergePartners {
		// contains the amount of partners of the current vertex + 1, which means it's 3 if this vertex has 2 partners.
		int numPartners = 1;
		// In the worst case, up to 12 Vertices must be merged into one.
		int partners[12] = { INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID };
	};

	class DeviceMergeList {
	public:
		__device__ MergePartners getAllPartnersOf(int index);

		__device__ inline int& getMergeElement(int index, int elementNumber) {
			return m_mergeElementIDs[index][elementNumber];
		}

		__device__ inline int& getNumPartners(int index) {
			return m_numMergeElements[index];
		}

	protected:
		friend class HostMergeList;

		DeviceMergeList(const LazyEngine::DataView<glm::ivec2>& mergeElementIDs, const LazyEngine::DataView<int>& numMergeElements);
	
		LazyEngine::DataView<glm::ivec2> m_mergeElementIDs;
		LazyEngine::DataView<int> m_numMergeElements;
	};

	class HostMergeList {
	public:
		HostMergeList(size_t numElements);

		void initialize();

		inline operator DeviceMergeList() const {
			assert(m_deviceList != nullptr);
			return *m_deviceList;
		}

	protected:
		// Lists the ID of elements that should be merged with the current element.
		// At most 3 elements should be merging into one, so each of them should have no more than two possible partners.
		thrust::device_vector<glm::ivec2> d_mergeElementIDs;
		// Contains the amount of elements that should be merged with the current element
		thrust::device_vector<int> d_numMergeElements;

		std::unique_ptr<DeviceMergeList> m_deviceList;
	};


	// ##################################################################### //
	// ### WorkingCopy ##################################################### //
	// ##################################################################### //

	/**
	 *	A struct that bundles a datavector with flags for each element of that vector.
	 *	Also mapping buffers are included, that map to and from the working copy from and to the original data.
	 *	This class only contains Device-side access data and is not managing the memory itself.
	 */
	template <typename T>
	struct DeviceWorkingCopy {
		T dataVector;
		LazyEngine::DataView<Flags> flags;
		LazyEngine::DataView<LookUpType> copyToOriginal;
		LazyEngine::DataView<LookUpType> originalToCopy;

		DeviceWorkingCopy(
			const T& dataVector,
			const LazyEngine::DataView<Flags>& flags,
			const LazyEngine::DataView<LookUpType>& copyToOriginal,
			const LazyEngine::DataView<LookUpType>& originalToCopy
		)
			: dataVector(dataVector)
			, flags(flags)
			, copyToOriginal(copyToOriginal)
			, originalToCopy(originalToCopy)
		{
			// empty
		}

	};

	/**
	 *	A struct that bundles a datavector with flags for each element of that vector.
	 *	Also mapping buffers are included, that map to and from the working copy from and to the original data.
	 *	This class is in charge of the memory (de)allocation. The DeviceWorkingCopy should be used to access the data in CUDA kernels.
	 *	T must be statically-castable to K, where T is the Host-side Memory-Managing part and K the Device-Side memory-accessing-part.
	 */
	template <typename T, typename K>
	struct WorkingCopy {
		T dataVector;
		std::unique_ptr<MonitoredThrustBuffer<Flags>> flags;
		std::unique_ptr<MonitoredThrustBuffer<LookUpType>> copyToOriginal;
		std::unique_ptr<MonitoredThrustBuffer<LookUpType>> originalToCopy;

		WorkingCopy(size_t copySize, size_t originalSize, Flags initialFlags = Flags())
			: dataVector(copySize)
			, flags(nullptr)
			, copyToOriginal(nullptr)
			, originalToCopy(nullptr)
		{
			flags = std::make_unique<MonitoredThrustBuffer<Flags>>(copySize, "Flags");
			copyToOriginal = std::make_unique<MonitoredThrustBuffer<LookUpType>>(copySize, INVALID, "CopyToOriginal");
			originalToCopy = std::make_unique<MonitoredThrustBuffer<LookUpType>>(originalSize + copySize, INVALID, "OriginalToCopy");
			cudaCheckError();
		}

		inline DeviceWorkingCopy<K> toDevice() {
			DeviceWorkingCopy<K> device(
				static_cast<K>(dataVector),
				*flags,
				*copyToOriginal,
				*originalToCopy
			);
			return device;
		}

		inline operator DeviceWorkingCopy<K>() {
			return toDevice();
		}
	};

	typedef WorkingCopy<VertexHostVector, VertexDeviceVector> WorkingCopyVertex;
	typedef DeviceWorkingCopy<VertexDeviceVector> DeviceWorkingCopyVertex;
	
	typedef WorkingCopy<HalfedgeHostVector, HalfedgeDeviceVector> WorkingCopyHalfedge;
	typedef DeviceWorkingCopy<HalfedgeDeviceVector> DeviceWorkingCopyHalfedge;
	
	typedef WorkingCopy<HalfedgeFaceHostVector, HalfedgeFaceDeviceVector> WorkingCopyFace;
	typedef DeviceWorkingCopy<HalfedgeFaceDeviceVector> DeviceWorkingCopyFace;


	// ##################################################################### //
	// ### ManifoldSplitter ################################################ //
	// ##################################################################### //

	enum SplittingStep : int {
		None,
		SubdivideQuads,
		CollapseNonManifoldEdges,
		MergeVertices,
		UpdateHalfedgeTwins,
		MoveVerticesToSurface,
		EnableIterativeRefinement,
		// Enter additional steps here
		All = 100,
		Ignore = 101,
	};

	/**
	 *	This class splits non-manifold edges (where 4 quads connect) into Manifold edges.
	 */
	class ManifoldSplitter {
	public:
		ManifoldSplitter(const UniformGridHost<float>& grid, HalfedgeMesh& halfedgeMesh, VertexHostVector& vertices);

		/**
		 *	Runs the non-manifold halfedge splitting
		 */
		Performance run(float isoValue, Mesh& debugMesh);

		static void enableUntilStep(SplittingStep step) {
			s_debugStep = step;
		}

		static SplittingStep getCurrentSplittingStep() {
			return s_debugStep;
		}

	protected:
		// helpers
		size_t countNonManifoldFaces() const;

		void copyOriginalVertices(WorkingCopyVertex& vertices);
		void copyOriginalHalfedges(WorkingCopyHalfedge& halfedges);
		void copyOriginalFaces(WorkingCopyFace& faces, WorkingCopyHalfedge& halfedges);
		void updateHalfedgeReferences(WorkingCopyVertex& vertices, WorkingCopyHalfedge& halfedges, WorkingCopyFace& faces);
		void updateFlags(WorkingCopyHalfedge& halfedges, WorkingCopyVertex& vertices);
		Performance createWorkingCopies(WorkingCopyFace& faces, WorkingCopyHalfedge& halfedges, WorkingCopyVertex& vertices);
		
		void subdivideQuads(WorkingCopyVertex& vertices, WorkingCopyHalfedge& halfedges, WorkingCopyFace& faces);
		
		Performance collapseNonManifoldEdges(WorkingCopyVertex& vertices, WorkingCopyHalfedge& halfedges, WorkingCopyFace& faces);
		
		void moveVerticesToSurface(WorkingCopyVertex& vertices, float isoValue);
		
		Performance copyWorkingCopiesBack(WorkingCopyVertex& vertices, WorkingCopyHalfedge& halfedges, WorkingCopyFace& faces);

		
		void updateDebugMesh(WorkingCopyVertex& vertices, WorkingCopyHalfedge& halfedges, WorkingCopyFace& faces, Mesh& debugMesh);


	protected:
		const UniformGridHost<float>& m_grid;
		HalfedgeMesh& m_halfedgeMesh;
		VertexHostVector& m_vertices;

	public:
		static void imguiOptions();
		static void imguiDebug();

		inline static bool rendersInterestingAreas() {
			return s_renderInterestingAreas;
		}

	protected:
		static SplittingStep s_debugStep;
		static bool s_renderInterestingAreas;
		static bool s_copyDataBack;
	};

}