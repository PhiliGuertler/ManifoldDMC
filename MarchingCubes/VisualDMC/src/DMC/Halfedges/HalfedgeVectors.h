#pragma once

#include <LazyEngine/LazyEngine.h>
#include "Halfedge.h"

#include "../DeviceVectors.h"
#include "../Utils/Utilities.h"

namespace DMC {

	// typedefs
	typedef glm::ivec4 NeighborFaces;
	typedef glm::ivec4 NeighborVertices;
	typedef QuadrilateralAttribute HalfedgeFace;


	// Forward declarations
	class HalfedgeHostVector;
	class HalfedgeVertexHostVector;
	class HalfedgeFaceHostVector;
	class VertexMapHost;


	// ##################################################################### //
	// ### HalfedgeVector ################################################## //
	// ##################################################################### //

	// ##################################################################### //
	// ### HalfedgeDeviceVector ############################################ //

	class HalfedgeDeviceVector : public DeviceVector {
	protected:
		typedef unsigned long long Key;

	public:
		__device__ inline int addHalfedge(const Halfedge& halfedge) {
			const int index = computeIndex();
			if (index == DeviceVector::INVALID_INDEX) {
				printf("HalfedgeDeviceVector's capacity limit was reached!\n");
				return index;
			}

			// insert the halfedge
			m_halfedges[index] = halfedge;

			return index;
		}

		/**
		 *	Inserts a halfedge and adjusts the vector size.
		 *	The vector size will only contain a meaningful value, if all halfedges get inserted unfragmented
		 */
		__device__ inline void insertHalfedge(int index, const Halfedge& halfedge) {
			incrementSize();
			m_halfedges[index] = halfedge;
		}

		/**
		 *	overload of operator[].
		 *	Can be used as a getter but it also allows mutation of the returned value.
		 */
		__device__ inline Halfedge& operator[](int index) {
			DEVICE_ASSERT((index < m_halfedges.size()), "Index out of Bounds: %llu/%llu", index, m_halfedges.size());
			return m_halfedges[index];
		}

		/**
		 *	Tries to find the halfedge between two adjacent vertices.
		 *	@param startEdgeID: The ID of a halfedge that is likely to be part of the face that contains an edge between vertex 1 and vertex 2
		 *	@param vertex1: the first vertex
		 *	@param vertex2: the second vertex
		 *	@returns the HalfedgeID of the halfedge from vertex1 to vertex2, or Halfedge::INVALID_ID if no such halfedge exists
		 */
		__device__ inline HalfedgeID findEdgeBetweenVertices(HalfedgeID startEdgeID, VertexID vertex1, VertexID vertex2) {
			// create a key for both vertices, that represents the desired combination of IDs.
			Key key = concatenateKeys(vertex1, vertex2);

			// Start with startEdgeID as edge 1.
			HalfedgeID edge1ID = startEdgeID;
			// check all four vertex-pairs of startEdgeID's face
			for (int i = 0; i < 4; ++i) {
				// get the halfedge of the start-edge id
				Halfedge halfedge1 = operator[](edge1ID);
				const VertexID origin1 = halfedge1.getOriginVertexID();

				// get the next halfedge
				const HalfedgeID edge2ID = halfedge1.getNext();
				Halfedge halfedge2 = operator[](edge2ID);
				const VertexID origin2 = halfedge2.getOriginVertexID();

				// create a key for the two vertices and compare them to the desired combination
				Key key2 = concatenateKeys(origin1, origin2);
				if (key == key2) return startEdgeID;

				edge1ID = edge2ID;
			}

			return Halfedge::INVALID_ID;
		}

		/**
		 *	Finds all neighboring FaceIDs of halfedge's face.
		 */
		__device__ inline NeighborFaces getNeighboringFaces(HalfedgeID halfedge) {
			NeighborFaces faces;

			HalfedgeID currentEdge = halfedge;
			// Iterate over this halfedge's face (which is a quad)
			for (int i = 0; i < 4; ++i) {
				// get the current edge
				Halfedge edge = m_halfedges[currentEdge];
				// get the edge's twin, which is part of the neighboring face
				Halfedge twin = m_halfedges[edge.getTwin()];
				// add the face to the result
				faces[i] = twin.getFace();

				// move on to the next edge
				currentEdge = edge.getNext();
			}

			return faces;
		}

		/**
		 *	Collects the face vertices of a quad, which is defined by a halfedge
		 *	@param halfedge: One of the face's halfedges
		 */
		__device__ inline NeighborVertices getFaceVertices(HalfedgeID halfedge) {
			NeighborVertices vertices;

			HalfedgeID currentEdge = halfedge;
			// Iterate over this halfedge's face (which is a quad)
			for (int i = 0; i < 4; ++i) {
				// get the current edge
				Halfedge edge = m_halfedges[currentEdge];
				// store its vertex id
				vertices[i] = edge.getOriginVertexID();

				// move on to the next edge
				currentEdge = edge.getNext();
			}

			return vertices;
		}

		__device__ inline void fetchFaceInfo(HalfedgeID firstHalfedge, HalfedgeID* ids, Halfedge* halfedges) {
			HalfedgeID currentEdge = firstHalfedge;
			for (int i = 0; i < 4; ++i) {
				Halfedge edge = m_halfedges[currentEdge];
				ids[i] = currentEdge;
				halfedges[i] = edge;

				// move to the next edge
				currentEdge = edge.getNext();
			}
		}

		__device__ inline void setAsTwins(HalfedgeID a, HalfedgeID b) {
			m_halfedges[a].setTwin(b);
			m_halfedges[b].setTwin(a);
		}

	protected:
		friend class HalfedgeHostVector;

		/**
		 *	protected constructor, only HalfedgeHostVector may construct this class
		 */
		HalfedgeDeviceVector(const thrust::device_vector<Halfedge>& halfedges, IndexType* size);

	protected:
		LazyEngine::DataView<Halfedge> m_halfedges;
	};


	// ##################################################################### //
	// ### HalfedgeHostVector ############################################## //

	/**
	 *	An std::vector type class for CUDA-usage.
	 *	This will store Halfedges
	 */
	class HalfedgeHostVector : public HostVector {
	public:
		HalfedgeHostVector(IndexType maxCapacity);
		virtual ~HalfedgeHostVector() = default;

		/**
		 *	Cast operator to HalfedgeDeviceVector (which must be used in CUDA-Kernels)
		 */
		inline operator HalfedgeDeviceVector() {
			DEVICE_ASSERT(m_deviceVector != nullptr, "DeviceVector is nullptr!");
			return *m_deviceVector;
		}

		/**
		 *	Resets the Vector size and initializes the data with invalid halfedges
		 */
		void clear();

		inline LazyEngine::DataView<Halfedge> getDataView() const {
			return d_halfedges;
		}

		virtual void resize(IndexType numElements) override;

		virtual inline IndexType capacity() const override {
			return d_halfedges.size();
		}

	protected:
		MonitoredThrustBuffer<Halfedge> d_halfedges;

		std::unique_ptr<HalfedgeDeviceVector> m_deviceVector;
	};


	// ##################################################################### //
	// ### HalfedgeFaceVector ############################################## //
	// ##################################################################### //

	// ##################################################################### //
	// ### HalfedgeFaceDeviceVector ######################################## //

	class HalfedgeFaceDeviceVector : public DeviceVector {
	public:
		virtual ~HalfedgeFaceDeviceVector() = default;

		__device__ inline void addFace(IndexType index, HalfedgeID face, int color = HalfedgeFace::INVALID_COLOR) {
			DEVICE_ASSERT(index < m_attributes.size(), "Index: %llu, size(): %llu", index, m_attributes.size());
			HalfedgeFace& halfedgeFace = getAttributes(index);
			halfedgeFace.setColor(color);
			halfedgeFace.unsetNonManifoldFlag();
			halfedgeFace.clearPatternFlags();
			m_firstHalfedgeIDs[index] = face;
			incrementSize();
		}

		__device__ inline void addFace(IndexType index, HalfedgeID face, HalfedgeFace attributes) {
			DEVICE_ASSERT(index < m_attributes.size(), "Index: %llu, size(): %llu", index, m_attributes.size());
			m_attributes[index] = attributes;
			m_firstHalfedgeIDs[index] = face;
			incrementSize();
		}

		__device__ inline IndexType pushFace(HalfedgeID face, HalfedgeFace attributes = HalfedgeFace()) {
			IndexType index = computeIndex();
			if (index == INVALID_INDEX) {
				printf("HalfedgeFaceDeviceVector's capacity limit was reached!\n");
				return index;
			}

			m_firstHalfedgeIDs[index] = face;
			m_attributes[index] = attributes;
			return index;
		}

		__device__ inline HalfedgeID& getFirstHalfedgeID(IndexType index) {
			DEVICE_ASSERT(index < capacity(), "Index: %llu, capacity(): %llu", index, capacity());
			return m_firstHalfedgeIDs[index];
		}

		__device__ inline HalfedgeFace& getAttributes(IndexType index) {
			DEVICE_ASSERT(index < capacity(), "Index: %llu, capacity(): %llu", index, capacity());
			return m_attributes[index];
		}

		__device__ inline IndexType capacity() const {
			return m_attributes.size();
		}

	protected:
		friend class HalfedgeFaceHostVector;

		HalfedgeFaceDeviceVector(const thrust::device_vector<HalfedgeFace>& attributes, const thrust::device_vector<HalfedgeID>& firstHalfedges, IndexType* size);

		// Stores Face-Attributes
		LazyEngine::DataView<HalfedgeFace> m_attributes;
		// Stores the HalfedgeID of any Halfedge of this Face, I guess?
		LazyEngine::DataView<HalfedgeID> m_firstHalfedgeIDs;

	};


	// ##################################################################### //
	// ### HalfedgeFaceHostVector ########################################## //

	class HalfedgeFaceHostVector : public HostVector {
	public:
		HalfedgeFaceHostVector(IndexType maxCapacity);
		virtual ~HalfedgeFaceHostVector() = default;

		inline operator HalfedgeFaceDeviceVector() {
			DEVICE_ASSERT(m_deviceVector != nullptr, "DeviceVector is nullptr");
			return *m_deviceVector;
		}

		void clear();

		inline LazyEngine::DataView<HalfedgeFace> getDataView() const {
			return d_attributes;
		}
		
		inline LazyEngine::DataView<HalfedgeID> getFirstHalfedgeIDDataView() const {
			return d_firstHalfedgeIDs;
		}

		virtual void resize(IndexType numElements) override;

		virtual inline IndexType capacity() const override {
			return d_firstHalfedgeIDs.size();
		}

	protected:
		MonitoredThrustBuffer<HalfedgeFace> d_attributes;
		MonitoredThrustBuffer<HalfedgeID> d_firstHalfedgeIDs;

		std::unique_ptr<HalfedgeFaceDeviceVector> m_deviceVector;

	};


	// ##################################################################### //
	// ### HalfedgeVertex ################################################## //
	// ##################################################################### //

	// ##################################################################### //
	// ### HalfedgeVertexDeviceVector ###################################### //

	/**
	 *	Stores one Halfedge that originates at a given Vertex and vertex-specific flags (e.g. non-manifoldness)
	 *	This Device-Variant is supposed to be used in CUDA-Kernels, as it does not manage the memory.
	 *	HalfedgeVertexHostVector is taking care of the memory instead.
	 */
	class HalfedgeVertexDeviceVector : public DeviceVector {
	public:
		/**
		 *	Returns a HalfedgeID that originates at a Vertex with the ID index.
		 *	@param index: The VertexID of the vertex whose HalfedgeID should be returned
		 */
		__device__ inline HalfedgeID& operator[](IndexType index) {
			return m_vertices[index];
		}

		/**
		 *	Returns the Flags of a specific vertex.
		 *	@param index: The VertexID of the vertex whose Flags should be returned
		 */
		__device__ inline Flags& getFlagsOf(IndexType index) {
			return m_flags[index];
		}

		__device__ inline IndexType pushVertex(HalfedgeID halfedge, Flags flags = Flags()) {
			IndexType index = computeIndex();
			if (index == INVALID_INDEX) {
				printf("HalfedgeVertexDeviceVector's capacity limit was reached!\n");
				return index;
			}

			m_vertices[index] = halfedge;
			m_flags[index] = flags;
			return index;
		}

	protected:
		friend class HalfedgeVertexHostVector;

		/**
		 *	protected constructor that can only be called from HalfedgeVertexHostVector
		 */
		HalfedgeVertexDeviceVector(const thrust::device_vector<VertexID>& vertices, const thrust::device_vector<Flags>& flags, IndexType* size);

	protected:
		// Stores the Halfedge-Address of one of this vertex's halfedges
		LazyEngine::DataView<HalfedgeID> m_vertices;
		// Stores the Flags of this Vertex
		LazyEngine::DataView<Flags> m_flags;
		
	};


	// ##################################################################### //
	// ### HalfedgeVertexHostVector ######################################## //
	
	/**
	 *	Stores one Halfedge that originates at a given Vertex and vertex-specific flags (e.g. non-manifoldness)
	 *	This Host-Variant is taking care of the memory, while HalfedgeVertexDeviceVector should be used in CUDA-Kernels.
	 */
	class HalfedgeVertexHostVector : public HostVector {
	public:
		HalfedgeVertexHostVector(IndexType maxCapacity);

		operator HalfedgeVertexDeviceVector() {
			DEVICE_ASSERT(m_deviceVector != nullptr, "DeviceVector is nullptr");
			return *m_deviceVector;
		}

		/**
		 *	Resets the Vector size and initializes the data with invalid halfedges
		 */
		void clear();

		inline LazyEngine::DataView<HalfedgeID> getDataView() const {
			return d_vertices;
		}
		
		inline LazyEngine::DataView<Flags> getFlagsDataView() const {
			return d_flags;
		}

		virtual void resize(IndexType numElements) override;
	
		virtual inline IndexType capacity() const override {
			return d_vertices.size();
		}

	protected:
		// Stores the Halfedge-Address of one of this vertex's halfedges
		MonitoredThrustBuffer<HalfedgeID> d_vertices;
		// Stores the Flags of this Vertex
		MonitoredThrustBuffer<Flags> d_flags;

		std::unique_ptr<HalfedgeVertexDeviceVector> m_deviceVector;
	};


	// ##################################################################### //
	// ### VertexMap ####################################################### //
	// ##################################################################### //

	enum VertexType: int {
		Neutral = 0,
		Pattern3X3Y = 1,
		Pattern3333 = 1,
		Pattern3333Neighbor = 2,
		PatternRemovable = 3
	};

	class VertexMapDevice {
	public:


		__device__ inline int& vertexValence(IndexType index) {
			return m_vertexValence[index];
		}

		__device__ inline int& elementCount(IndexType index) {
			return m_elementCount[index];
		}

		__device__ inline VertexType& vertexType(IndexType index) {
			return m_vertexType[index];
		}

		__device__ inline VertexID& mappingTarget(IndexType index) {
			return m_mappingTarget[index];
		}

		__device__ inline int& mappingAddress(IndexType index) {
			return m_mappingAddress[index];
		}

	protected:
		friend class VertexMapHost;

		VertexMapDevice(
			const thrust::device_vector<int>& vertexValence,
			const thrust::device_vector<int>& elementCount,
			const thrust::device_vector<VertexType>& vertexType,
			const thrust::device_vector<VertexID>& mappingTarget,
			const thrust::device_vector<int>& mappingAddress
		);

	protected:
		LazyEngine::DataView<int> m_vertexValence;
		LazyEngine::DataView<int> m_elementCount;
		LazyEngine::DataView<VertexType> m_vertexType;
		LazyEngine::DataView<VertexID> m_mappingTarget;
		LazyEngine::DataView<int> m_mappingAddress;
	};

	class VertexMapHost {
	public:
		VertexMapHost(IndexType size);

		inline IndexType capacity() const {
			return d_vertexValence.size();
		}

		void initialize();

		inline operator VertexMapDevice() const {
			DEVICE_ASSERT(m_deviceVector != nullptr, "DeviceVector is nullptr");
			return *m_deviceVector;
		}

	protected:
		// A Vertex's valence
		MonitoredThrustBuffer<int> d_vertexValence;
		// The number of elements sharing this vertex
		MonitoredThrustBuffer<int> d_elementCount;
		// The vertex type
		MonitoredThrustBuffer<VertexType> d_vertexType;
		// The mapping-target of this vertex (twin_ in original code)
		MonitoredThrustBuffer<VertexID> d_mappingTarget;
		// Mapping Address by vertex removal? (map_addr_ in original code)
		MonitoredThrustBuffer<int> d_mappingAddress;

		std::unique_ptr<VertexMapDevice> m_deviceVector;
	};


	// ##################################################################### //
	// ### Info Vectors #################################################### //
	// ##################################################################### //

	template <typename T>
	class InfoHostVector;

	template <typename T>
	class InfoDeviceVector : public DeviceVector {
	public:
		__device__ inline IndexType addInfo(const T& info) {
			const IndexType index = computeIndex();
			if (index == DeviceVector::INVALID_INDEX) return index;

			m_infos[index] = info;

			return index;
		}

		__device__ inline T& operator[](IndexType index) {
			return m_infos[index];
		}

	protected:
		friend class InfoHostVector<T>;

		inline InfoDeviceVector(const thrust::device_vector<T>& infos, IndexType* size)
			: DeviceVector(size, infos.size())
			, m_infos(infos)
		{
			// This must be called inside of a .cu file
		}

	protected:
		LazyEngine::DataView<T> m_infos;
	};

	template <typename T>
	class InfoHostVector : public HostVector {
	public:
		inline InfoHostVector(IndexType maxCapacity)
			: HostVector()
			, d_infos(maxCapacity)
			, m_deviceVector(nullptr)
		{
			// this must be called inside of a .cu file
			clear();
			m_deviceVector = std::unique_ptr<InfoDeviceVector<T>>(new InfoDeviceVector<T>(d_infos, d_size));
		}

		inline void clear() {
			// this must be called inside of a .cu file
			T invalidInfo;
			thrust::fill(d_infos.begin(), d_infos.end(), invalidInfo);
			resetSize();
		}

		inline operator InfoDeviceVector<T>() {
			DEVICE_ASSERT(m_deviceVector != nullptr, "DeviceVector is nullptr");
			return *m_deviceVector;
		}

		inline LazyEngine::DataView<T> getDataView() const {
			return d_infos;
		}

		virtual inline IndexType capacity() const override {
			return d_infos.size();
		}

	protected:
		thrust::device_vector<T> d_infos;

		std::unique_ptr<InfoDeviceVector<T>> m_deviceVector;
	};

	struct alignas(16) HalfedgeInfo {
		HalfedgeID id;
		Halfedge halfedge;
	};

	struct alignas(16) FaceInfo {
		FaceID id;
		HalfedgeFace attributes;
		HalfedgeID initialHalfedge;
	};

}
