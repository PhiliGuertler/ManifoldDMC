#pragma once

#include <LazyEngine/LazyEngine.h>

namespace DMC {

	typedef int HalfedgeID;
	typedef int VertexID;
	typedef int FaceID;


	// Watch Out! These functions are all racy, as they are not atomic!
	// However, there are no atomic and/or functions for 8-bit types, only for 32 and 64 bit types!
	class Flags {
		typedef uint8_t InternalType;
	
	public:
		HostDevice Flags()
			: m_flags(static_cast<InternalType>(0x0)) {}

		HostDevice Flags(InternalType initial) : m_flags(initial) {}

		/**
		 *	Should be set if at least one of this vertex's outgoing halfedges is Non-Manifold.
		 */
		HostDevice inline void setNonManifoldFlag() {
			m_flags |= static_cast<InternalType>(BIT(0));
		}
		/**
		 *	If true, at least one of this vertex's outgoing halfedges is Non-Manifold.
		 */
		HostDevice inline bool isNonManifold() const {
			return m_flags & static_cast<InternalType>(BIT(0));
		}
		
		/**
		 *	Should be set if the Vertex is part of a Boundary.
		 */
		HostDevice inline void setIsBoundaryFlag() {
			m_flags |= static_cast<InternalType>(BIT(1));
		}
		/**
		 *	If true, this Vertex is part of a Boundary.
		 */
		HostDevice inline bool isBoundary() const {
			return m_flags & static_cast<InternalType>(BIT(1));
		}

		/**
		 *	Currently used for halfedges that have exactly two twins (which is non-manifold)
		 */
		HostDevice inline void setWeirdFlag() {
			m_flags |= static_cast<InternalType>(BIT(2));
		}

		HostDevice inline bool isWeird() const {
			return m_flags & static_cast<InternalType>(BIT(2));
		}

		/**
		 *	Will be set when searching for Non-Manifold Vertices.
		 */
		HostDevice inline void setMultipleFansFlag() {
			m_flags |= static_cast<InternalType>(BIT(3));
		}
		HostDevice inline void unsetMutlipleFansFlag() {
			m_flags &= static_cast<InternalType>(~BIT(3));
		}
		/**
		 *	If true, this is a Non-Manifold Vertex
		 */
		HostDevice inline bool hasMultipleFans() const {
			return m_flags & static_cast<InternalType>(BIT(3));
		}

		// ### Remove Flag ### //
		HostDevice inline void setRemoveFlag() {
			m_flags |= static_cast<InternalType>(BIT(4));
		}
		HostDevice inline void unsetRemoveFlag() {
			m_flags &= static_cast<InternalType>(~BIT(4));
		}
		HostDevice inline bool isRemoveFlagSet() const {
			return m_flags & static_cast<InternalType>(BIT(4));
		}

		// ### Keep Twin Flag ### //
		HostDevice inline void setKeepTwinFlag() {
			m_flags |= static_cast<InternalType>(BIT(5));
		}
		HostDevice inline void unsetKeepTwinFlag() {
			m_flags &= static_cast<InternalType>(BIT(5));
		}
		HostDevice inline bool isKeepTwinFlagSet() {
			return m_flags & static_cast<InternalType>(BIT(5));
		}

		HostDevice inline void reset() {
			m_flags = static_cast<InternalType>(0x0);
		}

	public:
		HostDevice static Flags RemoveFlag() {
			Flags flag;
			flag.setRemoveFlag();
			return flag;
		}

	protected:
		InternalType m_flags;
	};

	/**
	 *	Represents a single half-edge, that stores
	 *	- its origin-vertex-ID,
	 *	- its face-ID,
	 *	- its next half-edge-ID,
	 *	- its twin half-edge-ID
	 */
	class alignas(16) Halfedge {
	public:
		static const int INVALID_ID = -1;


		/**
		 *	constructor
		 */
		HostDevice Halfedge(VertexID originVertexID = INVALID_ID, FaceID faceID = INVALID_ID, HalfedgeID nextHalfedgeID = INVALID_ID, HalfedgeID twinHalfedgeID = INVALID_ID)
			: m_data({ originVertexID, faceID, nextHalfedgeID, twinHalfedgeID })
		{
			// empty
		}

		// ### General Methods ############################################# //
		/**
		 *	Returns true if a twin is set
		 */
		HostDevice inline bool hasTwin() const {
			return m_data.w != INVALID_ID;
		}


		// ### Setters ##################################################### //
		/**
		 *	Sets the vertex-ID of this halfedge's origin
		 */
		HostDevice inline void setOriginVertexID(VertexID id) {
			m_data.x = id;
		}
		/**
		 *	Sets the Face-ID of which this halfedge is a part
		 */
		HostDevice inline void setFace(FaceID id) {
			m_data.y = id;
		}
		/**
		 *	Sets the id of the next halfedge of the current face
		 */
		HostDevice inline void setNext(HalfedgeID id) {
			m_data.z = id;
		}
		/**
		 *	Sets the id of this halfedge's twin
		 */
		HostDevice inline void setTwin(HalfedgeID id) {
			m_data.w = id;
		}

		// ### Getters ##################################################### //
		/**
		 *	Returns the ID of the origin vertex
		 *	equivalent to original .x
		 */
		HostDevice inline VertexID getOriginVertexID() const {
			return m_data.x;
		}
		/**
		 *	Returns the ID of this halfedge's corresponding face
		 *	equivalent to original .y
		 */
		HostDevice inline FaceID getFace() const {
			return m_data.y;
		}
		/**
		 *	Returns the ID of the next halfedge of this halfedge's face
		 *	equivalent to original .z
		 */
		HostDevice inline HalfedgeID getNext() const {
			return m_data.z;
		}
		/**
		 *	Returns the ID of this halfedge's twin.
		 *	equivalent to original .w
		 */
		HostDevice inline HalfedgeID getTwin() const {
			return m_data.w;
		}

		HostDevice inline void print(int id = -1) const {
			printf("[%d]: Vertex: %d, Face: %d, Next: %d, Twin: %d\n", id, getOriginVertexID(), getFace(), getNext(), getTwin());
		}

	protected:
		// x: Origin Vertex ID
		// y: Face ID
		// z: Next Halfedge ID
		// w: Twin Halfedge ID
		glm::ivec4 m_data;

	};


}