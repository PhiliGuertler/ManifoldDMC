#pragma once

// ######################################################################### //
// ### Author: Philipp Gürtler ############################################# //
// ### Intersections.h ##################################################### //
// ### This file defines ray-collision primitives, like ray-sphere       ### //
// ### or ray-quad intersections                                         ### //
// ######################################################################### //

#include <LazyEngine/LazyEngine.h>
#include "../Halfedges/Halfedge.h"
#include "../Halfedges/HalfedgeVectors.h"

namespace DMC {

	/**
	 *	Describes a collision between a ray and a quad
	 */
	struct alignas(16) CollisionRayQuad {
		FaceID faceID = -1;
		glm::ivec4 quadIndices = glm::ivec4(-1);
		glm::vec3 intersectionPoint = glm::vec3(NAN);
		float distanceFromOrigin = INFINITY;

		HostDevice bool operator()(const CollisionRayQuad& a, const CollisionRayQuad& b) {
			if (a.distanceFromOrigin < 0.f && b.distanceFromOrigin < 0.f) {
				// The intersection is in the opposite ray direction!
				return false;
			}
			else if (a.distanceFromOrigin < 0.f) {
				// Only a is in the opposite ray direction, so a is not closer to the origin
				return false;
			}
			else if (b.distanceFromOrigin < 0.f) {
				// Only b is in the opposite ray direction, so a is closer to the origin
				return true;
			}
			else {
				// Return true if a is closer to the origin
				return a.distanceFromOrigin < b.distanceFromOrigin;
			}
		}
	};

	/**
	 *  Struct of two ints representing the indices of two colliding
	 *  particles. Overrides operator() to be used as a predicate for thrust::min_element.
	 */
	struct alignas(16) CollisionRayData {
		int vertexID;
		float distance = INFINITY;
		float distanceToCenter = INFINITY;

		__host__ __device__
			bool operator()(const CollisionRayData& a, const CollisionRayData& b) {
			if (a.distance < 0.f && b.distance < 0.f) {
				return a.distance < b.distance;
			}
			else if (a.distance < 0.f) {
				return true;
			}
			else if (b.distance < 0.f) {
				return false;
			}
			else {
				return a.distanceToCenter < b.distanceToCenter;
			}
		}
	};

	namespace Intersection {
		struct Plane {
			// The plane's normal
			glm::vec3 normal;
			// A single point on the plane's surface
			glm::vec3 point;
		};

		struct Triangle {
			glm::vec3 A;
			glm::vec3 B;
			glm::vec3 C;

			HostDevice inline glm::vec3 computeNormal() const {
				return glm::normalize(glm::cross(glm::normalize(B - A), glm::normalize(C - A)));
			}

			HostDevice inline Plane computePlane() const {
				return { computeNormal(), A };
			}

		};

		struct Quad {
			glm::vec3 corners[4];

			HostDevice Quad() {
				for (int i = 0; i < 4; ++i) {
					corners[i] = { NAN, NAN, NAN };
				}
			}
		};

		class BarycentricCoordinates {
		public:
			HostDevice BarycentricCoordinates(const glm::vec3& data = glm::vec3(0.f))
				: m_data(data)
			{/* Empty */
			}

			HostDevice inline void set(const glm::vec3& data) {
				m_data = data;
			}

			HostDevice inline glm::vec3 get() const {
				return m_data;
			}

			HostDevice inline operator glm::vec3& () {
				return m_data;
			}

			HostDevice inline bool isInside() const {
				// All three parts must be in [0,1], but allow some minor deviations, as floats are not precise.
				return
					(-LazyEngine::Constants::EPSILON <= m_data.x && m_data.x <= 1.f + LazyEngine::Constants::EPSILON) &&
					(-LazyEngine::Constants::EPSILON <= m_data.y && m_data.y <= 1.f + LazyEngine::Constants::EPSILON) &&
					(-LazyEngine::Constants::EPSILON <= m_data.z && m_data.z <= 1.f + LazyEngine::Constants::EPSILON);
			}

		protected:
			glm::vec3 m_data;
		};

		struct NonManifoldRays {
			LazyEngine::CameraRay rays[3];
			glm::ivec4 halfedgeIDs;

			HostDevice NonManifoldRays()
				: rays{ {glm::vec3(0.f), glm::vec3(0.f)}, {glm::vec3(0.f), glm::vec3(0.f)}, {glm::vec3(0.f), glm::vec3(0.f)} }
				, halfedgeIDs(-1)
			{
				// empty
			}

			HostDevice NonManifoldRays(glm::vec3 directions[3], glm::vec3 origins[3])
				: rays{ {directions[0], origins[0]}, {directions[1], origins[1]}, {directions[2], origins[2]} }
				, halfedgeIDs(-1)
			{
				// empty
			}
		};
	}

	class Intersector {
	public:
		/**
		 *	Intersects the scene with a ray, while every vertex is treated as a sphere with a given radius.
		 */
		static void intersectRayWithScene(LazyEngine::CameraRay ray, LazyEngine::DataView<Vertex> vertices, LazyEngine::DataView<CollisionRayData> intersections, const glm::mat4& modelToWorld, float radius);

		/**
		 *	Intersects a bunch of quads with a ray
		 */
		static void intersectRayWithQuads(LazyEngine::CameraRay ray, HalfedgeFaceHostVector& faces, HalfedgeHostVector& halfedges, LazyEngine::DataView<Vertex> vertices, LazyEngine::DataView<CollisionRayQuad> intersections, glm::mat4 modelToWorld);
		static void intersectRayWithQuads(LazyEngine::CameraRay ray, LazyEngine::DataView<Intersection::Quad> quads, LazyEngine::DataView<CollisionRayQuad> intersections, glm::mat4 modelMatrix);
	};

}