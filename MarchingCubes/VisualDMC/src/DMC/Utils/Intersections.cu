// ######################################################################### //
// ### Author: Philipp Gürtler ############################################# //
// ### Intersections.cu #################################################### //
// ### Implements the intersection collision primitives from             ### //
// ### Intersections.h                                                   ### //
// ######################################################################### //

#include <LazyEngine/LazyEngine.h>
#include "Intersections.inl"

#define VECCC(a) a.x, a.y, a.z

namespace DMC {

	// ##################################################################### //
	// ### Device Code ##################################################### //
	// ##################################################################### //

	__global__ void intersectRayWithSceneKernel(
		LazyEngine::DataView<Vertex> vertices,
		LazyEngine::DataView<CollisionRayData> intersections,
		LazyEngine::CameraRay ray,
		glm::mat4 modelToWorld,
		float radius
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= vertices.size()) return;

		glm::vec4 transformedPosition = modelToWorld * glm::vec4(vertices[threadId].position, 1.f);

		CollisionRayData intersection = collideSphereRay(glm::vec3(transformedPosition / transformedPosition.w), radius, ray);
		intersection.vertexID = threadId;
		intersections[threadId] = intersection;
	}

	__global__ void intersectRayWithQuadsKernel(
		HalfedgeFaceDeviceVector faces,
		HalfedgeDeviceVector halfedges,
		LazyEngine::DataView<Vertex> vertices,
		LazyEngine::DataView<CollisionRayQuad> intersections,
		glm::mat4 modelToWorld,
		LazyEngine::CameraRay ray
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= faces.size()) return;

		// get the face-vertices of the current face
		HalfedgeID currentFaceHalfedge = faces.getFirstHalfedgeID(threadId);
		// first, fetch the vertex id's that make up the quad
		NeighborVertices neighborIDs = halfedges.getFaceVertices(currentFaceHalfedge);

		// next, create an intersection quad containing the vertices' positions
		Intersection::Quad quad;
		for (int i = 0; i < 4; ++i) {
			glm::vec4 transformedPosition = modelToWorld * glm::vec4(vertices[neighborIDs[i]].position, 1.f);
			quad.corners[i] = glm::vec3(transformedPosition / transformedPosition.w);
		}

		CollisionRayQuad intersection;
		intersection.intersectionPoint = intersectRayQuad(ray, quad);
		if (intersection.intersectionPoint.x == intersection.intersectionPoint.x) {
			// An intersection exists
			// compute the signed distance from the origin
			intersection.distanceFromOrigin = ray.signedDistanceTo(intersection.intersectionPoint);
			intersection.faceID = threadId;
			intersection.quadIndices = neighborIDs;
		}

		intersections[threadId] = intersection;
	}

	__global__ void intersectRayWithQuadsKernel(
		LazyEngine::CameraRay ray,
		LazyEngine::DataView<Intersection::Quad> quads,
		LazyEngine::DataView<CollisionRayQuad> intersections,
		glm::mat4 modelToWorld
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= quads.size()) return;

		Intersection::Quad quad = quads[threadId];
		// transform the quad from model-space to world space (the same space as the ray)
		for (int i = 0; i < 4; ++i) {
			glm::vec4 transformedPosition = modelToWorld * glm::vec4(quad.corners[i], 1.f);
			quad.corners[i] = glm::vec3(transformedPosition / transformedPosition.w);
		}

		// Perform the intersection itelf
		CollisionRayQuad intersection;
		intersection.intersectionPoint = intersectRayQuad(ray, quad);
		if (intersection.intersectionPoint.x == intersection.intersectionPoint.x) {
			// An intersection exists
			// compute the signed distance from the origin
			intersection.distanceFromOrigin = ray.signedDistanceTo(intersection.intersectionPoint);
			intersection.faceID = threadId;
			// Ignore the Vertex IDs in this case
			//intersection.quadIndices = neighborIDs;
		}

		intersections[threadId] = intersection;
	}

	// ##################################################################### //
	// ### Host Code ####################################################### //
	// ##################################################################### //

	void Intersector::intersectRayWithScene(LazyEngine::CameraRay ray, LazyEngine::DataView<Vertex> vertices, LazyEngine::DataView<CollisionRayData> intersections, const glm::mat4& modelToWorld, float radius) {
		intersectRayWithSceneKernel ToCudaArgs(vertices) (vertices, intersections, ray, modelToWorld, radius);
	}

	void Intersector::intersectRayWithQuads(LazyEngine::CameraRay ray, HalfedgeFaceHostVector& faces, HalfedgeHostVector& halfedges, LazyEngine::DataView<Vertex> vertices, LazyEngine::DataView<CollisionRayQuad> intersections, glm::mat4 modelToWorld) {
		intersectRayWithQuadsKernel ToCudaArgs(faces.getDataView()) (faces, halfedges, vertices, intersections, modelToWorld, ray);
	}

	void Intersector::intersectRayWithQuads(LazyEngine::CameraRay ray, LazyEngine::DataView<Intersection::Quad> quads, LazyEngine::DataView<CollisionRayQuad> intersections, glm::mat4 modelToWorld) {
		intersectRayWithQuadsKernel ToCudaArgs(quads) (ray, quads, intersections, modelToWorld);
	}

}