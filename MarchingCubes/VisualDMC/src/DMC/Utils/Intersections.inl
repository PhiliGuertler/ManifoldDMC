#pragma once

#include <LazyEngine/LazyEngine.h>
#include "../Mesh.h"
#include "Intersections.h"

namespace DMC {

	/**
	 *	Computes the barycentric coordinates of a given point on a triangle.
	 *	@param point: The point on the triangle
	 *	@param A, B, C: The triangle's corners
	 *	@returns the barycentric coordinates of the point
	 */
	HostDevice inline Intersection::BarycentricCoordinates computeBarycentricCoordinates(glm::vec3 point, Intersection::Triangle triangle) {
		glm::vec3 result = glm::vec3(0, 0, 0);

		glm::vec3 triangleNormal = triangle.computeNormal();

		float abc = glm::dot(triangleNormal, glm::cross(triangle.B - triangle.A, triangle.C - triangle.A));
		float pbc = glm::dot(triangleNormal, glm::cross(triangle.B - point, triangle.C - point));
		float pca = glm::dot(triangleNormal, glm::cross(triangle.C - point, triangle.A - point));

		result.x = pbc / abc;
		result.y = pca / abc;
		result.z = 1.f - result.x - result.y;

		return Intersection::BarycentricCoordinates(result);
	}

	/**
	 *	Projects a point straight down onto a 2D plane.
	 */
	HostDevice inline glm::vec3 projectPointOnPlane(glm::vec3 point, Intersection::Plane plane) {
		glm::vec3 tmp = plane.point - point;
		float distance = glm::dot(tmp, plane.normal);
		glm::vec3 result = point + distance * plane.normal;
		return result;
	}

	/**
	 *	Intersects a ray and a 2D plane.
	 */
	HostDevice inline glm::vec3 intersectRayPlane(LazyEngine::CameraRay ray, Intersection::Plane plane) {
		// check if the plane and ray are actually parallel, as there would be no intersection for that case
		float tmp = glm::dot(ray.getDirection(), plane.normal);
		if (-LazyEngine::Constants::EPSILON < tmp && tmp < LazyEngine::Constants::EPSILON) {
			// There is no intersection here
			return glm::vec3(NAN);
		}

		// Project the point straight onto the plane
		glm::vec3 pointOnPlane = projectPointOnPlane(ray.getOrigin(), plane);

		glm::vec3 originToPointOnPlane = pointOnPlane - ray.getOrigin();
		float fact = glm::length(originToPointOnPlane);
		float fact2 = glm::dot(glm::normalize(originToPointOnPlane), ray.getDirection());
		glm::vec3 result = ray.getOrigin() + ray.getDirection() * (fact / fact2);

		return result;
	}

	/**
	 *	Returns the intersection point of a triangle and a ray, or vec3(NAN), if there is none
	 */
	HostDevice inline glm::vec3 intersectRayTriangle(LazyEngine::CameraRay ray, Intersection::Triangle triangle) {
		// first, create the triangle's plane
		Intersection::Plane plane = triangle.computePlane();
		// now intersect the ray and the plane
		glm::vec3 intersection = intersectRayPlane(ray, plane);

		// get the intersection point's barycentric coordinates
		auto baryCoords = computeBarycentricCoordinates(intersection, triangle);
		if (baryCoords.isInside()) {
			return intersection;
		}
		else {
			// There is no intersection inside of the triangle!
			return glm::vec3(NAN);
		}
	}

	/**
	 *	Returns the intersection point of a quad and a ray, or vec3(NAN) if there is none
	 */
	HostDevice inline glm::vec3 intersectRayQuad(LazyEngine::CameraRay ray, Intersection::Quad quad) {
		// Check the first triangle of the quad for an intersection
		Intersection::Triangle tri1 = { quad.corners[0], quad.corners[1], quad.corners[2] };
		glm::vec3 intersection = intersectRayTriangle(ray, tri1);
		if (intersection.x == intersection.x) {
			// An intersection occured! Return the result early
			return intersection;
		}

		// No intersection occured, try the second triangle of the quad
		Intersection::Triangle tri2 = { quad.corners[2], quad.corners[3], quad.corners[0] };
		intersection = intersectRayTriangle(ray, tri2);
		// Whether there was an intersection or not doesn't need to be checked, as no intersection would return a vec3(NAN) anyways, just like this function.
		return intersection;
	}

	/**
	 *	Collides a sphere and a ray.
	 *	@param position: the position of the sphere's center
	 *	@param radius: The radius of the sphere
	 *	@param ray: The ray to intersect with
	 */
	HostDevice
	inline CollisionRayData collideSphereRay(glm::vec3 position, float radius, LazyEngine::CameraRay ray) {
		CollisionRayData data;

		glm::vec3 z = ray.getOrigin() - position;
		data.distanceToCenter = glm::length(z);
		float dot1 = glm::dot(ray.getDirection(), z);
		float Q = dot1 * dot1 - glm::dot(z, z) + radius * radius;
		if (Q > 0.f) {
			data.distance = -sqrt(Q);
			// collision! compute intersection points
		}
		else {
			data.distance = sqrt(Q);
		}

		return data;
	}

}