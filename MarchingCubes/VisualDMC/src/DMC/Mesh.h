#pragma once

#include <LazyEngine/LazyEngine.h>

#include "Utils/Utilities.h"

namespace DMC {
	class VertexHostVector;
	class HalfedgeHostVector;
	class HalfedgeFaceHostVector;

	/**
	 *	Defines a single vertex of the mesh
	 */
	struct Vertex {
		glm::vec3 position = glm::vec3(NAN);	// corresponds to "Vertex"
		glm::vec3 normal = glm::vec3(NAN);	// corresponds to "Normal"
		glm::vec4 color = glm::vec4(1.f, 0.f, 1.f, 1.f);	// corresponds to "Attribute"
	};

	/**
	 *	Defines the bounding box of the mesh
	 */
	struct BoundingBox {
		glm::vec3 min = glm::vec3(0.f);
		glm::vec3 max = glm::vec3(0.f);

		std::array<glm::vec3, 8> getCorners() const;
	};

	/**
	 *	Defines a mesh
	 */
	class Mesh : public LazyEngine::Object3D {
	public:
		__host__ __device__
			inline static glm::vec3 modelToWorld(const glm::vec3& modelVertex, const glm::mat4& modelToWorld) {
			glm::vec4 tmpVertex = modelToWorld * glm::vec4(modelVertex, 1.f);
			return glm::vec3(tmpVertex) / tmpVertex.w;
		}

	public:
		Mesh();
		virtual ~Mesh();

		void render();

		void renderWireframeTriangles(bool invertColors = false);

		void renderWireframeQuads(bool invertColors = false);

		void renderPoints();

		void renderSinglePoints(const std::vector<uint32_t>& indices);

		void renderHighlightPoints();
		void renderHighlightWireframeQuads();
		void renderHighlightSolid();
		void renderHighlightLines();
		void setHighlightIndices(const std::vector<uint32_t>& indices);
		void setHighlightIndices(LazyEngine::Ref<LazyEngine::IndexBuffer> indices);

		// --- DMC's functions --- //
		/**
		 *	Resizes the vertex buffer.
		 *	This cannot be done while the buffer is mapped into CUDA's memory space!
		 */
		void resizeVertices(size_t size);
		/**
		 *	Resizes the index buffer.
		 *  THis cannot be done while the buffer is mapped into CUDA's memory space!
		 */
		void resizeIndices(size_t size);

		void resize(size_t vertexSizeInBytes, size_t numIndices);

		void addVertex(uint32_t index, const glm::vec3& position, const glm::vec3& normal, const glm::vec4& color = {250.f/255.f, 250.f/255.f, 250.f/255.f, 1.f});
		void addVertex(const glm::vec3& position, const glm::vec3& normal, const glm::vec4& color = {202.f/255.f,200.f/255.f,201.f/255.f,1.f});

		void flipNormals();

		void updateBoundingBox();

		inline LazyEngine::InteroperableOpenGLVertexBuffer& getVertices() {
			return *m_vertices;
		}

		inline LazyEngine::InteroperableOpenGLIndexBuffer& getIndices() {
			return *m_triangleIndices;
		}

		inline LazyEngine::Ref<LazyEngine::ColoredSphereBillboardMaterial>& getVertexMaterial() {
			return m_vertexMaterial;
		}

		inline LazyEngine::Ref<LazyEngine::ColoredLineBillboardMaterial>& getLineMaterial() {
			return m_lineMaterial;
		}

		inline LazyEngine::Ref<LazyEngine::PhongMaterial>& getFaceMaterial() {
			return m_faceMaterial;
		}
		
		inline glm::vec3 modelToWorld(const glm::vec3& modelVertex) const {
			return modelToWorld(modelVertex, getModelToWorld());
		}

		void fromHalfedgeBuffers(VertexHostVector& vertices, HalfedgeHostVector& halfedges, HalfedgeFaceHostVector& faces);

		void setAlphaValue(float alpha);

		void writeToFile(const std::string& filePath);

		void setDefaultQuad();
	protected:
		// contains both vertices and normals
		LazyEngine::Ref<LazyEngine::VertexArray> m_vertexArray;
		LazyEngine::Ref<LazyEngine::InteroperableOpenGLVertexBuffer> m_vertices;
		LazyEngine::Ref<LazyEngine::InteroperableOpenGLIndexBuffer> m_triangleIndices;

		// Selection IndexBuffers
		LazyEngine::Ref<LazyEngine::IndexBuffer> m_selectedVertexIndices;
		LazyEngine::Ref<LazyEngine::IndexBuffer> m_selectedHalfedges;

		LazyEngine::Ref<LazyEngine::ColoredPhongMaterial> m_material;
		LazyEngine::Ref<LazyEngine::ColoredSphereBillboardMaterial> m_vertexMaterial;
		LazyEngine::Ref<LazyEngine::ColoredLineBillboardMaterial> m_lineMaterial;
		LazyEngine::Ref<LazyEngine::PhongMaterial> m_faceMaterial;

		std::shared_ptr<BufferInfo> m_vertexBufferInfo;
		std::shared_ptr<BufferInfo> m_indexBufferInfo;

		BoundingBox m_boundingBox;
	};

}