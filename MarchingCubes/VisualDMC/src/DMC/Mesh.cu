#include "Mesh.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>

#include "DeviceVectors.h"
#include "Halfedges/HalfedgeVectors.h"


namespace DMC {

	// ##################################################################### //
	// ### CUDA kernels #################################################### //
	// ##################################################################### //

	__global__ void flipNormalsOf(LazyEngine::DataView<Vertex> vertices) {
		LazyEngine::CUDAInfo info;

		unsigned int id = info.getGlobalThreadId();

		if (id >= vertices.size()) return;

		vertices[id].normal = -vertices[id].normal;
	}


	// ##################################################################### //
	// ### BoundingBox ##################################################### //
	// ##################################################################### //

	std::array<glm::vec3, 8> BoundingBox::getCorners() const {
		std::array<glm::vec3, 8> result;
		result[0] = { min.x, min.y, min.z };
		result[1] = { max.x, min.y, min.z };
		result[2] = { min.x, max.y, min.z };
		result[3] = { max.x, max.y, min.z };
		result[4] = { min.x, min.y, max.z };
		result[5] = { max.x, min.y, max.z };
		result[6] = { min.x, max.y, max.z };
		result[7] = { max.x, max.y, max.z };

		return result;
	}


	// ##################################################################### //
	// ### Mesh ############################################################ //
	// ##################################################################### //

	Mesh::Mesh()
		: m_vertexArray(nullptr)
		, m_vertices(nullptr)
		, m_triangleIndices(nullptr)
		, m_selectedVertexIndices(nullptr)
		, m_boundingBox()
		, m_material(LazyEngine::createRef<LazyEngine::ColoredPhongMaterial>())
		, m_vertexMaterial(LazyEngine::createRef<LazyEngine::ColoredSphereBillboardMaterial>())
		, m_lineMaterial(LazyEngine::createRef<LazyEngine::ColoredLineBillboardMaterial>())
		, m_faceMaterial(LazyEngine::createRef<LazyEngine::PhongMaterial>())
		, m_vertexBufferInfo(nullptr)
		, m_indexBufferInfo(nullptr)
	{
		setScale(glm::vec3(10.f));

		// create the vertex array that contains the vertex and index buffers
		m_vertexArray = LazyEngine::VertexArray::create();

		setDefaultQuad();

		// setup the layout of the vertex buffer to match that of "DMC::Vertex"
		LazyEngine::BufferLayout layout = {
			{LazyEngine::ShaderDataType::Float3, "a_position"},
			{LazyEngine::ShaderDataType::Float3, "a_normal"},
			{LazyEngine::ShaderDataType::Float4, "a_color"},
		};
		m_vertices->setLayout(layout);

		// register the vertex and index buffer in the vertex array
		m_vertexArray->addVertexBuffer(m_vertices);
		m_vertexArray->setIndexBuffer(m_triangleIndices);

		m_material->setSpecularFactor(0.2f);
		m_material->setAmbientFactor(0.8f);
		m_material->setSpecularExponent(12);

		m_vertexMaterial->setColor(glm::vec4(0.5f, 1.f, 0.3f, 1.f));
		m_vertexMaterial->setZOffset(0.01f);
		m_lineMaterial->setColor(glm::vec4(0.5f, 1.f, 0.3f, 1.f));
		m_lineMaterial->setZOffset(0.01f);
		m_faceMaterial->setColor(glm::vec4(0.3f, 1.f, 0.3f, 1.f));
		m_faceMaterial->setZOffset(0.01f);
	}

	void Mesh::setDefaultQuad() {
		setScale(glm::vec3(10.f));

		Vertex testVertices[4] = {
			{glm::vec3(-1, -1, 0), glm::vec3(0,0,1), glm::vec4(1,0,0,1)},
			{glm::vec3(1, -1, 0), glm::vec3(0,0,1), glm::vec4(0,1,0,1)},
			{glm::vec3(1, 1, 0), glm::vec3(0,0,1), glm::vec4(0,0,1,1)},
			{glm::vec3(-1, 1, 0), glm::vec3(0,0,1), glm::vec4(1,1,0,1)},
		};
		uint32_t testIndices[6] = {
			0,1,2,
			2,3,0,
		};

		// create the vertex buffer
		if (m_vertexBufferInfo != nullptr) {
			MemoryMonitor::getInstance().unregisterBuffer(m_vertexBufferInfo);
		}
		auto vertexbuffer = LazyEngine::VertexBuffer::create((float*)testVertices, sizeof(Vertex) * 4);
		m_vertexBufferInfo = MemoryMonitor::getInstance().registerBuffer({
			4, sizeof(Vertex), "VertexBuffer"
			});
		m_vertices = std::dynamic_pointer_cast<LazyEngine::InteroperableOpenGLVertexBuffer>(vertexbuffer);

		// create the index buffer
		if (m_indexBufferInfo != nullptr) {
			MemoryMonitor::getInstance().unregisterBuffer(m_indexBufferInfo);
		}
		auto indexbuffer = LazyEngine::IndexBuffer::create(testIndices, 6);
		m_indexBufferInfo = MemoryMonitor::getInstance().registerBuffer({
			6, sizeof(uint32_t), "IndexBuffer"
			});
		m_triangleIndices = std::dynamic_pointer_cast<LazyEngine::InteroperableOpenGLIndexBuffer>(indexbuffer);

		// create the vertex array that contains the vertex and index buffers
		m_vertexArray = LazyEngine::VertexArray::create();

		// setup the layout of the vertex buffer to match that of "DMC::Vertex"
		LazyEngine::BufferLayout layout = {
			{LazyEngine::ShaderDataType::Float3, "a_position"},
			{LazyEngine::ShaderDataType::Float3, "a_normal"},
			{LazyEngine::ShaderDataType::Float4, "a_color"},
		};
		m_vertices->setLayout(layout);

		// register the vertex and index buffer in the vertex array
		m_vertexArray->addVertexBuffer(m_vertices);
		m_vertexArray->setIndexBuffer(m_triangleIndices);

		// create the index buffer for the selected vertices
		m_selectedVertexIndices = LazyEngine::IndexBuffer::create(nullptr, 1);
	}

	Mesh::~Mesh() {
		MemoryMonitor::getInstance().unregisterBuffer(m_vertexBufferInfo);
		MemoryMonitor::getInstance().unregisterBuffer(m_indexBufferInfo);
	}

	void Mesh::resize(size_t vertexSizeInBytes, size_t numIndices) {
		// Reset all pointers
		m_vertices = nullptr;
		m_triangleIndices = nullptr;
		m_vertexArray = nullptr;

		m_selectedVertexIndices = nullptr;

		// create the vertex buffer
		MemoryMonitor::getInstance().unregisterBuffer(m_vertexBufferInfo);
		auto vertexbuffer = LazyEngine::VertexBuffer::create(nullptr, vertexSizeInBytes);
		m_vertexBufferInfo = MemoryMonitor::getInstance().registerBuffer({
			vertexSizeInBytes / sizeof(Vertex), sizeof(Vertex), "VertexBuffer"
		});
		m_vertices = std::dynamic_pointer_cast<LazyEngine::InteroperableOpenGLVertexBuffer>(vertexbuffer);
		// create the index buffer
		MemoryMonitor::getInstance().unregisterBuffer(m_indexBufferInfo);
		auto indexbuffer = LazyEngine::IndexBuffer::create(nullptr, numIndices);
		m_indexBufferInfo = MemoryMonitor::getInstance().registerBuffer({
			numIndices, sizeof(uint32_t), "IndexBuffer"
		});
		m_triangleIndices = std::dynamic_pointer_cast<LazyEngine::InteroperableOpenGLIndexBuffer>(indexbuffer);

		// create the vertex array that contains the vertex and index buffers
		m_vertexArray = LazyEngine::VertexArray::create();

		// setup the layout of the vertex buffer to match that of "DMC::Vertex"
		LazyEngine::BufferLayout layout = {
			{LazyEngine::ShaderDataType::Float3, "a_position"},
			{LazyEngine::ShaderDataType::Float3, "a_normal"},
			{LazyEngine::ShaderDataType::Float4, "a_color"},
		};
		m_vertices->setLayout(layout);

		// register the vertex and index buffer in the vertex array
		m_vertexArray->addVertexBuffer(m_vertices);
		m_vertexArray->setIndexBuffer(m_triangleIndices);

		// create the index buffer for the selected vertices
		m_selectedVertexIndices = LazyEngine::IndexBuffer::create(nullptr, 1);
	}

	void Mesh::setAlphaValue(float alpha) {
		m_material->setColor(glm::vec4(alpha));
	}

	void Mesh::render() {
		m_material->updateUniforms();
		m_material->getShader()->bind();
		m_vertexArray->setIndexBuffer(m_triangleIndices);
		LazyEngine::Renderer::submit(m_vertexArray, m_material->getShader(), getModelToWorld());
	}

	void Mesh::renderWireframeTriangles(bool invertColors) {
		m_material->updateUniforms();
		m_material->getTriangleOutlineShader()->bind();
		m_material->getTriangleOutlineShader()->uniformBool("u_invertColors", invertColors);
		m_vertexArray->setIndexBuffer(m_triangleIndices);
		LazyEngine::Renderer::submit(m_vertexArray, m_material->getTriangleOutlineShader(), getModelToWorld());
	}

	void Mesh::renderWireframeQuads(bool invertColors) {
		m_material->updateUniforms();
		m_material->getQuadOutlineShader()->bind();
		m_material->getQuadOutlineShader()->uniformBool("u_invertColors", invertColors);
		m_vertexArray->setIndexBuffer(m_triangleIndices);
		LazyEngine::Renderer::submit(m_vertexArray, m_material->getQuadOutlineShader(), getModelToWorld());
	}

	void Mesh::renderPoints() {
		m_vertexMaterial->updateUniforms();

		m_vertexArray->setIndexBuffer(m_triangleIndices);
		LazyEngine::Renderer::submitPoints(m_vertexArray, m_vertexMaterial->getShader(), getModelToWorld());
	}

	void Mesh::renderSinglePoints(const std::vector<uint32_t>& indices){
		if (indices.size() < 1) return;

		setHighlightIndices(indices);
		renderHighlightPoints();
	}

	void Mesh::renderHighlightPoints() {
		if (m_selectedVertexIndices == nullptr || m_selectedVertexIndices->getCount() < 1) return;

		m_vertexMaterial->updateUniforms();

		m_vertexArray->setIndexBuffer(m_selectedVertexIndices);
		LazyEngine::Renderer::submitPoints(m_vertexArray, m_vertexMaterial->getShader(), getModelToWorld());
	}

	void Mesh::renderHighlightWireframeQuads() {
		throw std::exception("renderHighlightWireframeQuads is not implemented!");
	}

	void Mesh::renderHighlightSolid() {
		if (m_selectedVertexIndices == nullptr || m_selectedVertexIndices->getCount() < 1) return;

		m_faceMaterial->updateUniforms();
		m_faceMaterial->getShader()->bind();

		m_vertexArray->setIndexBuffer(m_selectedVertexIndices);
		LazyEngine::Renderer::submit(m_vertexArray, m_faceMaterial->getShader(), getModelToWorld());
	}

	void Mesh::renderHighlightLines() {
		if (m_selectedVertexIndices == nullptr || m_selectedVertexIndices->getCount() < 1) return;

		m_lineMaterial->updateUniforms();
		m_lineMaterial->getShader()->bind();

		m_vertexArray->setIndexBuffer(m_selectedVertexIndices);
		LazyEngine::Renderer::submitLines(m_vertexArray, m_lineMaterial->getShader(), getModelToWorld());
	}

	void Mesh::setHighlightIndices(const std::vector<uint32_t>& indices) {
		m_selectedVertexIndices = nullptr;
		if (indices.size() > 0) {
			m_selectedVertexIndices = LazyEngine::IndexBuffer::create(const_cast<uint32_t*>(indices.data()), indices.size());
		}
	}

	void Mesh::setHighlightIndices(LazyEngine::Ref<LazyEngine::IndexBuffer> indices) {
		m_selectedVertexIndices = indices;
	}

	void Mesh::resizeVertices(size_t size) {
		throw std::exception("resizeVertices is not working! Use resize() instead!");
		m_vertices->resize(size * sizeof(DMC::Vertex));
		m_vertexArray = LazyEngine::VertexArray::create();
		m_vertexArray->addVertexBuffer(m_vertices);
		m_vertexArray->setIndexBuffer(m_triangleIndices);
	}

	void Mesh::resizeIndices(size_t size) {
		throw std::exception("resizeIndices is not working! Use resize() instead!");
		m_triangleIndices->resize(size);
		m_vertexArray = LazyEngine::VertexArray::create();
		m_vertexArray->addVertexBuffer(m_vertices);
		m_vertexArray->setIndexBuffer(m_triangleIndices);
	}


	void Mesh::addVertex(uint32_t index, const glm::vec3& position, const glm::vec3& normal, const glm::vec4& color) {
		Vertex vertex;
		vertex.position = position;
		vertex.normal = normal;
		vertex.color = color;
		// upload the vertex
		float* vertexData = (float*)(&vertex);
		m_vertices->uploadData(vertexData, sizeof(Vertex), index * sizeof(Vertex));
	}

	void Mesh::addVertex(const glm::vec3& position, const glm::vec3& normal, const glm::vec4& color) {
		Vertex vertex;
		vertex.position = position;
		vertex.normal = normal;
		vertex.color = color;

		// resize the buffer
		uint32_t index = m_vertices->getSize() / sizeof(Vertex);
		m_vertices->resize((index + 1) * sizeof(Vertex));

		// upload the vertex
		float* vertexData = (float*)(&vertex);
		m_vertices->uploadData(vertexData, sizeof(Vertex), index * sizeof(Vertex));
	}

	void Mesh::flipNormals() {
		LazyEngine::ScopedCUDAInterop<Vertex> interop(*m_vertices);
		auto mapping = interop.getMapping();

		flipNormalsOf << <mapping.getNumBlocks(), mapping.getNumThreadsPerBlock() >> > (mapping);
	}

	struct CompareVertices {
		int index = 0;

		CompareVertices(int index) : index(index) { /*empty*/ }

		__host__ __device__
			bool operator()(const Vertex& a, const Vertex& b) {
			return a.position[index] < b.position[index];
		}
	};

	void Mesh::updateBoundingBox() {


		LazyEngine::ScopedCUDAInterop<Vertex> interop(*m_vertices);
		auto mapping = interop.getMapping();

		Vertex minX = *thrust::min_element(mapping.begin(), mapping.end(), CompareVertices(0));
		Vertex minY = *thrust::min_element(mapping.begin(), mapping.end(), CompareVertices(1));
		Vertex minZ = *thrust::min_element(mapping.begin(), mapping.end(), CompareVertices(2));
		m_boundingBox.min = { minX.position.x, minY.position.y, minZ.position.z };

		Vertex maxX = *thrust::max_element(mapping.begin(), mapping.end(), CompareVertices(0));
		Vertex maxY = *thrust::max_element(mapping.begin(), mapping.end(), CompareVertices(1));
		Vertex maxZ = *thrust::max_element(mapping.begin(), mapping.end(), CompareVertices(2));
		m_boundingBox.max = { maxX.position.x, maxY.position.y, maxZ.position.z };
	}



	__global__ void insertVertices(
		VertexDeviceVector vertices,
		HalfedgeFaceDeviceVector faces,
		HalfedgeDeviceVector halfedges,
		LazyEngine::DataView<Vertex> output
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= halfedges.size()) return;

		// get the halfedge originating at the current vertex
		Halfedge halfedge = halfedges[threadId];

		// get the halfedge's origin Vertex
		Vertex vertex;
		vertex.position = vertices.getPosition(halfedge.getOriginVertexID());
		vertex.normal = vertices.getNormal(halfedge.getOriginVertexID());
		// Default hue: Cyan
		float hue = 180.f;
		vertex.color = glm::vec4(LazyEngine::Color::HSVtoRGB(hue, 0.8f, 0.8f), 1.f);

		output[halfedge.getOriginVertexID()] = vertex;
	}

	__global__ void insertIndices(
		HalfedgeDeviceVector halfedges,
		HalfedgeFaceDeviceVector faces,
		LazyEngine::DataView<uint32_t> output
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();

		if (threadId >= faces.size()) return;

		// get the first halfedge of this face
		HalfedgeID firstHalfedge = faces.getFirstHalfedgeID(threadId);

		NeighborVertices vertices = halfedges.getFaceVertices(firstHalfedge);

		// Triangle 1
		int i = 0;
		output[6 * threadId + (i++)] = vertices.x;
		output[6 * threadId + (i++)] = vertices.y;
		output[6 * threadId + (i++)] = vertices.z;
		// Triangle 2
		output[6 * threadId + (i++)] = vertices.z;
		output[6 * threadId + (i++)] = vertices.w;
		output[6 * threadId + (i++)] = vertices.x;
	}

	void Mesh::fromHalfedgeBuffers(VertexHostVector& vertices, HalfedgeHostVector& halfedges, HalfedgeFaceHostVector& faces) {
		// resize buffers first
		size_t verticesBytes = vertices.size() * sizeof(Vertex);
		size_t numIndices = faces.size() * 6;
		resize(verticesBytes, numIndices);

		LazyEngine::ScopedCUDAInterop<Vertex> vertexInterop(getVertices());
		LazyEngine::ScopedCUDAInterop<uint32_t> indexInterop(getIndices());

		auto vertexBuffer = vertexInterop.getMapping();
		auto indexBuffer = indexInterop.getMapping();

		{
			// copy vertices with their respective colors into the vertex buffer
			auto dataView = halfedges.getDataView();
			insertVertices ToCudaArgs(dataView) (vertices, faces, halfedges, vertexBuffer);
		}

		{
			// copy halfedgeFaces into the index buffer
			auto dataView = faces.getDataView();
			insertIndices ToCudaArgs(dataView) (halfedges, faces, indexBuffer);

		}
	}

	inline bool endsWith(const std::string& string, const std::string& ending) {
		if (string.length() >= ending.length()) {
			return (string.compare(string.length() - ending.length(), ending.length(), ending) == 0);
		}
		return false;
	}

	void Mesh::writeToFile(const std::string& filePath) {
		// check if an ".obj" is already part of the filepath.
		// if not, append it.
		std::string extension = ".obj";
		std::string finalFilePath = filePath;
		if (!endsWith(filePath, extension)) {
			finalFilePath = filePath + extension;
		}

		// open the file for writing
		std::ofstream output(finalFilePath);

		if (!output.is_open()) {
			LAZYENGINE_ERROR("Could not open file \"{0}\" for writing", finalFilePath);
			return;
		}
		LAZYENGINE_INFO("Writing file \"{0}\"", finalFilePath);

		{
			// copy vertices to host
			int numVertices = getVertices().getSize();
			std::vector<Vertex> hostVertices(numVertices / sizeof(Vertex));

			LazyEngine::ScopedCUDAInterop<Vertex> vertexInterop(getVertices());
			auto vertexBuffer = vertexInterop.getMapping();
			thrust::copy(vertexBuffer.begin(), vertexBuffer.end(), hostVertices.begin());

			for (auto& vertex : hostVertices) {
				// write the vertex positions
				output << "v " << vertex.position[0] << " " << vertex.position[1] << " " << vertex.position[2] << std::endl;
			}
			for (auto& vertex : hostVertices) {
				// write the vertex normals
				output << "vn " << vertex.normal[0] << " " << vertex.normal[1] << " " << vertex.normal[2] << std::endl;
			}
		}

		{
			// copy indices to host
			int numIndices = getIndices().getCount();
			std::vector<uint32_t> hostIndices(numIndices);
			
			LazyEngine::ScopedCUDAInterop<uint32_t> indexInterop(getIndices());
			auto indexBuffer = indexInterop.getMapping();
			thrust::copy(indexBuffer.begin(), indexBuffer.end(), hostIndices.begin());

			for (int i = 0; i < hostIndices.size(); i += 6) {
				// write the quads
				glm::ivec4 quad = { -1,-1,-1,-1 };
				quad.x = hostIndices[i + 0];
				quad.y = hostIndices[i + 1];
				quad.z = hostIndices[i + 2];
				quad.w = hostIndices[i + 4];
				quad += 1;
				output << "f "
					<< quad.x << "//" << quad.x << " "
					<< quad.y << "//" << quad.y << " "
					<< quad.z << "//" << quad.z << " "
					<< quad.w << "//" << quad.w
					<< std::endl;
			}
		}
		output.close();
	}

}