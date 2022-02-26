#include "DeviceVectors.h"

#include <cuda_runtime.h>


// ######################################################################### //
// ### CUDA Kernels ######################################################## //
// ######################################################################### //

__global__ void copyVertices(LazyEngine::DataView<glm::vec3> positions, LazyEngine::DataView<glm::vec3> normals, LazyEngine::DataView<DMC::Vertex> output, IndexType* vectorSize, const glm::vec4 color) {
	LazyEngine::CUDAInfo info;
	const unsigned int threadId = info.getGlobalThreadId();
	if (threadId >= *vectorSize) return;

	DMC::Vertex result;
	result.position = positions[threadId];
	result.normal = normals[threadId];
	result.color = color;

	output[threadId] = result;
	//LazyEngine::quickCopy<DMC::Vertex, int2>(&result, &output[threadId]);
}

struct DoubleTriangle {
	uint32_t triangle1[3];
	uint32_t triangle2[3];
};

__global__ void copyIndices(LazyEngine::DataView<glm::ivec4> quads, LazyEngine::DataView<uint32_t> output, IndexType* vectorSize) {
	LazyEngine::CUDAInfo info;
	const unsigned int threadId = info.getGlobalThreadId();
	if (threadId >= *vectorSize) return;

	// convert the quadrilateral into two triangles
	const glm::ivec4 quad = quads[threadId];
#if 0
	DoubleTriangle trianglesIndices;
	// Triangle 1
	int i = 0;
	trianglesIndices.triangle1[i++] = quad.x;
	trianglesIndices.triangle1[i++] = quad.y;
	trianglesIndices.triangle1[i++] = quad.z;
	// Triangle 2
	i = 0;
	trianglesIndices.triangle2[i++] = quad.z;
	trianglesIndices.triangle2[i++] = quad.w;
	trianglesIndices.triangle2[i++] = quad.x;
#endif
	// copy two indices at a time (pretty HACKy) for a total of 3 instead of 6 copies
	//LazyEngine::quickCopy<DoubleTriangle, int2>(&trianglesIndices, (DoubleTriangle*)&output[threadId * 6]);
	output[threadId * 6] = quad.x;
	output[threadId * 6 + 1] = quad.y;
	output[threadId * 6 + 2] = quad.z;
	output[threadId * 6 + 3] = quad.z;
	output[threadId * 6 + 4] = quad.w;
	output[threadId * 6 + 5] = quad.x;
}

namespace DMC {
	
	HostVector::HostVector() 
		: m_size(nullptr)
		, d_size(nullptr)
	{
		// allocate memory for the size and initialize it with 0
		cudaMalloc(&d_size, sizeof(IndexType));
		cudaMemset(d_size, 0, sizeof(IndexType));

		// set up freeing the memory on destruction
		m_size = std::shared_ptr<IndexType>(d_size, cudaFree);
	}

	HostVector::~HostVector() {
		m_size = nullptr;
		d_size = nullptr;
	}

	IndexType HostVector::size() const {
		// copy the device-stored size to the host
		IndexType result;
		cudaMemcpy(&result, d_size, sizeof(IndexType), cudaMemcpyDeviceToHost);
		return result;
	}

	void HostVector::resetSize() {
		cudaMemset(d_size, 0, sizeof(IndexType));
	}

	void HostVector::copyTo(HostVector& other) const {
		cudaMemcpy(other.d_size, d_size, sizeof(IndexType), cudaMemcpyDeviceToDevice);
	}


	// ##################################################################### //
	// ### AtomicCounter ################################################### //
	// ##################################################################### //

	AtomicCounterHost::AtomicCounterHost(IndexType initialValue)
		: m_counter(nullptr)
		, d_counter(nullptr)
		, m_deviceCounter(nullptr)
	{
		// allocate memory for the size and initialize it with 0
		cudaMalloc(&d_counter, sizeof(IndexType));
		reset(initialValue);

		// set up freeing the memory on destruction
		m_counter = std::shared_ptr<IndexType>(d_counter, cudaFree);

		// create the device counterpart
		m_deviceCounter = std::unique_ptr<AtomicCounter>(new AtomicCounter(d_counter));
	}

	void AtomicCounterHost::reset(IndexType resetValue) {
		cudaMemset(d_counter, resetValue, sizeof(IndexType));
	}

	AtomicCounterHost::operator AtomicCounter() {
		assert(m_deviceCounter != nullptr);
		return *m_deviceCounter;
	}

	// ##################################################################### //
	// ### VertexDeviceVector ############################################## //
	// ##################################################################### //

	VertexDeviceVector::VertexDeviceVector(const thrust::device_vector<glm::vec3>& positions, const thrust::device_vector<glm::vec3>& normals, IndexType* size)
		: DeviceVector(size, positions.size())
		, m_vertexPositions(positions)
		, m_vertexNormals(normals)
	{
		// empty
	}

	// ##################################################################### //

	VertexHostVector::VertexHostVector(IndexType size)
		: HostVector()
		, d_vertexPositions(size, "VertexHostVector-VertexPositions")
		, d_vertexNormals(size, "VertexHostVector-VertexNormals")
		, m_deviceVector(nullptr)
	{
		clear();
		m_deviceVector = std::unique_ptr<VertexDeviceVector>(new VertexDeviceVector(d_vertexPositions, d_vertexNormals, d_size));
	}

	void VertexHostVector::copyToVertexBuffer(LazyEngine::DataView<Vertex> vertexBuffer, const glm::vec4& color) {
		copyVertices ToCudaArgs(vertexBuffer) (d_vertexPositions, d_vertexNormals, vertexBuffer, d_size, color);
	}

	void VertexHostVector::clear() {
		const glm::vec3 initialValue = glm::vec3(0.f, 0.f, 0.f);
		thrust::fill(d_vertexPositions.begin(), d_vertexPositions.end(), initialValue);
		thrust::fill(d_vertexNormals.begin(), d_vertexNormals.end(), initialValue);
		resetSize();
	}

	void VertexHostVector::copyTo(VertexHostVector& other) const {
		assert(size() <= other.d_vertexPositions.size());
		assert(size() <= other.d_vertexNormals.size());
		
		// copy vertex positions and normals
		thrust::copy(d_vertexPositions.begin(), d_vertexPositions.end(), other.d_vertexPositions.begin());
		thrust::copy(d_vertexNormals.begin(), d_vertexNormals.end(), other.d_vertexNormals.begin());
		
		// copy vector size
		HostVector::copyTo(other);
	}


	// ##################################################################### //
	// ### QuadrilateralDeviceVector ####################################### //
	// ##################################################################### //

	QuadrilateralDeviceVector::QuadrilateralDeviceVector(const thrust::device_vector<glm::ivec4> & quads, const thrust::device_vector<QuadrilateralAttribute>& attributes, IndexType* size)
		: DeviceVector(size, quads.size())
		, m_quads(quads)
		, m_attributes(attributes)
	{
		// empty
	}

	// ##################################################################### //

	QuadrilateralHostVector::QuadrilateralHostVector(IndexType size)
		: HostVector()
		, d_quadrilaterals(size, "QuadrilateralHostVector-Quadrilaterals")
		, d_attributes(size, "QuadrilateralHostVector-Attributes")
		, m_deviceVector(nullptr)
	{
		clear();
		m_deviceVector = std::unique_ptr<QuadrilateralDeviceVector>(new QuadrilateralDeviceVector(d_quadrilaterals, d_attributes, d_size));
	}

	QuadrilateralHostVector::~QuadrilateralHostVector() {
	}

	void QuadrilateralHostVector::copyToIndexBuffer(LazyEngine::DataView<uint32_t> indexBuffer) {
		// cast quadrilaterals to a DataView
		LazyEngine::DataView<glm::ivec4> quads = d_quadrilaterals;
		copyIndices ToCudaArgs(quads) (quads, indexBuffer, d_size);
	}

	void QuadrilateralHostVector::clear() {
		thrust::fill(d_quadrilaterals.begin(), d_quadrilaterals.end(), glm::ivec4(-1, -1, -1, -1));
		thrust::fill(d_attributes.begin(), d_attributes.end(), QuadrilateralAttribute());
		resetSize();
	}

	void QuadrilateralHostVector::copyTo(QuadrilateralHostVector& other) const {
		assert(size() <= other.d_quadrilaterals.size());
		assert(size() <= other.d_attributes.size());

		// copy quads and attributes
		thrust::copy(d_quadrilaterals.begin(), d_quadrilaterals.end(), other.d_quadrilaterals.begin());
		thrust::copy(d_attributes.begin(), d_attributes.end(), other.d_attributes.begin());

		// copy vector size
		HostVector::copyTo(other);
	}


}