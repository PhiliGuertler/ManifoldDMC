#include "HalfedgeVectors.h"



namespace DMC {

	// ##################################################################### //
	// ### Halfedge ######################################################## //
	// ##################################################################### //

	HalfedgeDeviceVector::HalfedgeDeviceVector(const thrust::device_vector<Halfedge>& halfedges, IndexType* size)
		: DeviceVector(size, halfedges.size())
		, m_halfedges(halfedges)
	{
		// empty
	}


	HalfedgeHostVector::HalfedgeHostVector(IndexType maxCapacity)
		: HostVector()
		, d_halfedges(maxCapacity, "HalfedgeHostVector-Halfedges")
		, m_deviceVector(nullptr)
	{
		clear();
		m_deviceVector = std::unique_ptr<HalfedgeDeviceVector>(new HalfedgeDeviceVector(d_halfedges, d_size));
	}

	void HalfedgeHostVector::clear() {
		const Halfedge invalidHalfedge;
		thrust::fill(d_halfedges.begin(), d_halfedges.end(), invalidHalfedge);
		resetSize();
	}

	void HalfedgeHostVector::resize(IndexType numElements) {
		d_halfedges.resize(numElements);
		m_deviceVector.reset(new HalfedgeDeviceVector(d_halfedges, d_size));
	}


	// ##################################################################### //
	// ### HalfedgeVertex ################################################## //
	// ##################################################################### //

	HalfedgeVertexDeviceVector::HalfedgeVertexDeviceVector(const thrust::device_vector<VertexID>& vertices, const thrust::device_vector<Flags>& flags, IndexType* size)
		: DeviceVector(size, vertices.size())
		, m_vertices(vertices)
		, m_flags(flags)
	{
		// empty
	}

	HalfedgeVertexHostVector::HalfedgeVertexHostVector(IndexType maxCapacity)
		: HostVector()
		, d_vertices(maxCapacity, "HalfedgeVertices-Vertices")
		, d_flags(maxCapacity, "HalfedgeVertices-Flags")
		, m_deviceVector(nullptr)
	{
		clear();
		m_deviceVector = std::unique_ptr<HalfedgeVertexDeviceVector>(new HalfedgeVertexDeviceVector(d_vertices, d_flags, d_size));
	}

	void HalfedgeVertexHostVector::clear() {
		thrust::fill(d_vertices.begin(), d_vertices.end(), DMC::INVALID_INDEX);
		thrust::fill(d_flags.begin(), d_flags.end(), Flags());
		resetSize();
	}

	void HalfedgeVertexHostVector::resize(IndexType numElements) {
		d_vertices.resize(numElements);
		d_flags.resize(numElements);
		m_deviceVector.reset(new HalfedgeVertexDeviceVector(d_vertices, d_flags, d_size));
	}

	// ##################################################################### //
	// ### HalfedgeFace #################################################### //
	// ##################################################################### //

	HalfedgeFaceDeviceVector::HalfedgeFaceDeviceVector(const thrust::device_vector<HalfedgeFace>& attributes, const thrust::device_vector<HalfedgeID>& firstHalfedges, IndexType* size)
		: DeviceVector(size, attributes.size())
		, m_attributes(attributes)
		, m_firstHalfedgeIDs(firstHalfedges)
	{
		// empty
	}


	HalfedgeFaceHostVector::HalfedgeFaceHostVector(IndexType maxCapacity)
		: HostVector()
		, d_attributes(maxCapacity, "HalfedgeFaces-Attributes")
		, d_firstHalfedgeIDs(maxCapacity, "HalfedgeFaces-FirstHalfedgeIDs")
		, m_deviceVector(nullptr)
	{
		clear();
		m_deviceVector = std::unique_ptr<HalfedgeFaceDeviceVector>(new HalfedgeFaceDeviceVector(d_attributes, d_firstHalfedgeIDs, d_size));
	}

	void HalfedgeFaceHostVector::clear() {
		const HalfedgeFace invalidAttributes;
		thrust::fill(d_attributes.begin(), d_attributes.end(), invalidAttributes);
		thrust::fill(d_firstHalfedgeIDs.begin(), d_firstHalfedgeIDs.end(), DMC::INVALID_INDEX);
		resetSize();
	}

	void HalfedgeFaceHostVector::resize(IndexType numElements) {
		d_attributes.resize(numElements);
		d_firstHalfedgeIDs.resize(numElements);
		m_deviceVector.reset(new HalfedgeFaceDeviceVector(d_attributes, d_firstHalfedgeIDs, d_size));
	}


	// ##################################################################### //
	// ### VertexMap ####################################################### //
	// ##################################################################### //

	VertexMapDevice::VertexMapDevice(
		const thrust::device_vector<int>& vertexValence,
		const thrust::device_vector<int>& elementCount,
		const thrust::device_vector<VertexType>& vertexType,
		const thrust::device_vector<VertexID>& mappingTarget,
		const thrust::device_vector<int>& mappingAddress)
		: m_vertexValence(vertexValence)
		, m_elementCount(elementCount)
		, m_vertexType(vertexType)
		, m_mappingTarget(mappingTarget)
		, m_mappingAddress(mappingAddress)
	{
		// empty
	}


	VertexMapHost::VertexMapHost(IndexType size)
		: d_vertexValence(size, "VertexMapHost-vertexValence")
		, d_elementCount(size, "VertexMapHost-elementCount")
		, d_vertexType(size, "VertexMapHost-vertexType")
		, d_mappingTarget(size, "VertexMapHost-mappingTarget")
		, d_mappingAddress(size, "VertexMapHost-mappingAddress")
		, m_deviceVector(nullptr)
	{
		initialize();
	}

	void VertexMapHost::initialize() {
		thrust::fill(d_vertexValence.begin(), d_vertexValence.end(), 0);
		thrust::fill(d_elementCount.begin(), d_elementCount.end(), 0);
		thrust::fill(d_vertexType.begin(), d_vertexType.end(), VertexType::Neutral);
		thrust::fill(d_mappingTarget.begin(), d_mappingTarget.end(), INVALID_INDEX);
		thrust::fill(d_mappingAddress.begin(), d_mappingAddress.end(), INVALID_INDEX);

		m_deviceVector = std::unique_ptr<VertexMapDevice>(new VertexMapDevice(
			d_vertexValence,
			d_elementCount,
			d_vertexType,
			d_mappingTarget,
			d_mappingAddress
		));
	}

}