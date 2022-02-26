#pragma once

// ######################################################################### //
// ### Buffer.h ############################################################ //
// ### Abstraction of General Buffers, like Vertex- or Indexbuffers and  ### //
// ### also Bufferlayouts without a specific Renderer API                ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"

namespace LazyEngine {

	/**
	 *	Lists all Kinds of Shader Datatypes in a Renderer API independent fashion
	 */
	enum class ShaderDataType {
		None = 0,
		Float,
		Float2,
		Float3,
		Float4,
		Mat3,
		Mat4,
		Uint,
		Int,
		Int2,
		Int3,
		Int4,
		Bool
	};

	/**
	 *	Returns The size in bytes of a given ShaderDataType
	 */
	static uint32_t ShaderDataTypeSize(ShaderDataType type) {
		switch (type) {
		case ShaderDataType::Float:		return 4; // 4 Bytes per float
		case ShaderDataType::Float2:	return 4 * 2;
		case ShaderDataType::Float3:	return 4 * 3;
		case ShaderDataType::Float4:	return 4 * 4;
		case ShaderDataType::Mat3:		return 4 * 3 * 3;
		case ShaderDataType::Mat4:		return 4 * 4 * 4;
		case ShaderDataType::Uint:		return 4; // 4 Bytes per uint
		case ShaderDataType::Int:		return 4; // 4 Bytes per int
		case ShaderDataType::Int2:		return 4 * 2;
		case ShaderDataType::Int3:		return 4 * 3;
		case ShaderDataType::Int4:		return 4 * 4;
		case ShaderDataType::Bool:		return 1; // FIXME: this might not be correct
		default: 
			LAZYENGINE_CORE_ASSERT(false, "Unknown ShaderDataType!");
			return 0;
		}
	}

	// ######################################################################## //
	// ### Experimental ####################################################### //
	// see buffer object documentation at khronos
	enum class BufferUsage {
		StaticDraw,		// write to the buffer, but don't read it. Data will be set only once.
		DynamicDraw,	// write to the buffer, but don't read it. Data will be set occasionally.
		StreamDraw,		// write to the buffer, but don't read it. Data will be set almost every use.
		StaticRead,		// don't write to the buffer, but read it. Data will be set only once.
		DynamicRead,	// don't write to the buffer, but read it. Data will be set occasionally.
		StreamRead,		// don't write to the buffer, but read it. Data will be set almost every use.
		StaticCopy,		// don't read or write to the buffer. Data will be set only once.
		DynamicCopy,	// don't read or write to the buffer. Data will be set occasionally.
		StreamCopy		// don't read or write to the buffer. Data will be set almost every use.
	};
	// ######################################################################## //

	/**
	 *	Defines an Element of a BufferLayout
	 *	TODO: Feature-Request: Create these automatically when parsing a shader
	 */
	struct BufferElement {
		// Name of the Element in the Shader
		std::string name;
		// Type of the Element in the Shader
		ShaderDataType type;
		// Offset inside Element in the VertexBuffer 
		uint32_t offset;
		// Size of the Element in Bytes
		uint32_t size;
		// True if normalized, false otherwise
		bool normalized;
		// ################################################################# //
		// ### Experimental ################################################ //
		// the location index inside the shader. 
		// TODO: this should be automatically extracted in the future
		int location;
		// specifies after how many instances this element should advance.
		// a value of 0 means that it advances for every vertex, 1 for every instance, 2 for every second instance...
		int instanceDivisor;
		// ################################################################# //

		/**
		 *	Constructor
		 */
		BufferElement(ShaderDataType type, const std::string& name, bool normalized = false /*experimental:*/, int location = 0, int instanceDivisor = 0/*/experimental*/)
			: name(name)
			, type(type)
			, size(ShaderDataTypeSize(type))
			, offset(0)
			, normalized(normalized)
			// ############################################################# //
			// ### Experimental ############################################ //
			, location(location)
			, instanceDivisor(instanceDivisor)
			// ############################################################# //
		{
			// Do nothing
		}

		/**
		 *	returns the number of components of the ShaderDataType of this element
		 */
		uint32_t getComponentCount() const {
			switch (type) {
			case ShaderDataType::Float:		return 1;
			case ShaderDataType::Float2:	return 2;
			case ShaderDataType::Float3:	return 3;
			case ShaderDataType::Float4:	return 4;
			case ShaderDataType::Mat3:		return 3 * 3;
			case ShaderDataType::Mat4:		return 4 * 4;
			case ShaderDataType::Uint:		return 1;
			case ShaderDataType::Int:		return 1;
			case ShaderDataType::Int2:		return 2;
			case ShaderDataType::Int3:		return 3;
			case ShaderDataType::Int4:		return 4;
			case ShaderDataType::Bool:		return 1;
			default:
				LAZYENGINE_CORE_ASSERT(false, "Unknown ShaderDataType!");
				return 0;
			}
		}

	};

	/**
	 *	Represents the Layout of a VertexBuffer.
	 *	Can be constructed in this way:
	 *	BufferLayout layout = {
	 *		{ ShaderDataType::Float3, "vertexPosition" },
	 *		{ ShaderDataType::Float3, "vertexNormal" },
	 *		{ ShaderDataType::Float4, "vertexColor" }
	 *	};
	 */
	class BufferLayout {
	public:
		// HACK: Allows a BufferLayout to be an uninitialized Member of a Class, for example
		BufferLayout() {}
		// KCAH

		/**
		 *	Constructor
		 *	@param elements: see class description on elements
		 */
		BufferLayout(const std::initializer_list<BufferElement>& elements)
			: m_elements(elements)
		{
			calculateOffsetsAndStride();
		}

		// iterators, which are used in for-each loops, like this for example:
		// for(const auto& element: bufferLayout) { ... }

		/**
		 *	returns a begin iterator to the elements of this layout
		 */
		std::vector<BufferElement>::iterator begin() { return m_elements.begin(); }
		/**
		 *	returns an end iterator to the elements of this layout
		 */
		std::vector<BufferElement>::iterator end() { return m_elements.end(); }

		/**
		 *	returns a begin const_iterator to the elements of this layout
		 */
		std::vector<BufferElement>::const_iterator begin() const { return m_elements.begin(); }
		/**
		 *	returns an end const_iterator to the elements of this layout
		 */
		std::vector<BufferElement>::const_iterator end() const { return m_elements.end(); }

		/**
		 *	getter for the elements-vector
		 */
		inline const std::vector<BufferElement>& getElements() const { return m_elements; }

		/**
		 *	returns stride of this layout which is effectively the sum 
		 *	of all element's length in bytes
		 */
		inline uint32_t getStride() const { return m_stride; }
	private:
		/**
		 *	computes offset and stride of each individual element of this BufferLayout
		 */
		inline void calculateOffsetsAndStride() {
			m_stride = 0;
			for (auto& element : m_elements) {
				element.offset = m_stride;
				m_stride += element.size;
			}
		}

	private:
		// a list of elements inside this layout
		std::vector<BufferElement> m_elements;
		// the stride of a single VertexBuffer-Element, which consists of one
		// of each element from m_elements.
		uint32_t m_stride = 0;
	};

	/**
	 *	Wraps a Rendering-API specific VertexBuffer.
	 *	FIXME: Currently there is no way to alter the contents of the buffer.
	 *		Also the vertices are only in the gpu memory, not on cpu side.
	 */
	class VertexBuffer {
	public:
		// there is no default constructor, because a vertexbuffer should be
		// created via "create()".

		/**
		 *	default destructor
		 */
		virtual ~VertexBuffer() = default;

		/**
		 *	Binds the buffer. This must be called before uploading any data.
		 */
		virtual void bind() const = 0;
		/**
		 *	unbinds the buffer
		 */
		virtual void unbind() const = 0;

		/**
		 *	set the layout of this specific vertexbuffer, e.g. 3 floats position, then 4 floats color.
		 *	@param layout: the actual layout
		 */
		virtual void setLayout(const BufferLayout& layout) = 0;

		/**
		 *	returns the buffer layout that can be modified afterwards
		 */
		virtual BufferLayout& getLayout() = 0;

		/**
		 *	returns the size of this buffer's content in bytes
		 */
		virtual uint32_t getSize() const = 0;
		
		/**
		 *	resizes the buffer on the gpu
		 *	@param size: the new size of the buffer in bytes
		 */
		virtual void resize(uint32_t size) = 0;

		/**
		 *	sets the buffer usage of this buffer
		 *	@param usage: the new usage of this buffer
		 */
		virtual void setBufferUsage(BufferUsage usage) = 0;

		/**
		 *	uploads data to the gpu.
		 *	@param data: a pointer to the data to be uploaded
		 *	@param size: the size of the data in bytes
		 *	@param offset: an offset (in elements) at which the data should be inserted
		 *	@param usage: the usage of the buffer, e.g. for meshes this might be
		 *		StaticDraw, but particles will be better off with DynamicDraw.
		 */
		virtual void uploadData(float *data, uint32_t size, uint32_t offset = 0) = 0;


	public:
		/**
		 *	creates a vertexbuffer without any content, however, the memory will be allocated.
		 *  @param size: The size of the resulting buffer in bytes
		 *  @param usage: The intended usage of the buffer
		 */
		static Ref<VertexBuffer> create(uint32_t size, BufferUsage usage = BufferUsage::StaticDraw);

		/**
		 *	creates a vertexbuffer corresponding to the selected rendering api,
		 *	e.g. opengl, directx, vulcan, etc.
		 *  @param vertices: A pointer to an array of floats that define vertices
		 *  @param size: The size of the vertex-data in bytes
		 *  @param usage: The intended usage of the buffer
		 */
		static Ref<VertexBuffer> create(float *vertices, uint32_t size, BufferUsage usage = BufferUsage::StaticDraw);
		

	protected:
		// only allow creation via 'create'
		VertexBuffer() = default;
	};

	/**
	 *	Wraps a RenderingAPI specific IndexBuffer
	 *	TODO: add support for 16-Bit index-buffers
	 */
	class IndexBuffer {
	public:
		// there is no default constructor, because an indexbuffer should be
		// created via "create()".

		/**
		 *	default destructor
		 */
		virtual ~IndexBuffer() = default;

		/**
		 *	binds this index buffer
		 */
		virtual void bind() const = 0;
		/**
		 *	unbinds this index buffer
		 */
		virtual void unbind() const = 0;

		/**
		 *	returns the amount of indices stored in this buffer.
		 *	the buffer size itself is sizeof(uint32_t) * getCount().
		 */
		virtual uint32_t getCount() const = 0;

		/**
		 *	sets the buffer usage of this buffer
		 *	@param usage: the new usage of this buffer
		 */
		virtual void setBufferUsage(BufferUsage usage) = 0;

		/**
		 *	resizes the buffer on the gpu
		 *	@param count: the new count of indices of the buffer in #indices
		 */
		virtual void resize(uint32_t count) = 0;
		/**
		 *	uploads data to the gpu.
		 *	@param data: a pointer to the data to be uploaded
		 *	@param count: the amount of elements
		 *	@param offset: an offset (in bytes) at which the data should be inserted
		 *	@param usage: the usage of the buffer, e.g. for meshes this might be
		 *		StaticDraw, but particles will be better off with DynamicDraw.
		 */
		virtual void uploadData(uint32_t *data, uint32_t count, uint32_t offset = 0) = 0;

	public:
		/**
		 *	creates an indexbuffer corresponding to the selected rendering api,
		 *	e.g. opengl, directx, vulcan, etc.
		 *  @param vertices: A pointer to an array of uint32_t's that should be uploaded
		 *  @param count: The amount of uint32_t that should be uploaded into the buffer
		 *  @param usage: The intended usage of the buffer
		 */
		static Ref<IndexBuffer> create(uint32_t* indices, uint32_t count, BufferUsage usage = BufferUsage::StaticDraw);
		
	protected:
		// only allow creation via 'create'
		IndexBuffer() = default;
	};

}