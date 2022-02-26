#pragma once

#include <LazyEngine/LazyEngine.h>

namespace DMC {

	struct BufferInfo {
		size_t elementCount;
		size_t elementSize;
		std::string name;
	};

	constexpr int INVALID_INDEX = -1;

	__host__ __device__ inline size_t concatenateKeys(const int value1, const int value2) {
		if (value1 < value2) {
			return (static_cast<size_t>(value1) << 32) | (value2 & 0xFF'FF'FF'FF'FF'FFULL);
		}
		else {
			return (static_cast<size_t>(value2) << 32) | (value1 & 0xFF'FF'FF'FF'FF'FFULL);
		}
	}

	class ImGuiStyling {
	public:
		static void pushColors(int id, int currentlySelectedId, int maxId);
		static void popColors();

		static void pushTextColors(int id, int maxId);
		static void popTextColors();
	};

	class Performance {
	public:
		static void resetStaticId();

	public:
		/**
		 *	constructor
		 */
		Performance(const std::string& name = "Performance", const LazyEngine::TimeStep& time = 0.f);

		/**
		 *	Adds a single child performance
		 */
		void addChild(const Performance& performance);

		/**
		 *	adds a whole bunch of child performances
		 */
		void addChildren(const std::vector<Performance>& performances);

		/** 
		 *	returns the name of this performance
		 */
		const std::string& getName() const;

		/**
		 *	returns the cumulative time of this performance and its children
		 */
		const LazyEngine::TimeStep& getTime() const;

		/**
		 *	returns the child performances
		 */
		const std::vector<Performance>& getChildren() const;

		/**
		 *	Returns the Memory Info Snapshot
		 */
		const std::vector<BufferInfo>& getMemorySnapshot() const;

		// ### ImGui Things ################################################ //

		void renderChildrenUntilMatch(int* currentlySelectedId, const LazyEngine::TimeStep& totalTime) const;

		/**
		 *	Renders the children as buttons in a single line, with their sizes relative
		 *	to their time amount.
		 */
		void renderImguiChildrenButtons(int *currentlySelectedId) const;

		/**
		 *	Renders the children as buttons in a clickable way
		 */
		void renderImGuiChildrenButtonsHuge(int* currentlySelectedId) const;

		/**
		 *	renders the name, time and percentage of the total time
		 */
		void renderInfo(const LazyEngine::TimeStep& totalTime) const;

		/**
		 *	Returns true if any of the children (recursively) match the given id
		 */
		bool containsID(int id) const;

		/**
		 *	Returns true if this Performance matches the given id
		 */
		bool matchesID(int id) const;

		std::string toString(int depth = 0) const;


	protected:
		void updateTime();

	protected:
		std::string m_name;
		LazyEngine::TimeStep m_time;
		LazyEngine::TimeStep m_cumulativeTime;
		std::vector<Performance> m_children;

		std::vector<BufferInfo> m_memorySnapshot;

		int m_id;

	private:
		// makes sure that every new Performance has its own id
		static int s_id;
	};
	
	/**
	 *	starts timing upon construction and stops when destructed
	 */
	class PerformanceTimer {
	public:
		PerformanceTimer(const std::string& name, Performance& performances)
			: m_name(name)
			, m_timer()
			, m_performances(performances)
		{
#ifdef LAZYENGINE_DEBUG
			LAZYENGINE_INFO("Start {0}", m_name);
#endif
			m_timer.start();
		}

		~PerformanceTimer() {
			cudaCheckError();
#ifdef LAZYENGINE_DEBUG
			LAZYENGINE_INFO("End {0}", m_name);
#endif
			m_performances.addChild({ m_name, m_timer.stop() });
		}

	protected:
		std::string m_name;
		LazyEngine::Timer m_timer;
		Performance& m_performances;
	};

	/**
	 *	Turns a LazyEngine::DataView into cuda kernel-start arguments
	 */
#define ToCudaArgs(x) CUDA_KERNEL(x.getNumBlocks(), x.getNumThreadsPerBlock())

	// ##################################################################### //
	// ### Memory Profiling ################################################ //
	// ##################################################################### //


	 /**
	  *	A singleton class monitoring cuda-allocations (mostly of MonitoredDeviceVectors, though)
	  */
	class MemoryMonitor {
	public:
		static MemoryMonitor& getInstance();
		~MemoryMonitor() = default;

		/**
		 *	registers a single buffer
		 *	returns a handle that can be used to unregister the buffer
		 */
		std::shared_ptr<BufferInfo> registerBuffer(const BufferInfo& bufferInfo);
		/**
		 *	unregisters a single buffer with its handle (returned by registerBuffer.
		 *	If no buffer with that name was registered in the first place, nothing happens
		 */
		void unregisterBuffer(const std::shared_ptr<BufferInfo>& handle);

		/**
		 *	renders the buffers in a percentage-like
		 */
		void displayPercentagesImGui();
		void displayLargeButtonsImGui();
		void displayDetailsImGui();

		/**
		 *	renders the memory usage of the GPU, as well as the GPU usage of registered buffers
		 */
		void displayGPUInfoImGui();

		/**
		 *	returns the total amount of bytes, that is used by the registered buffers
		 */
		inline size_t getTotalByteCount() const {
			return m_totalByteCount;
		}

		std::vector<BufferInfo> getMemorySnapshot() const;

		inline void resetMaxTotalByteCount() {
			m_maxTotalByteCount = 0;
		}

		inline size_t getMaxTotalByteCount() {
			return m_maxTotalByteCount;
		}

	protected:
		/**
		 *	protected constructor to ensure singleton nature
		 */
		MemoryMonitor();

		void updateTotalByteCount();

	protected:
		std::vector<std::shared_ptr<BufferInfo>> m_activeBuffers;
		// stores {name, byte-count} pairs of MonitoredDeviceVectors
		size_t m_totalByteCount;

		int m_selectedBufferIndex;

		size_t m_maxTotalByteCount;

	protected:
		// The singleton instance
		static std::unique_ptr<MemoryMonitor> s_instance;
	};

	/**
	 *	A wrapper for buffers that register and unregister themselves automatically upon construction/destruction
	 */
	template <typename T>
	class MonitoredThrustBuffer {
	public:
		MonitoredThrustBuffer(size_t elementCount, const std::string& name) 
			: m_buffer()
			, m_bufferInfo(nullptr)
		{
			init(elementCount, name);
		}

		MonitoredThrustBuffer(size_t elementCount, T initialValue, const std::string& name)
			: m_buffer()
			, m_bufferInfo(nullptr)
		{
			init(elementCount, initialValue, name);
		}

		~MonitoredThrustBuffer() {
			MemoryMonitor::getInstance().unregisterBuffer(m_bufferInfo);
		}

		inline void checkMemory(size_t elementCount, const std::string& name) {
			size_t numBytes = elementCount * sizeof(T);
			auto info = LazyEngine::cudaMemoryInfo();
			if (info.free < numBytes) {
				LAZYENGINE_ERROR("Attempting to reserve {0} Bytes, while only {1} Bytes are available: {2}", numBytes, info.free, name);
				throw std::exception("Out of Memory!");
			}
		}

		inline void init(size_t elementCount, const std::string& name) {
			checkMemory(elementCount, name);

			m_buffer = thrust::device_vector<T>(elementCount);
			m_bufferInfo = MemoryMonitor::getInstance().registerBuffer({ elementCount, sizeof(T), name });
		}

		inline void init(size_t elementCount, T initialValue, const std::string& name) {
			checkMemory(elementCount, name);

			m_buffer = thrust::device_vector<T>(elementCount, initialValue);
			m_bufferInfo = MemoryMonitor::getInstance().registerBuffer({ elementCount, sizeof(T), name });
		}

		inline operator thrust::device_vector<T>&() {
			return m_buffer;
		}

		inline operator const thrust::device_vector<T>&() const {
			return m_buffer;
		}

		inline operator LazyEngine::DataView<T>() {
			return m_buffer;
		}

		inline operator const LazyEngine::DataView<T>() const {
			return m_buffer;
		}

		inline thrust::device_vector<T>& getBuffer() {
			return m_buffer;
		}

		inline const thrust::device_vector<T>& getBuffer() const {
			return m_buffer;
		}

		inline decltype(std::declval<thrust::device_vector<T>>().begin()) begin() {
			return m_buffer.begin();
		}

		inline decltype(std::declval<thrust::device_vector<T>>().end()) end() {
			return m_buffer.end();
		}

		inline decltype(std::declval<thrust::device_vector<T> const>().begin()) begin() const {
			return m_buffer.begin();
		}

		inline decltype(std::declval<thrust::device_vector<T> const>().end()) end() const {
			return m_buffer.end();
		}

		inline size_t size() const {
			return m_buffer.size();
		}

		inline void resize(size_t numElements) {
			checkMemory(numElements, m_bufferInfo->name);

			m_buffer.resize(numElements);
			m_bufferInfo->elementCount = numElements;
		}

	protected:
		thrust::device_vector<T> m_buffer;
		//std::string m_name;
		std::shared_ptr<BufferInfo> m_bufferInfo;
	};

#define VECC(a) a.x, a.y
#define VECCC(a) a.x, a.y, a.z
#define VECCCC(a) a.x, a.y, a.z, a.w

	// stringifys the name of a variable
#define NameOf(variable) ((void)variable, #variable)

#ifdef LAZYENGINE_DEBUG
#define DEVICE_ASSERT(x, message, ...) if (!(x)) printf("%s (%d): " message "\n", __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define DEVICE_ASSERT(x, message, ...)
#endif
}
