#include "LazyEngine/gepch.h"

#include "Profiler.h"

#include "LazyEngine/Core/Application.h" // used to profile the next n frames

namespace LazyEngine {

	Profiler::Profiler()
		: m_outputStream()
		, m_profileCount(0)
		, m_currentSession(nullptr)
		, m_endFrame(-1)
	{
		// empty
	}

	void Profiler::beginSession(const std::string& sessionName, const std::string& filePath) {
		m_outputStream.open(filePath);
		writeHeader();
		LAZYENGINE_CORE_ASSERT(m_currentSession == nullptr, "Current Session is not nullptr! Did you forget to call endSession()?");
		m_currentSession = std::make_unique<ProfilerSession>();
		m_currentSession->name = sessionName;
	}

	void Profiler::beginNSession() {
		beginSession(m_profileSessionName, m_profileSessionFilePath);
	}

	void Profiler::endSession() {
		writeFooter();
		m_outputStream.close();
		m_currentSession = nullptr;
		m_profileCount = 0;
		m_endFrame = -1;
		m_profileSessionName = "";
		m_profileSessionFilePath = "";
	}

	void Profiler::profileNextNFrames(int n, const std::string& sessionName, const std::string& filePath) {
		Application& app = Application::getInstance();
		// set the end frame to be n frames in the future
		m_endFrame = app.getFrameCount() + n;
		m_profileSessionName = sessionName;
		m_profileSessionFilePath = filePath;
	}


	void Profiler::writeProfile(const ProfileResult& result) {
		// add a comma before every profile that is not the first one.
		if (m_profileCount > 0) {
			m_outputStream << ",";
		}
		++m_profileCount;

		// escape quotes in the name of the result
		std::string name = result.name;
		std::replace(name.begin(), name.end(), '"', '\'');

		// output the data in a format that chrome understands
		m_outputStream << "{";												// begin of a new element
		m_outputStream << "\"cat\":\"function\",";							// ???
		TimeStep duration = TimeStep(result.start, result.end);
		m_outputStream << "\"dur\":" << duration.getMicroseconds() << ",";	// duration of the profiled section
		m_outputStream << "\"name\":\"" << name << "\",";					// name of the segment
		m_outputStream << "\"ph\":\"X\",";									// ???
		m_outputStream << "\"pid\":0,";										// ???
		m_outputStream << "\"tid\":" << result.threadID << ",";				// threadID of the thread that was profiled
		m_outputStream << "\"ts\":" << result.start.toMicroseconds();						// start of the section
		m_outputStream << "}";												// end of element

		// force the outputstream to write to file so it won't be lost
		// in the case of something crashing in this program.
		m_outputStream.flush();
	}

	void Profiler::writeHeader() {
		m_outputStream << "{\"otherData\":{},\"traceEvents\":[";
		m_outputStream.flush();
	}

	void Profiler::writeFooter() {
		m_outputStream << "]}";
		m_outputStream.flush();
	}

}