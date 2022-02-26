#pragma once

#include "Utilities.h"
#include <iostream>
#include <iomanip>

namespace DMC {



	int Performance::s_id = 0;
	std::unique_ptr<MemoryMonitor> MemoryMonitor::s_instance = nullptr;


	// ######################################################################## //
	// ### ImGuiStyling ####################################################### //
	// ######################################################################## //

	void ImGuiStyling::pushColors(int id, int currentlySelectedId, int maxId) {
		float maxColorNum = static_cast<float>(maxId);

		float defaultSaturation = 0.6f;
		float defaultHoverSaturation = 0.7f;
		float defaultActiveSaturation = 0.8f;
		if (id == currentlySelectedId) {
			defaultSaturation = 0.9f;
			defaultHoverSaturation = 0.95f;
			defaultActiveSaturation = 1.f;
		}
		ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(id / maxColorNum, defaultSaturation, defaultSaturation));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(id / maxColorNum, defaultHoverSaturation, defaultHoverSaturation));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(id / maxColorNum, defaultActiveSaturation, defaultActiveSaturation));
	}

	void ImGuiStyling::popColors() {
		ImGui::PopStyleColor(3);
	}

	void ImGuiStyling::pushTextColors(int id, int maxId) {
		float maxColorNum = static_cast<float>(maxId);
		ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)ImColor::HSV(id / maxColorNum, 0.9f, 0.9f));
		ImGui::PushStyleColor(ImGuiCol_PlotHistogram, (ImVec4)ImColor::HSV(id / maxColorNum, 0.9f, 0.9f));
	}

	void ImGuiStyling::popTextColors() {
		ImGui::PopStyleColor(2);

	}

	// ######################################################################## //
	// ### Performance ######################################################## //
	// ######################################################################## //

	void Performance::resetStaticId() {
		s_id = 0;
	}

	Performance::Performance(const std::string& name, const LazyEngine::TimeStep& time)
		: m_name(name)
		, m_time(time)
		, m_cumulativeTime(time)
		, m_children()
		, m_id(s_id++)
	{
		m_memorySnapshot = MemoryMonitor::getInstance().getMemorySnapshot();
	}

	void Performance::addChild(const Performance& performance) {
		m_children.push_back(performance);
		updateTime();
	}

	void Performance::addChildren(const std::vector<Performance>& performances) {
		m_children.insert(m_children.end(), performances.begin(), performances.end());
		updateTime();
	}

	const std::string& Performance::getName() const {
		return m_name;
	}

	const LazyEngine::TimeStep& Performance::getTime() const {
		return m_cumulativeTime;
	}

	const std::vector<Performance>& Performance::getChildren() const {
		return m_children;
	}

	void Performance::updateTime() {
		m_cumulativeTime = m_time;
		for (auto& child : m_children) {
			// update children recursively
			child.updateTime();
			// add the children's times to the cumulative time
			m_cumulativeTime += child.getTime();
		}
	}

	void Performance::renderChildrenUntilMatch(int* currentlySelectedId, const LazyEngine::TimeStep& totalTime) const {
		if (m_children.size() > 0) {
			float horizontalSpace = ImGui::GetContentRegionAvail().x;
			ImGui::PushID(m_id);
			ImGuiStyling::pushColors(m_id, *currentlySelectedId, s_id);
			if (ImGui::Button(m_name.c_str(), { horizontalSpace, 25.f })) {
				const int before = *currentlySelectedId;
				*currentlySelectedId = m_id;
			}
			ImGuiStyling::popColors();
			ImGui::PopID();

			renderImguiChildrenButtons(currentlySelectedId);
		}

		if (containsID(*currentlySelectedId)) {
			// find the child that matches the id
			for (auto& child : m_children) {
				if (child.containsID(*currentlySelectedId) || child.matchesID(*currentlySelectedId)) {
					// Let the child render its children
					child.renderChildrenUntilMatch(currentlySelectedId, totalTime);
				}
			}
		}
		else {
			renderInfo(totalTime);
		}
	}

	void Performance::renderImguiChildrenButtons(int* currentlySelectedId) const {
		assert(currentlySelectedId != nullptr);
		float horizontalSpace = ImGui::GetContentRegionAvail().x;

		for (int i = 0; i < m_children.size(); ++i) {
			if (i > 0) ImGui::SameLine(0.f, 0.f);
			const auto& child = m_children[i];
			const int id = child.m_id;
			ImGui::PushID(id);
			ImGuiStyling::pushColors(id, *currentlySelectedId, s_id);
			float percentage = child.getTime().getMilliseconds() / getTime().getMilliseconds();
			std::stringstream ss;
			ss.precision(3);
			ss << percentage * 100.f;
			ss << "%";
			if (ImGui::Button(ss.str().c_str(), { horizontalSpace * percentage, 25.f })) {
				*currentlySelectedId = id;
			}
			ImGuiStyling::popColors();

			ImGui::PopID();
		}

		std::stringstream ss;
		ss << "Detals for " << m_name;
		if (ImGui::CollapsingHeader(ss.str().c_str())) {
			ImGui::Indent();
			renderImGuiChildrenButtonsHuge(currentlySelectedId);
			ImGui::Unindent();
		}
	}

	void Performance::renderImGuiChildrenButtonsHuge(int* currentlySelectedId) const {
		float horizontalSpace = ImGui::GetContentRegionAvail().x;
		float maxColorNum = static_cast<float>(s_id);

		for (int i = 0; i < m_children.size(); ++i) {
			auto& child = m_children[i];
			const int id = child.m_id;
			ImGui::PushID(id);
			ImGuiStyling::pushColors(id, *currentlySelectedId, s_id);
			float percentage = child.getTime().getMilliseconds() / m_cumulativeTime.getMilliseconds();
			std::stringstream ss;
			ss << child.m_name;
			if (ImGui::Button(ss.str().c_str(), { horizontalSpace, 25.f })) {
				*currentlySelectedId = id;
			}
			ImGuiStyling::popColors();
			ImGui::PopID();

		}
	}

	void Performance::renderInfo(const LazyEngine::TimeStep& totalTime) const {
		float milliseconds = m_cumulativeTime.getMilliseconds();
		float percentage = milliseconds / totalTime.getMilliseconds();
		float maxColorNum = static_cast<float>(s_id);
		ImGuiStyling::pushTextColors(m_id, maxColorNum);
		ImGui::Text("%s: %.3fms", m_name.c_str(), milliseconds);
		ImGui::ProgressBar(percentage);
		ImGuiStyling::popTextColors();
	}


	bool Performance::containsID(int id) const {
		for (const auto& child : m_children) {
			// first, check if the child matches the id
			bool childMatches = child.matchesID(id);
			if (childMatches) {
				return true;
			}
			// second, check the child's children
			bool childContains = child.containsID(id);
			if (childContains) {
				return true;
			}
		}
		return false;
	}

	bool Performance::matchesID(int id) const {
		return id == m_id;
	}

	std::string Performance::toString(int depth) const {
		std::string padding = "";
		{
			std::stringstream ss;
			for (int i = 0; i < depth; ++i) {
				ss << "\t";
			}
			padding = ss.str();
		}
		std::stringstream ss;
		ss << padding << "{\n";
		ss << padding << "\t" << "\"name\": " << "\"" << getName() << "\",\n";
		ss << padding << "\t" << "\"duration\": " << getTime().getMilliseconds() << ",\n";
		ss << padding << "\t" << "\"children\": [\n";
		int childIndex = 0;
		for (const auto& child : m_children) {
			ss << child.toString(depth + 1);
			++childIndex;
			if (childIndex == m_children.size()) {
				ss << "\n";
			}
			else {
				ss << ",\n";
			}
		}
		ss << padding << "\t" << "],\n";
		ss << padding << "\t" << "\"memoryUsage\": [\n";
		int bufferIndex = 0;
		for (const auto& buffer : m_memorySnapshot) {
			ss << padding << "\t\t" << "{\n";
			ss << padding << "\t\t\t" << "\"bufferName\": " << "\"" << buffer.name << "\"" << ",\n";
			ss << padding << "\t\t\t" << "\"elementSize\": " << buffer.elementSize << ",\n";
			ss << padding << "\t\t\t" << "\"elementCount\": " << buffer.elementCount << "\n";
			ss << padding << "\t\t";
			++bufferIndex;
			if (bufferIndex == m_memorySnapshot.size()) {
				ss << "}\n";
			}
			else {
				ss << "},\n";
			}
		}
		ss << padding << "\t]\n";
		ss << padding << "}";

		return ss.str();
	}


	// ##################################################################### //
	// ### MemoryMonitor ############################################### //

	MemoryMonitor& MemoryMonitor::getInstance() {
		if (s_instance == nullptr) {
			s_instance = std::unique_ptr<MemoryMonitor>(new MemoryMonitor());
		}
		return *s_instance;
	}

	static inline std::string createStringMessage(size_t totalBytes, size_t currentBytes, const std::string& message) {
		// convert total byte count to kilo-/mega-/giga-bytes
		float kilo = static_cast<float>(totalBytes) / 1000.f;
		float mega = kilo / 1000.f;
		float giga = mega / 1000.f;

		// convert bufferInfo's byte count to kilo-/mega-/giga-bytes
		float bufferBytes = static_cast<float>(currentBytes);
		float bufferKilo = bufferBytes / 1000.f;
		float bufferMega = bufferKilo / 1000.f;
		float bufferGiga = bufferMega / 1000.f;

		std::stringstream ss;
		ss << "Reserved: ";
		if (giga > 0.5f) {
			ss << std::setfill(' ') << std::setw(7) << std::fixed << std::setprecision(2) << giga << " GB";
		}
		else if (mega > 0.5f) {
			ss << std::setfill(' ') << std::setw(7) << std::fixed << std::setprecision(2) << mega << " MB";
		}
		else if (kilo > 0.5f) {
			ss << std::setfill(' ') << std::setw(7) << std::fixed << std::setprecision(2) << kilo << " KB";
		}
		else {
			ss << std::setfill(' ') << std::setw(7) << totalBytes << " B ";
		}
		ss << ". " << message << " ";
		if (bufferGiga > 0.5f) {
			ss << std::setfill(' ') << std::setw(7) << std::fixed << std::setprecision(2) << bufferGiga << " GB";
		}
		else if (bufferMega > 0.5f) {
			ss << std::setfill(' ') << std::setw(7) << std::fixed << std::setprecision(2) << bufferMega << " MB";
		}
		else if (bufferKilo > 0.5f) {
			ss << std::setfill(' ') << std::setw(7) << std::fixed << std::setprecision(2) << bufferKilo << " KB";
		}
		else {
			ss << std::setfill(' ') << std::setw(7) << bufferBytes << " B ";
		}
		ss << " of {0}";

		return ss.str();
	}

	std::shared_ptr<BufferInfo> MemoryMonitor::registerBuffer(const BufferInfo& bufferInfo) {
		std::shared_ptr<BufferInfo> entry = std::make_shared<BufferInfo>(bufferInfo);
		m_activeBuffers.push_back(entry);

		std::sort(m_activeBuffers.begin(), m_activeBuffers.end(), [](const std::shared_ptr<BufferInfo>& a, const std::shared_ptr<BufferInfo>& b) {
			return (a->elementCount * a->elementSize) < (b->elementCount * b->elementSize);
		});

		updateTotalByteCount();

#ifdef LAZYENGINE_DEBUG
		LAZYENGINE_WARN(createStringMessage(m_totalByteCount, bufferInfo.elementCount * bufferInfo.elementSize, "Registered  ").c_str(), bufferInfo.name);
#endif

		return entry;
	}

	void MemoryMonitor::unregisterBuffer(const std::shared_ptr<BufferInfo>& handle) {
		for (int i = 0; i < m_activeBuffers.size(); ++i) {
			if (m_activeBuffers[i] == handle) {
				m_activeBuffers.erase(m_activeBuffers.begin() + i);
				break;
			}
		}

		updateTotalByteCount();

#ifdef LAZYENGINE_DEBUG
		LAZYENGINE_WARN(createStringMessage(m_totalByteCount, handle->elementCount * handle->elementSize, "Unregistered").c_str(), handle->name);
#endif
	}

	void MemoryMonitor::displayPercentagesImGui() {
		float horizontalSpace = ImGui::GetContentRegionAvail().x;
		int index = 0.f;
		int maxIndex = m_activeBuffers.size() - 1;

		float defaultSaturation = 0.6f;
		float defaultHoverSaturation = 0.7f;
		float defaultActiveSaturation = 0.8f;
		for (auto& entry : m_activeBuffers) {
			if (index > 0) ImGui::SameLine(0.f, 0.f);


			ImGuiStyling::pushColors(index, m_selectedBufferIndex, maxIndex);

			size_t entryBytes = entry->elementCount * entry->elementSize;
			float percentage = static_cast<float>(entryBytes) / static_cast<float>(m_totalByteCount);
			percentage = std::max(percentage, 0.001f);

			ImGui::PushID(index);
			if (ImGui::Button(entry->name.c_str(), { horizontalSpace * percentage, 25.f })) {
				m_selectedBufferIndex = index;
			}
			ImGui::PopID();

			ImGuiStyling::popColors();


			++index;
		}
	}

	void MemoryMonitor::displayLargeButtonsImGui() {
		float horizontalSpace = ImGui::GetContentRegionAvail().x;
		int index = 0;
		int maxIndex = m_activeBuffers.size() - 1;

		float defaultSaturation = 0.6f;
		float defaultHoverSaturation = 0.7f;
		float defaultActiveSaturation = 0.8f;
		for (auto& entry : m_activeBuffers) {

			ImGuiStyling::pushColors(index, m_selectedBufferIndex, maxIndex);

			ImGui::PushID(index + 50000);
			if (ImGui::Button(entry->name.c_str(), { horizontalSpace, 20.f })) {
				m_selectedBufferIndex = index;
			}
			ImGui::PopID();

			ImGuiStyling::popColors();

			++index;
		}
	}

	void MemoryMonitor::displayDetailsImGui() {
		if (m_selectedBufferIndex == -1) return;
		float maxIndex = m_activeBuffers.size() - 1;

		size_t freeMemory;
		size_t totalMemory;
		checkCudaErrors(cudaMemGetInfo(&freeMemory, &totalMemory));

		auto iter = m_activeBuffers.begin();
		std::advance(iter, m_selectedBufferIndex);
		BufferInfo info = **iter;
		size_t totalBytes = info.elementCount * info.elementSize;
		float totalPercentage = static_cast<float>(totalBytes * 100) / static_cast<float>(totalMemory);
		float monitoredPercentage = static_cast<float>(totalBytes * 100) / static_cast<float>(m_totalByteCount);

		ImGuiStyling::pushTextColors(m_selectedBufferIndex, maxIndex);
		ImGui::Text("%s", info.name.c_str());
		ImGui::Text("ElementCount: %d", info.elementCount);
		ImGui::Text("ElementSize: %d", info.elementSize);
		ImGui::Text("Byte Count: %d", totalBytes);
		ImGui::Text("KB: %.1f", static_cast<float>(totalBytes) * 0.001f);
		ImGui::Text("MB: %.1f", static_cast<float>(totalBytes) * 0.000001f);
		ImGui::Text("GB: %.1f", static_cast<float>(totalBytes)*0.000000001f);
		ImGui::Text("Percentage (Of GPU-Memory): %.2f%%", totalPercentage);
		ImGui::Text("Percentage (Of Monitored Buffers): %.2f%%", monitoredPercentage);
		ImGuiStyling::popTextColors();
		ImGui::Separator();
	}

	void MemoryMonitor::displayGPUInfoImGui() {
		// Get the Memory consumption on GPU
		size_t freeMemory;
		size_t totalMemory;
		checkCudaErrors(cudaMemGetInfo(&freeMemory, &totalMemory));
		float usedPercentage = static_cast<float>(totalMemory - freeMemory) / static_cast<float>(totalMemory);
		
		ImGui::Text("Total GPU-Memory usage");
		ImGui::ProgressBar(usedPercentage);
		float freePercentage = static_cast<float>(freeMemory) / static_cast<float>(totalMemory);
		freePercentage *= 100.f;
		ImGui::Text("%llu/%llu (%.3f%%) Bytes available on GPU", freeMemory, totalMemory, freePercentage);

		float registeredPercentage = static_cast<float>(m_totalByteCount) / static_cast<float>(totalMemory);

		ImGui::Text("Registered Buffer GPU-Memory usage");
		ImGui::ProgressBar(registeredPercentage);

		ImGui::Text("%.1f MB registered", static_cast<float>(m_totalByteCount) * 0.000001f);
	}


	MemoryMonitor::MemoryMonitor()
		: m_activeBuffers()
		, m_totalByteCount(0)
		, m_selectedBufferIndex(-1)
		, m_maxTotalByteCount(0)
	{
		// empty
	}

	void MemoryMonitor::updateTotalByteCount() {
		m_totalByteCount = 0;
		for (auto& entry : m_activeBuffers) {
			m_totalByteCount += entry->elementCount * entry->elementSize;
		}
		if (m_totalByteCount > m_maxTotalByteCount) {
			m_maxTotalByteCount = m_totalByteCount;
		}
	}

	std::vector<BufferInfo> MemoryMonitor::getMemorySnapshot() const {
		std::vector<BufferInfo> result;
		for (auto& entry : m_activeBuffers) {
			result.push_back(*entry);
		}
		return result;
	}


}