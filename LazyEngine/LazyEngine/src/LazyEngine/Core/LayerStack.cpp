// ######################################################################### //
// ### LayerStack.cpp ###################################################### //
// ### Implements LayerStack.h                                           ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"

#include "LayerStack.h"

#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	LayerStack::LayerStack() 
		: m_layerInsertIndex(0)
	{
		LAZYENGINE_PROFILE_FUNCTION();

		// empty
	}

	LayerStack::~LayerStack() {
		LAZYENGINE_PROFILE_FUNCTION();

		for (Layer *layer : m_layers) {
			layer->onDetach();
			delete layer;
		}
	}

	void LayerStack::pushLayer(Layer *layer) {
		LAZYENGINE_PROFILE_FUNCTION();

		// insert the layer before the first overlay which will be the top if there is none.
		m_layers.emplace(m_layers.begin() + m_layerInsertIndex, layer);
		m_layerInsertIndex++;
		layer->onAttach();
	}

	void LayerStack::pushOverlay(Layer *overlay) {
		LAZYENGINE_PROFILE_FUNCTION();

		// insert the overlay at the top of the stack
		m_layers.emplace_back(overlay);
		overlay->onAttach();
	}

	void LayerStack::popLayer(Layer *layer) {
		LAZYENGINE_PROFILE_FUNCTION();

		// find layer in the stack
		auto iter = std::find(m_layers.begin(), m_layers.end(), layer);
		if (iter != m_layers.end()) {
			// pop the found layer 
			layer->onDetach();
			m_layers.erase(iter);
			m_layerInsertIndex--;
		}
	}

	void LayerStack::popOverlay(Layer *overlay) {
		LAZYENGINE_PROFILE_FUNCTION();

		// find the overlay in the stack
		auto iter = std::find(m_layers.begin(), m_layers.end(), overlay);
		if (iter != m_layers.end()) {
			// pop the found overlay
			overlay->onDetach();
			m_layers.erase(iter);
		}
	}
}