#pragma once

// ######################################################################### //
// ### LayerStack.h ######################################################## //
// ### A wrapper for a std::vector of Layers that distinguishes between  ### //
// ### regular Layers and Overlays that are always on top of them.       ### //
// ######################################################################### //

#include "LazyEngine/Core/Core.h"
#include "Layer.h"

#include <vector>

namespace LazyEngine {

	/**
	 *	a Stack of Layers that can be traversed like an std::vector
	 */
	class LAZYENGINE_API LayerStack {
	public:
		/**
		 *	default constructor that just sets initial values
		 */
		LayerStack();
		/**
		 *	performs an onDetach for every layer inside this stack and deletes it afterwards.
		 */
		~LayerStack();

		/**
		 *	Pushes a layer on top of all existing layers, but below the overlays.
		 *	Then calls layer->onAttach().
		 *	@param layer: the layer to be pushed.
		 *		ownership remains with the creator of the layer who is responsible of its proper deletion
		 */
		void pushLayer(Layer *layer);
		/**
		 *	Pushes an overlay on top of all existing layers and overlays.
		 *	Then calls layer->onAttach().
		 *	@param layer: the layer to be pushed.
		 *		ownership remains with the creator of the layer who is responsible of its proper deletion
		 */
		void pushOverlay(Layer *overlay);
		/**
		 *	Pops a layer from the stack without deleting it.
		 *	Then calls layer->onDetach().
		 *
		 *	Has no effect if the layer is not part of this LayerStack
		 *
		 *	Calling this function with a layer that has previously been registered as an overlay
		 *	will lead to undefined behaviour.
		 *	@param layer: the layer to be removed
		 *		the caller should delete the layer if it will not be used any longer
		 */
		void popLayer(Layer *layer);
		/**
		 *	Pops an overlay from the stack without deleting it.
		 *	Then calls layer->onDetach().
		 *
		 *	Has no effect if the overlay is not part of this LayerStack
		 *
		 *	Calling this function with a layer that has previously been registered as an overlay
		 *	will lead to undefined behaviour.
		 *	@param layer: the layer to be removed
		 *		the caller should delete the layer if it will not be used any longer
		 */
		void popOverlay(Layer *overlay);

		// iterators to access the layers
		std::vector<Layer *>::iterator begin() { return m_layers.begin(); }
		std::vector<Layer *>::iterator end() { return m_layers.end(); }
	private:
		// the vector storing all the layers
		std::vector<Layer *> m_layers;
		// keeps track of where layers should be inserted to not overlay overlays
		unsigned int m_layerInsertIndex;
	};

}