#pragma once

// ######################################################################### //
// ### gepch.h ############################################################# //
// ### "Game Engine Pre Compiled Headers"                                ### //
// ######################################################################### //

#include <iostream>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <sstream>

#include <string>
#include <sstream>
#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>

#include "LazyEngine/Core/Log.h"

#ifdef LAZYENGINE_PLATFORM_WINDOWS
	//#define NOMINMAX // avoid compilation errors with spdlog
	//#define WIN32_LEAN_AND_MEAN	// exclude APIs like Cryptography, DDE, RPC, Shell and Windows Sockets
	#include <Windows.h>
#endif