#pragma once

// ######################################################################### //
// ### EntryPoint.h ######################################################## //
// ### Defines the Entry Point for the Application that will be created  ### //
// ### via CreateApplication that should be defined in a Client App.     ### //
// ######################################################################### //

#include "LazyEngine/Profiling/Profiler.h"

extern LazyEngine::Application* LazyEngine::createApplication();

#if defined(LAZYENGINE_DEBUG) && defined(LAZYENGINE_PLATFORM_WINDOWS)
// enable memory leak detection on windows in debug builds
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

// the starting point of the program
int main(int argc, char** argv) {

#if defined(LAZYENGINE_DEBUG) && defined(LAZYENGINE_PLATFORM_WINDOWS)
	_CrtMemState s1;
	_CrtMemCheckpoint(&s1);
#endif
	{

		LazyEngine::Application* app;
		LAZYENGINE_PROFILE_BEGIN_SESSION("LazyEngine Startup", "LazyEngine-Startup.json");
		{
			LAZYENGINE_PROFILE_SCOPE("Startup");

			// initialize log system
			LazyEngine::Log::init();
			LAZYENGINE_CORE_WARN("Initialized Log!");

			// create the game application
			LAZYENGINE_INFO("Starting Game Engine!");
			app = LazyEngine::createApplication();
		}
		LAZYENGINE_PROFILE_END_SESSION();

		{
			// start the renderloop
			// This part of the application must be profiled seperately
			// by the push of a button for example.
			app->run();
		}

		LAZYENGINE_PROFILE_BEGIN_SESSION("LazyEngine Shutdown", "LazyEngine-Shutdown.json");
		{
			LAZYENGINE_PROFILE_SCOPE("Shutdown");

			// explicit clean up
			delete app;
		}
		LAZYENGINE_PROFILE_END_SESSION();

	}
#if defined(LAZYENGINE_DEBUG) && defined(LAZYENGINE_PLATFORM_WINDOWS)
	_CrtMemState s2;
	_CrtMemCheckpoint(&s2);

	_CrtMemState s3;
	if (_CrtMemDifference(&s3, &s1, &s2)) {
		_CrtMemDumpStatistics(&s3);
	}
#endif

}