# The DMC project with a graphical interface
add_subdirectory(VisualDMC)	# defines target VisualDMC
set_target_properties(VisualDMC PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
set_target_properties(VisualDMC PROPERTIES FOLDER DualMarchingCubes)
set_target_properties(VisualDMC PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# The original DMC project
add_subdirectory(OriginalDMC)	# defines target OriginalDMC
set_target_properties(OriginalDMC PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
set_target_properties(OriginalDMC PROPERTIES FOLDER DualMarchingCubes)
