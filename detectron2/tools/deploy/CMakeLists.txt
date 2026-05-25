# Copyright (c) Facebook, Inc. and its affiliates.
# See https://pytorch.org/tutorials/advanced/cpp_frontend.html
#
# Notes for pet-vton:
#   This C++ deploy example is OPTIONAL. The pet-vton pipeline is implemented
#   in Python (see backend_worker/run_pet_tryon.py). If Torch / TorchVision /
#   OpenCV C++ libraries are not available, this CMake file will skip building
#   the deploy binary instead of failing the entire configure step.
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(torchscript_mask_rcnn)

# -----------------------------------------------------------------------------
# Local dependency prefix: <repo_root>/runtime/cuda-toolkit
# -----------------------------------------------------------------------------
get_filename_component(_repo_root "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(_local_dep_prefix "${_repo_root}/runtime/cuda-toolkit")
if(EXISTS "${_local_dep_prefix}")
  list(APPEND CMAKE_PREFIX_PATH "${_local_dep_prefix}")
  if(EXISTS "${_local_dep_prefix}/bin/nvcc")
    set(CUDA_TOOLKIT_ROOT_DIR "${_local_dep_prefix}" CACHE PATH "Local CUDA toolkit root")
    set(CUDAToolkit_ROOT "${_local_dep_prefix}" CACHE PATH "Local CUDA toolkit root")
    set(CUDA_NVCC_EXECUTABLE "${_local_dep_prefix}/bin/nvcc" CACHE FILEPATH "Local nvcc")
  endif()
endif()

# -----------------------------------------------------------------------------
# Auto-detect Torch_DIR from environment / Python installs.
# Handles: undefined, NOTFOUND-cached, stale paths.
# -----------------------------------------------------------------------------
if(NOT DEFINED Torch_DIR OR Torch_DIR MATCHES "NOTFOUND$" OR NOT EXISTS "${Torch_DIR}/TorchConfig.cmake")
  unset(Torch_DIR CACHE)

  if(DEFINED ENV{Torch_DIR} AND EXISTS "$ENV{Torch_DIR}/TorchConfig.cmake")
    set(Torch_DIR "$ENV{Torch_DIR}" CACHE PATH "Path to TorchConfig.cmake" FORCE)
  else()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    set(_torch_python_candidates "")
    if(Python3_Interpreter_FOUND)
      list(APPEND _torch_python_candidates "${Python3_EXECUTABLE}")
    endif()
    if(DEFINED ENV{PYTHON_BIN})
      list(APPEND _torch_python_candidates "$ENV{PYTHON_BIN}")
    endif()
    if(DEFINED ENV{CONDA_PREFIX})
      list(APPEND _torch_python_candidates "$ENV{CONDA_PREFIX}/bin/python")
    endif()
    if(DEFINED ENV{HOME})
      list(APPEND _torch_python_candidates "$ENV{HOME}/miniconda3/bin/python")
      file(GLOB _conda_env_pythons "$ENV{HOME}/miniconda3/envs/*/bin/python")
      list(APPEND _torch_python_candidates ${_conda_env_pythons})
    endif()
    list(APPEND _torch_python_candidates python3 python)

    foreach(_py IN LISTS _torch_python_candidates)
      if(Torch_DIR AND EXISTS "${Torch_DIR}/TorchConfig.cmake")
        break()
      endif()
      execute_process(
        COMMAND "${_py}" -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake', 'Torch'))"
        RESULT_VARIABLE _torch_detect_rc
        OUTPUT_VARIABLE _torch_dir_candidate
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(_torch_detect_rc EQUAL 0 AND _torch_dir_candidate AND EXISTS "${_torch_dir_candidate}/TorchConfig.cmake")
        set(Torch_DIR "${_torch_dir_candidate}" CACHE PATH "Path to TorchConfig.cmake" FORCE)
        get_filename_component(_torch_cmake_prefix "${_torch_dir_candidate}/.." ABSOLUTE)
        list(APPEND CMAKE_PREFIX_PATH "${_torch_cmake_prefix}")
        message(STATUS "Detected Torch_DIR: ${Torch_DIR} (via ${_py})")
      endif()
    endforeach()
  endif()
endif()

# -----------------------------------------------------------------------------
# Suppress benign 'library kineto not found' warning from TorchConfig.cmake.
# Modern PyTorch wheels statically link libkineto into libtorch_cpu.so, so the
# separate libkineto.a/.so is missing. We pre-populate the cache variable that
# TorchConfig's find_library uses, pointing it at libtorch_cpu.so (already
# linked anyway) so the find succeeds silently.
# -----------------------------------------------------------------------------
if(Torch_DIR AND EXISTS "${Torch_DIR}/TorchConfig.cmake" AND NOT kineto_LIBRARY)
  get_filename_component(_torch_install_prefix "${Torch_DIR}/../../../" ABSOLUTE)
  set(_torch_lib_dir "${_torch_install_prefix}/lib")
  if(EXISTS "${_torch_lib_dir}/libtorch_cpu.so")
    set(kineto_LIBRARY "${_torch_lib_dir}/libtorch_cpu.so" CACHE FILEPATH "kineto bundled into libtorch_cpu" FORCE)
  endif()
endif()

# -----------------------------------------------------------------------------
# Soft package discovery: never FATAL_ERROR if a dep is missing.
# Build the deploy binary only if everything is available.
# -----------------------------------------------------------------------------
set(_deploy_can_build TRUE)

find_package(Torch CONFIG QUIET)
if(NOT Torch_FOUND)
  message(WARNING
    "[deploy] Torch (libtorch C++) not found. "
    "Skipping torchscript_mask_rcnn target. "
    "This does NOT affect the Python pet-vton pipeline.")
  set(_deploy_can_build FALSE)
endif()

if(_deploy_can_build)
  find_package(OpenCV QUIET)
  if(NOT OpenCV_FOUND)
    message(WARNING "[deploy] OpenCV not found. Skipping torchscript_mask_rcnn target.")
    set(_deploy_can_build FALSE)
  endif()
endif()

# TorchVisionConfig internally calls find_package(Torch REQUIRED), which
# would FATAL_ERROR if Torch is missing. Only attempt it when Torch is found.
if(_deploy_can_build)
  find_package(TorchVision QUIET)
  if(NOT TorchVision_FOUND)
    message(WARNING "[deploy] TorchVision (C++) not found. Skipping torchscript_mask_rcnn target.")
    set(_deploy_can_build FALSE)
  endif()
endif()

if(_deploy_can_build)
  add_executable(torchscript_mask_rcnn torchscript_mask_rcnn.cpp)
  target_link_libraries(
    torchscript_mask_rcnn
    -Wl,--no-as-needed TorchVision::TorchVision -Wl,--as-needed
    "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
  set_property(TARGET torchscript_mask_rcnn PROPERTY CXX_STANDARD 17)
  message(STATUS "[deploy] torchscript_mask_rcnn target enabled.")
else()
  message(STATUS "[deploy] torchscript_mask_rcnn target skipped (missing C++ deps).")
endif()