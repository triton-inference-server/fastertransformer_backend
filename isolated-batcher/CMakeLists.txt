cmake_minimum_required(VERSION 3.17)

project(isolated-batcher LANGUAGES C CXX)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all
# repos that we depend on so that we can override the tags.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND_ModelBatch API.
#
configure_file(src/libtriton_isolated_batcher.ldscript libtriton_isolated_batcher.ldscript COPYONLY)

add_library(
  triton-isolated-batcher SHARED
  src/isolated_batcher.cc
)

target_include_directories(
  triton-isolated-batcher
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(triton-isolated-batcher PRIVATE cxx_std_11)
target_compile_options(
  triton-isolated-batcher PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
)

target_link_libraries(
  triton-isolated-batcher
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
)

if(WIN32)
  set_target_properties(
    triton-isolated-batcher PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_isolated_batcher
  )
else()
  set_target_properties(
    triton-isolated-batcher PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_isolated_batcher
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_isolated_batcher.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_isolated_batcher.ldscript"
  )
endif()

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/IsolatedBatcher)

install(
  TARGETS
    triton-isolated-batcher
  EXPORT
    triton-isolated-batcher-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/batching/isolated_batcher
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/batching/isolated_batcher
)

install(
  EXPORT
    triton-isolated-batcher-targets
  FILE
    IsolatedBatcherTargets.cmake
  NAMESPACE
    triton-isolated-batcher
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/triton-isolated-batcher.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/triton-isolated-batcher.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/triton-isolated-batcher.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-isolated-batcher-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/triton-isolated-batcher.cmake
  NAMESPACE triton-isolated-batcher
)

export(PACKAGE triton-isolated-batcher)
