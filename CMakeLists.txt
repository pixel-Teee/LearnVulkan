cmake_minimum_required(VERSION 3.5)

project(VK)

# executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# module libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
# static libraries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(VK/ThirdParty/glfw)
add_subdirectory(VK)
add_subdirectory(App)

target_compile_features(VK
    PUBLIC cxx_std_17
)