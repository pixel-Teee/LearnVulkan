cmake_minimum_required(VERSION 3.5)
project(VK)

file(GLOB_RECURSE VKSrc
"src/*.h"
"src/*.cpp"
"ThirdParty/glm/glm/*.h"
"ThirdParty/glm/glm/*.cpp"
"ThirdParty/stb/stb_image.h")

source_group(TREE ${CMAKE_SOURCE_DIR}/VK FILES ${VKSrc})

add_library(VK STATIC ${VKSrc})

target_include_directories(VK PRIVATE
"src"
"ThirdParty/glfw/include"
"ThirdParty/glm"
"ThirdParty/stb")

target_precompile_headers(VK
PUBLIC "src/pch.h"
)

#---Vulkan---
find_package(Vulkan REQUIRED)
target_link_libraries(${PROJECT_NAME}
PRIVATE ${Vulkan_LIBRARIES}
PRIVATE glfw
)
target_include_directories(${PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIR})