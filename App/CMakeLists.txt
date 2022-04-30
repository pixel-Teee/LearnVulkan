cmake_minimum_required(VERSION 3.5)
project(App)

file(GLOB_RECURSE AppSrc
"src/*.h"
"src/*.cpp"
)

source_group(TREE ${CMAKE_SOURCE_DIR}/App FILES ${AppSrc})

add_executable(App ${AppSrc})

target_link_libraries(App PRIVATE VK)

target_include_directories(App PRIVATE
"../VK/src"
"../VK/ThirdParty/glm"
"../VK/ThirdParty/glfw/include")

set_target_properties(
	App PROPERTIES
	VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/App")