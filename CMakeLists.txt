# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# 设置CMake的最低版本要求为3.18
cmake_minimum_required(VERSION 3.18)

# 设置项目名称和版本号
# project(<project_name>
#         [VERSION <major>[.<minor>[.<patch>[.<tweak>]]]]
#         [DESCRIPTION <project_description>]
#         [HOMEPAGE_URL <url>]
#         [LANGUAGES <language_name>...])
# 这段代码的作用是设置项目的名称、版本号和描述。它使用CMake的`project`命令来定义项目的基本信息。
# 参数说明:
# - `instant-ngp` 是项目的名称。
# - `VERSION 1.0` 设置项目的版本号为1.0。
# - `DESCRIPTION "Instant Neural Graphics Primitives"` 给项目添加了一个描述，描述为"Instant Neural Graphics Primitives"。
# - `LANGUAGES C CXX CUDA` 指定项目支持的编程语言，包括C、C++和CUDA。
# 这些信息在构建过程中起到了多个作用，包括构建生成文件、生成配置文件、设置输出目录和生成项目文件等。同时，它们也提供了对项目的简要描述和版本号等元数据。
project(instant-ngp
	VERSION 1.0
	DESCRIPTION "Instant Neural Graphics Primitives"
	LANGUAGES C CXX CUDA
)

# 定义NGP的版本变量
# 这段代码的作用是定义了一个CMake变量NGP_VERSION，它的值被设置为${CMAKE_PROJECT_VERSION}。
# ${}是一种变量引用语法，用于引用和展开变量的值。它允许在CMake脚本中使用变量的值。
set(NGP_VERSION "${CMAKE_PROJECT_VERSION}")

# 如果NGP_DEPLOY变量未设置，则在NGP_VERSION后追加"dev"后缀
# 这段代码的目的是根据 NGP_DEPLOY 变量的设置，调整版本号。
# 可能是为了在开发阶段和部署阶段之间区分版本，或者在构建不同环境的构建过程中进行版本控制。
# 具体的用途取决于项目的需求和开发流程。
# 部署阶段就没有dev后缀，非部署（开发）阶段，就会有dev
if (NOT NGP_DEPLOY)
	set(NGP_VERSION "${NGP_VERSION}dev")
endif()

# 定义构建选项
# option(<option_variable> "<option_description>" <initial_value>)
# 参数说明：
# - `<option_variable>`：指定选项的变量名，用于在 CMake 中表示这个选项。
# - `"<option_description>"`：给选项提供一个描述，用于解释选项的作用和影响。
# - `<initial_value>`：指定选项的初始值，通常是 `ON` 或 `OFF`，表示是否启用或禁用选项。
# 在给定的代码中，定义了以下构建选项：
# - `NGP_BUILD_EXECUTABLE`：用于控制是否构建 `instant-ngp.exe` 可执行文件。
# - `NGP_BUILD_WITH_GUI`：用于控制是否构建带有 GUI 支持的版本，需要依赖 GLFW 和 GLEW。
# - `NGP_BUILD_WITH_OPTIX`：用于控制是否构建带有 OptiX 硬件光线追踪支持的版本。
# - `NGP_BUILD_WITH_PYTHON_BINDINGS`：用于控制是否构建允许使用 Python 进行绑定的版本。
# - `NGP_BUILD_WITH_VULKAN`：用于控制是否构建带有 Vulkan 支持以启用 DLSS（深度学习超采样）功能的版本。
# 这些选项可以根据项目的需求进行配置。开发人员可以根据需要启用或禁用特定功能，从而自定义构建过程和生成的可执行文件。
option(NGP_BUILD_EXECUTABLE "Build instant-ngp.exe?" ON)
option(NGP_BUILD_WITH_GUI "Build with GUI support (requires GLFW and GLEW)?" ON)
option(NGP_BUILD_WITH_OPTIX "Build with OptiX to enable hardware ray tracing?" ON)
option(NGP_BUILD_WITH_PYTHON_BINDINGS "Build bindings that allow instrumenting instant-ngp with Python?" ON)
option(NGP_BUILD_WITH_VULKAN "Build with Vulkan to enable DLSS support?" ON)


# 设置CMake模块路径
# set(<variable> <value> [CACHE <type> <docstring>])
# CMAKE_CURRENT_SOURCE_DIR 是一个 CMake 预定义的变量，表示正在处理的模块（CMakeLists.txt 文件）所在的目录路径。
# cmake中在设置路径变量时，不要使用引号，故${CMAKE_CURRENT_SOURCE_DIR}/cmake，就是一个路径变量
# ${CMAKE_MODULE_PATH}的初值为空
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

###############################################################################
# Build type and C++ compiler setup
###############################################################################

# 设置默认的构建类型为Release
# 将变量声明为CACHE，可以将变量的值缓存起来，以便在下一次CMake配置时保持不变。
# 在set命令中使用FORCE关键字，可以强制设置变量的值，覆盖任何已经存在的缓存值。
# 如果变量已经被缓存，并且没有使用FORCE关键字，那么后续的set操作将不会改变该变量的值。
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

# 检查依赖项库glfw的CMakeLists.txt文件是否存在
# 在instant-ngp库中，有很多的依赖项，依赖于已有的库
# 其图标为蓝色，表识如：glfw @ 71eb703
# 我们必须在git clone时加上--recursive命令，才能够将这些引用的依赖项的库给完整下载下来
# git clone --recursive https://github.com/NVlabs/instant-ngp.git
if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glfw/CMakeLists.txt")
	message(FATAL_ERROR
		"Some instant-ngp dependencies are missing. "
		"If you forgot the \"--recursive\" flag when cloning this project, "
		"this can be fixed by calling \"git submodule update --init --recursive\"."
	)
endif()

# 设置苹果系统下的RPATH
if (APPLE)
	set(CMAKE_MACOSX_RPATH ON)
endif()

# 设置MSVC编译器的编译选项
if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fms-extensions")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

# 设置C++标准为C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# CUDA compiler setup
###############################################################################


# 设置CUDA编译器的标准为CUDA 14
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# 针对不同编译器设置CUDA编译选项
if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fms-extensions")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

###############################################################################
# Dependencies
###############################################################################

# 添加依赖项
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
set(TCNN_ALLOW_CUBLAS_CUSOLVER OFF)
add_subdirectory(dependencies/tiny-cuda-nn)

# 设置CUDA的架构
set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})

# 检查是否构建GUI
if (NGP_BUILD_WITH_GUI)
	find_package(Vulkan)
	if (Vulkan_FOUND AND NGP_BUILD_WITH_VULKAN)
		set(NGP_VULKAN ON)
		list(APPEND NGP_DEFINITIONS -DNGP_VULKAN -DGLFW_INCLUDE_VULKAN)
		list(APPEND NGP_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIRS}")
		list(APPEND NGP_LIBRARIES ${Vulkan_LIBRARIES})

		list(APPEND GUI_SOURCES src/dlss.cu)

		# DLSS depends on vulkan, so appears here
		list(APPEND NGP_INCLUDE_DIRECTORIES "dependencies/dlss/include")
		if (MSVC)
			list(APPEND NGP_LINK_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dlss/lib/Windows_x86_64/x86_64")
			list(APPEND NGP_LIBRARIES "$<IF:$<CONFIG:Debug>,nvsdk_ngx_d_dbg,nvsdk_ngx_d>")
		else()
			list(APPEND NGP_LINK_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dlss/lib/Linux_x86_64")
			list(APPEND NGP_LIBRARIES nvsdk_ngx)
		endif()
	else()
		set(NGP_VULKAN OFF)
		if (NGP_BUILD_WITH_VULKAN)
			message(WARNING
				"Vulkan was not found. Instant neural graphics primitives will still "
				"compile and run correctly, but DLSS will not be supported."
			)
		endif()
	endif()

	# OpenXR
	if (WIN32)
		list(APPEND NGP_DEFINITIONS -DXR_USE_PLATFORM_WIN32 -DGLFW_EXPOSE_NATIVE_WGL)
	elseif (UNIX AND NOT APPLE)
		list(APPEND NGP_DEFINITIONS -DGLFW_EXPOSE_NATIVE_GLX)
		if (JK_USE_WAYLAND)
			set(PRESENTATION_BACKEND wayland CACHE STRING " " FORCE)
			set(BUILD_WITH_XLIB_HEADERS OFF CACHE BOOL " " FORCE)
			set(BUILD_WITH_XCB_HEADERS OFF CACHE BOOL " " FORCE)
			set(BUILD_WITH_WAYLAND_HEADERS ON CACHE BOOL " " FORCE)
			list(APPEND NGP_DEFINITIONS -DGLFW_EXPOSE_NATIVE_WAYLAND -DXR_USE_PLATFORM_WAYLAND)
		else()
			set(PRESENTATION_BACKEND xlib CACHE STRING " " FORCE)
			set(BUILD_WITH_XLIB_HEADERS ON CACHE BOOL " " FORCE)
			set(BUILD_WITH_XCB_HEADERS OFF CACHE BOOL " " FORCE)
			set(BUILD_WITH_WAYLAND_HEADERS OFF CACHE BOOL " " FORCE)
			list(APPEND NGP_DEFINITIONS -DGLFW_EXPOSE_NATIVE_X11 -DXR_USE_PLATFORM_XLIB)
		endif()
	else()
		message(FATAL_ERROR "No OpenXR platform set for this OS")
	endif()

	add_subdirectory(dependencies/OpenXR-SDK)

	list(APPEND NGP_INCLUDE_DIRECTORIES "dependencies/OpenXR-SDK/include" "dependencies/OpenXR-SDK/src/common")
	list(APPEND NGP_LIBRARIES openxr_loader)
	list(APPEND GUI_SOURCES src/openxr_hmd.cu)

	# OpenGL
	find_package(OpenGL REQUIRED)

	# GLFW
	set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
	set(GLFW_BUILD_TESTS OFF CACHE BOOL " " FORCE)
	set(GLFW_BUILD_DOCS OFF CACHE BOOL " " FORCE)
	set(GLFW_BUILD_INSTALL OFF CACHE BOOL " " FORCE)
	set(GLFW_INSTALL OFF CACHE BOOL " " FORCE)
	set(GLFW_USE_CHDIR OFF CACHE BOOL " " FORCE)
	set(GLFW_VULKAN_STATIC OFF CACHE BOOL " " FORCE)
	set(BUILD_SHARED_LIBS ON CACHE BOOL " " FORCE)

	add_subdirectory(dependencies/glfw)

	set_target_properties(glfw PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)

	mark_as_advanced(
		GLFW_BUILD_DOCS GLFW_BUILD_EXAMPLES GLFW_BUILD_INSTALL GLFW_BUILD_TESTS
		GLFW_DOCUMENT_INTERNALS GLFW_INSTALL GLFW_USE_CHDIR GLFW_USE_MENUBAR
		GLFW_USE_OSMESA GLFW_VULKAN_STATIC GLFW_USE_RETINA GLFW_USE_MIR
		BUILD_SHARED_LIBS USE_MSVC_RUNTIME_LIBRARY_DLL
	)

	list(APPEND NGP_INCLUDE_DIRECTORIES "dependencies/glfw/include" "dependencies/imgui")

	if (MSVC)
		list(APPEND NGP_INCLUDE_DIRECTORIES "dependencies/gl3w")
		list(APPEND GUI_SOURCES "dependencies/gl3w/GL/gl3w.c")
		list(APPEND NGP_LIBRARIES opengl32 $<TARGET_OBJECTS:glfw_objects>)
	else()
		find_package(GLEW REQUIRED)
		list(APPEND NGP_INCLUDE_DIRECTORIES ${GLEW_INCLUDE_DIRS})
		list(APPEND NGP_LIBRARIES GL ${GLEW_LIBRARIES} $<TARGET_OBJECTS:glfw_objects>)
	endif()

	list(APPEND GUI_SOURCES
		dependencies/imguizmo/ImGuizmo.cpp
		dependencies/imgui/imgui.cpp
		dependencies/imgui/backends/imgui_impl_glfw.cpp
		dependencies/imgui/backends/imgui_impl_opengl3.cpp
		dependencies/imgui/imgui_draw.cpp
		dependencies/imgui/imgui_tables.cpp
		dependencies/imgui/imgui_widgets.cpp
	)

	list(APPEND NGP_DEFINITIONS -DNGP_GUI)
endif(NGP_BUILD_WITH_GUI)

# 添加其他依赖项的路径
list(APPEND NGP_INCLUDE_DIRECTORIES
	"dependencies"
	"dependencies/filesystem"
	"dependencies/glm"
	"dependencies/nanovdb"
	"dependencies/NaturalSort"
	"dependencies/tinylogger"
)

# 查找并使用OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 如果构建选项为NGP_BUILD_WITH_OPTIX，则添加OptiX相关依赖项
if (NGP_BUILD_WITH_OPTIX)
	set(NGP_OPTIX ON)
	list(APPEND NGP_INCLUDE_DIRECTORIES "dependencies/optix")
	list(APPEND NGP_DEFINITIONS -DNGP_OPTIX)
endif()


if (NGP_BUILD_WITH_PYTHON_BINDINGS)
	find_package(Python 3.7 COMPONENTS Interpreter Development)
	if (Python_FOUND)
		add_subdirectory("dependencies/pybind11")
	endif()
endif()


# Compile zlib (only on Windows)
if (WIN32)
	set(ZLIB_USE_STATIC_LIBS ON CACHE BOOL " " FORCE)
	set(ZLIB_BUILD_STATIC_LIBS ON CACHE BOOL " " FORCE)
	set(ZLIB_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
	set(SKIP_INSTALL_ALL ON CACHE BOOL " " FORCE)
	add_subdirectory("dependencies/zlib")
	set_property(TARGET zlibstatic PROPERTY FOLDER "dependencies")

	set(ZLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/zlib" CACHE PATH " " FORCE)
	set(ZLIB_LIBRARY zlibstatic)

	include_directories(${ZLIB_INCLUDE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/dependencies/zlib")

	list(APPEND NGP_LIBRARIES zlibstatic)
endif()


add_subdirectory("dependencies/zstr")
list(APPEND NGP_LIBRARIES zstr::zstr)


###############################################################################
# Program
###############################################################################

# 设置程序的编译选项和源文件
list(APPEND NGP_DEFINITIONS -DNGP_VERSION="${NGP_VERSION}")
list(APPEND NGP_INCLUDE_DIRECTORIES "include")
if (NOT MSVC)
	list(APPEND NGP_LIBRARIES ${CMAKE_DL_LIBS})
endif()
list(APPEND NGP_SOURCES
	${GUI_SOURCES}
	src/camera_path.cu
	src/common.cu
	src/common_device.cu
	src/marching_cubes.cu
	src/nerf_loader.cu
	src/render_buffer.cu
	src/testbed.cu
	src/testbed_image.cu
	src/testbed_nerf.cu
	src/testbed_sdf.cu
	src/testbed_volume.cu
	src/thread_pool.cpp
	src/tinyexr_wrapper.cu
	src/tinyobj_loader_wrapper.cpp
	src/triangle_bvh.cu
)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

# 获取CUDA编译器的路径
get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)

# 如果构建选项为NGP_OPTIX，则添加optix_program目标
if (NGP_OPTIX)
	add_library(optix_program OBJECT
		src/optix/pathescape.cu
		src/optix/raystab.cu
		src/optix/raytrace.cu
	)

	set_target_properties(optix_program PROPERTIES CUDA_PTX_COMPILATION ON CUDA_ARCHITECTURES OFF)
	target_compile_definitions(optix_program PUBLIC ${NGP_DEFINITIONS} -DTCNN_MIN_GPU_ARCH=0)
	target_compile_options(optix_program PUBLIC "--expt-relaxed-constexpr")

	get_target_property(TCNN_INCLUDE_DIRECTORIES tiny-cuda-nn INCLUDE_DIRECTORIES)
	target_include_directories(optix_program PUBLIC ${NGP_INCLUDE_DIRECTORIES} ${TCNN_INCLUDE_DIRECTORIES})

	# OptiX programs will be compiled as PTX and packaged
	# as headers to be included from the binary dir.
	list(APPEND NGP_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}")
	set(OPTIX_PTX_HEADER ${CMAKE_CURRENT_BINARY_DIR}/optix_ptx.h)

	find_program(bin_to_c NAMES bin2c PATHS ${CUDA_COMPILER_BIN})
	if (NOT bin_to_c)
		message(FATAL_ERROR
			"bin2c not found:\n"
			"  CMAKE_CUDA_COMPILER='${CMAKE_CUDA_COMPILER}'\n"
			"  CUDA_COMPILER_BIN='${CUDA_COMPILER_BIN}'\n"
		)
	endif()

	add_custom_command(
		OUTPUT "${OPTIX_PTX_HEADER}"
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS=$<TARGET_OBJECTS:optix_program>"
		"-DOUTPUT=${OPTIX_PTX_HEADER}"
		-P ${PROJECT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS optix_program
		COMMENT "Converting PTX files to a C header"
	)

	list(APPEND NGP_SOURCES ${OPTIX_PTX_HEADER})
endif()

# 添加ngp目标
add_library(ngp STATIC ${NGP_SOURCES})
set_target_properties(ngp PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(ngp PUBLIC ${NGP_DEFINITIONS})
target_compile_options(ngp PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_include_directories(ngp PUBLIC ${NGP_INCLUDE_DIRECTORIES})
target_link_directories(ngp PUBLIC ${NGP_LINK_DIRECTORIES})
target_link_libraries(ngp PUBLIC ${NGP_LIBRARIES} tiny-cuda-nn)

# 如果构建选项为NGP_VULKAN，则将共享库复制到二进制目录
# Copy shared libraries to the binary directory as needed
if (NGP_VULKAN)
	set(NGX_BUILD_DIR "$<IF:$<CONFIG:Debug>,dev,rel>")
	if (MSVC)
		set(NGX_SHARED_LIB "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dlss/lib/Windows_x86_64/${NGX_BUILD_DIR}/nvngx_dlss.dll")
	else()
		set(NGX_SHARED_LIB "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dlss/lib/Linux_x86_64/${NGX_BUILD_DIR}/libnvidia-ngx-dlss.so.*")
	endif()

	add_custom_command(TARGET ngp POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${NGX_SHARED_LIB}" "${CMAKE_CURRENT_BINARY_DIR}" COMMAND_EXPAND_LISTS)
endif()

# 如果是MSVC编译器，则复制相应的CUDA DLL文件
if (MSVC)
	file(GLOB CUDA_DLLS "${CUDA_COMPILER_BIN}/cudart64*.dll")
	if (CUDA_DLLS)
		add_custom_command(TARGET ngp POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CUDA_DLLS} "${CMAKE_CURRENT_BINARY_DIR}" COMMAND_EXPAND_LISTS)
	endif()
endif()

# 如果构建选项为NGP_BUILD_EXECUTABLE，则添加instant-ngp可执行目标
if (NGP_BUILD_EXECUTABLE)
	add_executable(instant-ngp src/main.cu)
	target_link_libraries(instant-ngp PRIVATE ngp)

	# Link the executable to the project directory and copy over DLLs such that instant-ngp can be invoked without going into the build folder.
	set(NGP_BINARY_FILE "\"${CMAKE_CURRENT_SOURCE_DIR}/$<TARGET_FILE_NAME:instant-ngp>\"")
	if (MSVC)
		add_custom_command(TARGET instant-ngp POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:instant-ngp> ${CMAKE_CURRENT_SOURCE_DIR})
		file(GLOB NGP_REQUIRED_DLLS "${CUDA_COMPILER_BIN}/cudart64*.dll")
		if (NGP_VULKAN)
			list(APPEND NGP_REQUIRED_DLLS "${NGX_SHARED_LIB}")
		endif()
		if (NGP_REQUIRED_DLLS)
			add_custom_command(TARGET instant-ngp POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${NGP_REQUIRED_DLLS} ${CMAKE_CURRENT_SOURCE_DIR} COMMAND_EXPAND_LISTS)
		endif()
	else()
		add_custom_command(TARGET instant-ngp POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:instant-ngp> "${NGP_BINARY_FILE}")
	endif()
endif(NGP_BUILD_EXECUTABLE)

# 如果Python_FOUND为真，则添加pyngp目标
if (Python_FOUND)
	add_library(pyngp SHARED src/python_api.cu)
	set_target_properties(pyngp PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")
	target_link_libraries(pyngp PRIVATE ngp PUBLIC ${PYTHON_LIBRARIES} pybind11::module)
	target_compile_definitions(pyngp PUBLIC -DNGP_PYTHON)
	pybind11_extension(pyngp)
endif()
