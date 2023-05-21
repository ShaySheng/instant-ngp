# 版权所有（C）2021-2022年，NVIDIA CORPORATION。保留所有权利。
# NVIDIA CORPORATION及其许可方保留此软件、相关文档及其任何修改
# 的所有知识产权和专有权。未经NVIDIA CORPORATION明确许可协议的
# 书面同意，严禁使用、复制、披露或分发本软件和相关文档。

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
    # 初始化了CMAKE_BUILD_TYPE这个变量
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # set_property用于限定属性的作用域，也就是让CMAKE_BUILD_TYPE只能在后面的三者中取值
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
# RPATH，即Runtime Path，即运行时搜索路径
# 在苹果操作系统上运行一个程序时，它可能会依赖于其他的库文件。
# 为了确保程序能够找到这些库文件，需要设置一个运行时搜索路径。
# 通过这个RPATH路径，程序就能够找到所需的库文件，而不必担心它们是否位于默认位置。
if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()

# 若使用MSVC编译器（Microsoft Visual C++）的情况
# /D_CRT_SECURE_NO_WARNINGS：这个选项用于禁止警告相关于使用不安全的函数（如strcpy、sprintf等）。
# 通过设置这个选项，编译器将不会产生与这些函数相关的警告信息。
# /MP24：这个选项用于启用并行编译，允许使用多个处理器核心并行地编译源文件。
# /MP24表示使用24个并行编译任务。这可以加快编译速度。
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
# 其他编译器（非MSVC）
# -fms-extensions：这个选项用于启用编译器对Microsoft特定的扩展语法的支持。
# 这样可以在源代码中使用一些Microsoft特有的语法和功能。
# -fPIC：这个选项用于生成与位置无关的代码（Position Independent Code），
# 以便在共享库（动态链接库）中使用。它是一种编译选项，使得生成的代码可以在内存中的任意位置加载和执行，
# 而不依赖于特定的内存地址。这对于共享库的使用非常重要。
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fms-extensions")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

# 设置C++标准为C++14
set(CMAKE_CXX_STANDARD 14)
# 用于控制不启用 C++ 扩展
# C++ 扩展是一些与标准 C++ 不兼容的语言特性或语法，
# 它们通常是由编译器提供的特定扩展。
# 禁用 C++ 扩展有助于提高代码的可移植性和标准符合性。
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# CUDA compiler setup
###############################################################################


# 设置CUDA编译器的标准为CUDA 14
set(CMAKE_CUDA_STANDARD 14)
# 要求必须使用指定的 CUDA 标准版本，否则报错
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# 不启用 CUDA 扩展
set(CMAKE_CUDA_EXTENSIONS OFF)
# 指定在链接 CUDA 库时使用的关键字
# PRIVATE：表示将库作为私有依赖项进行链接。
# 私有依赖项只对当前目标有效，不会传递给依赖该目标的其他目标。
# INTERFACE：表示将库作为接口依赖项进行链接。
# 接口依赖项只会传递给依赖该目标的其他目标，而不会在当前目标中使用。
# PUBLIC：表示将库作为公共依赖项进行链接。
# 公共依赖项既会对当前目标生效，也会传递给依赖该目标的其他目标。
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
# CUDA 运行时库的选项
# CUDA Driver API 提供了对底层 CUDA 驱动程序功能的直接访问。
# CUDA Runtime Library 是一个高级的编程接口，封装了 CUDA Driver API。
# Shared 选项表示将 CUDA 运行时库配置为共享库（Shared Library）的形式。
# 共享库是一种可在多个程序中共享使用的库形式，它可以在运行时被动态加载和链接。
# 减小可执行文件的大小；节省内存空间；动态加载和链接
# Static：表示使用静态库形式的 CUDA 运行时库。
# 静态库是编译时将库代码嵌入到可执行文件中的库形式，
# 使得可执行文件在运行时不需要依赖外部的 CUDA 运行时库文件。
# None：表示不使用 CUDA 运行时库。
# 这意味着 CUDA 程序必须通过显式地加载和管理 CUDA 运行时功能，
# 例如使用 CUDA Driver API 进行编程。
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# 针对不同编译器设置CUDA编译选项
# list是一个用于处理列表的命令，用于添加、删除、获取和修改列表中的元素。
# 添加元素到列表：list(APPEND my_list element1 element2 element3)
# 获取列表的长度：list(LENGTH my_list length)
# 获取列表中指定位置的元素：list(GET my_list index output_variable)
# 删除列表中的元素：list(REMOVE_ITEM my_list element)
# 替换列表中的元素：list(REPLACE my_list index new_element)
# 迭代列表中的元素：
# foreach(element IN LISTS my_list)
#     # 使用元素执行操作
#     ...
# endforeach()
# NVCC（NVIDIA CUDA Compiler）是 NVIDIA 
# 提供的用于编译和构建 CUDA C/C++ 代码的编译器。
if (MSVC)
    # 大型项目时，MSVC 默认的对象文件格式可能会导致链接错误。
    # 这是由于 MSVC 对象文件格式所支持的符号表大小有限制。
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/bigobj")
else()
    # 禁止显示浮点数转换的警告。
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
    # 禁用严格别名规则，允许不同类型的指针相互转换。
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
    # 启用 Microsoft 扩展，允许使用一些 MSVC 特定的语法扩展。
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fms-extensions")
    # 生成位置无关代码，用于支持共享库的编译。
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
endif()
# 允许在 CUDA C++ 中使用扩展的 lambda 表达式。
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
# 启用对松散常量表达式（relaxed constexpr）的支持
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
# 启用快速数学运算的优化, 可能会影响计算结果的精度:
# 1.将某些精度要求较高的计算替换为更快但精度稍低的算法，以加快计算速度。
# 2.使用近似算法，例如使用快速的三角函数和指数函数实现。
# 3.允许并行化和向量化优化，以利用 GPU 的并行计算能力。
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

###############################################################################
# Dependencies
###############################################################################

# 添加依赖项
# 禁用了构建基准测试的选项
# 基准测试：评估计算机系统或软件性能的方法
# 	运行一组标准化的测试任务或程序来测量系统
# 	的处理能力、响应时间、吞吐量等性能指标
set(TCNN_BUILD_BENCHMARK OFF)
# 禁用了构建示例程序的选项
# 示例程序：编译和生成一些演示性质的程序
# 	用于展示库、框架或工具的使用方法和功能。
set(TCNN_BUILD_EXAMPLES OFF)
# 禁用了使用CuBLAS和CuSolver库的选项。
# 这意味着构建过程将不会使用这些库进行加速。
set(TCNN_ALLOW_CUBLAS_CUSOLVER OFF)
# 添加了tiny-cuda-nn库作为依赖项
add_subdirectory(dependencies/tiny-cuda-nn)

# 设置CUDA的架构
set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})

# 检查是否构建GUI
if (NGP_BUILD_WITH_GUI)
    # 查找Vulkan库
    find_package(Vulkan)
    # # 检查Vulkan库是否被找到并且NGP_BUILD_WITH_VULKAN为真
    if (Vulkan_FOUND AND NGP_BUILD_WITH_VULKAN)
        set(NGP_VULKAN ON)
        # 添加宏定义
        list(APPEND NGP_DEFINITIONS -DNGP_VULKAN -DGLFW_INCLUDE_VULKAN)
        # 添加包含目录
        list(APPEND NGP_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIRS}")
        # 添加链接库
        list(APPEND NGP_LIBRARIES ${Vulkan_LIBRARIES})

        # 添加GUI相关源文件
        list(APPEND GUI_SOURCES src/dlss.cu)

        # DLSS依赖于Vulkan，因此出现在这里
        # DLSS（Deep Learning Super Sampling）
        # 	是一种基于深度学习的超级采样技术，由NVIDIA开发。
        list(APPEND NGP_INCLUDE_DIRECTORIES "dependencies/dlss/include")
        
        # 根据操作系统设置链接目录和库文件
        # MSVC（Microsoft Visual C++）
        if (MSVC)
            # 这个路径用于指定在Windows系统构建过程中所需的 DLSS 库的位置。
            list(APPEND NGP_LINK_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dlss/lib/Windows_x86_64/x86_64")
            # $<CONFIG:Debug> 是一个条件判断，检查构建配置是否为 Debug 模式
            # 条件为真，生成器表达式将返回 nvsdk_ngx_d_dbg，表示 Debug 模式下的库文件名称。
            # 条件为假，生成器表达式将返回 nvsdk_ngx_d，表示非 Debug 模式下的库文件名称。
            list(APPEND NGP_LIBRARIES "$<IF:$<CONFIG:Debug>,nvsdk_ngx_d_dbg,nvsdk_ngx_d>")
        else()
            # 这个路径用于指定在Linux系统构建过程中所需的 DLSS 库的位置。
            list(APPEND NGP_LINK_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dlss/lib/Linux_x86_64")
            # 添加库
            list(APPEND NGP_LIBRARIES nvsdk_ngx)
        endif()
    else()
        # 若未安装VULKAN，则设置NGP_VULKAN为OFF
        set(NGP_VULKAN OFF)
        # 如果NGP_BUILD_WITH_VULKAN为真，
        #   并且找不到Vulkan库，则发出警告消息
        if (NGP_BUILD_WITH_VULKAN)
            message(WARNING
                "Vulkan was not found. Instant neural graphics primitives will still "
                "compile and run correctly, but DLSS will not be supported."
            )
        endif()
    endif()

    # OpenXR
    # OpenXR是专注于虚拟和增强现实领域的API
    # 在OpenXR中，可以选择使用不同的平台后端，包括Wayland和Xlib。
    if (WIN32)
        # 如果是Windows平台，添加宏定义，
        #   指定使用Win32平台和GLFW的本地WGL（Windows图形库）扩展
        list(APPEND NGP_DEFINITIONS -DXR_USE_PLATFORM_WIN32 -DGLFW_EXPOSE_NATIVE_WGL)
    elseif (UNIX AND NOT APPLE)
        # 如果是UNIX平台（非苹果平台），
        #   添加宏定义，指定使用GLX（OpenGL扩展）
        #   和GLFW的本地GLX扩展
        list(APPEND NGP_DEFINITIONS -DGLFW_EXPOSE_NATIVE_GLX)
        # 如果使用Wayland作为显示服务器，设置相关的配置选项
        # Wayland是一种计算机图形显示协议, 被广泛用于许多Linux发行版
        if (JK_USE_WAYLAND)
            # 将PRESENTATION_BACKEND变量设置为wayland，指定使用Wayland作为显示服务器
            set(PRESENTATION_BACKEND wayland CACHE STRING " " FORCE)
            # 将BUILD_WITH_XLIB_HEADERS变量设置为OFF，禁用Xlib头文件的构建
            set(BUILD_WITH_XLIB_HEADERS OFF CACHE BOOL " " FORCE)
            # 将BUILD_WITH_XCB_HEADERS变量设置为OFF，禁用XCB头文件的构建
            set(BUILD_WITH_XCB_HEADERS OFF CACHE BOOL " " FORCE)
            # 将BUILD_WITH_WAYLAND_HEADERS变量设置为ON，启用Wayland头文件的构建
            set(BUILD_WITH_WAYLAND_HEADERS ON CACHE BOOL " " FORCE)
            # 向NGP_DEFINITIONS列表添加宏定义，
            #   指定GLFW暴露原生的Wayland扩展和使用Wayland作为OpenXR的平台
            list(APPEND NGP_DEFINITIONS -DGLFW_EXPOSE_NATIVE_WAYLAND -DXR_USE_PLATFORM_WAYLAND)
        # 否则使用xlib作为图形系统接口库
        else()
            # 将PRESENTATION_BACKEND变量设置为xlib，
            #   指定使用Xlib作为显示服务器
            set(PRESENTATION_BACKEND xlib CACHE STRING " " FORCE)
            # 将BUILD_WITH_XLIB_HEADERS变量设置为ON，启用Xlib头文件的构建
            set(BUILD_WITH_XLIB_HEADERS ON CACHE BOOL " " FORCE)
            # 将BUILD_WITH_XCB_HEADERS变量设置为OFF，禁用XCB头文件的构建
            set(BUILD_WITH_XCB_HEADERS OFF CACHE BOOL " " FORCE)
            # 将BUILD_WITH_WAYLAND_HEADERS变量设置为OFF，
            #   禁用Wayland头文件的构建
            set(BUILD_WITH_WAYLAND_HEADERS OFF CACHE BOOL " " FORCE)
            # 向NGP_DEFINITIONS列表添加宏定义，
            #   指定GLFW暴露原生的X11扩展和使用Xlib作为OpenXR的平台
            list(APPEND NGP_DEFINITIONS -DGLFW_EXPOSE_NATIVE_X11 -DXR_USE_PLATFORM_XLIB)
        endif()
    else()
        # 如果不是Windows也不是UNIX平台，则抛出致命错误
        message(FATAL_ERROR "No OpenXR platform set for this OS")
    endif()
    # 添加OpenXR-SDK子目录进行构建
    add_subdirectory(dependencies/OpenXR-SDK)
    # 添加OpenXR-SDK相关的包含目录和链接库
    list(APPEND NGP_INCLUDE_DIRECTORIES "dependencies/OpenXR-SDK/include" "dependencies/OpenXR-SDK/src/common")
    list(APPEND NGP_LIBRARIES openxr_loader)
    list(APPEND GUI_SOURCES src/openxr_hmd.cu)

    # OpenGL
    # 查找OpenGL库
    find_package(OpenGL REQUIRED)
    # GLFW：用于创建窗口、处理用户输入、获取
    #   OpenGL上下文以及管理OpenGL的扩展和版本
    # 配置GLFW选项
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL " " FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL " " FORCE)
    set(GLFW_BUILD_INSTALL OFF CACHE BOOL " " FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL " " FORCE)
    set(GLFW_USE_CHDIR OFF CACHE BOOL " " FORCE)
    set(GLFW_VULKAN_STATIC OFF CACHE BOOL " " FORCE)
    set(BUILD_SHARED_LIBS ON CACHE BOOL " " FORCE)
    # 添加glfw子目录进行构建
    add_subdirectory(dependencies/glfw)
    # 设置glfw目标属性
    set_target_properties(glfw PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
    # 标记某些glfw选项为高级选项
    mark_as_advanced(
        GLFW_BUILD_DOCS GLFW_BUILD_EXAMPLES GLFW_BUILD_INSTALL GLFW_BUILD_TESTS
        GLFW_DOCUMENT_INTERNALS GLFW_INSTALL GLFW_USE_CHDIR GLFW_USE_MENUBAR
        GLFW_USE_OSMESA GLFW_VULKAN_STATIC GLFW_USE_RETINA GLFW_USE_MIR
        BUILD_SHARED_LIBS USE_MSVC_RUNTIME_LIBRARY_DLL
    )
    # 添加glfw和imgui的包含目录
    # imgui是一个轻量级的、即时模式的图形用户界面库
    # 设计思想是即时模式，即在每一帧渲染时，
    # 应用程序可以根据当前状态直接构建和绘制UI，而无需维护复杂的UI层次结构
    list(APPEND NGP_INCLUDE_DIRECTORIES "dependencies/glfw/include" "dependencies/imgui")

    if (MSVC)
        # 添加gl3w的包含目录
        # gl3w是一个用于加载和管理OpenGL函数指针的库
        list(APPEND NGP_INCLUDE_DIRECTORIES "dependencies/gl3w")
        # 添加gl3w的源文件
        list(APPEND GUI_SOURCES "dependencies/gl3w/GL/gl3w.c")
        # 添加OpenGL和glfw的链接库
        list(APPEND NGP_LIBRARIES opengl32 $<TARGET_OBJECTS:glfw_objects>)
    else()
        # 在非MSVC平台上，查找并添加GLEW库的依赖
        find_package(GLEW REQUIRED)
        list(APPEND NGP_INCLUDE_DIRECTORIES ${GLEW_INCLUDE_DIRS})
        list(APPEND NGP_LIBRARIES GL ${GLEW_LIBRARIES} $<TARGET_OBJECTS:glfw_objects>)
    endif()

    # 添加GUI相关的源文件
    list(APPEND GUI_SOURCES
        dependencies/imguizmo/ImGuizmo.cpp
        dependencies/imgui/imgui.cpp
        dependencies/imgui/backends/imgui_impl_glfw.cpp
        dependencies/imgui/backends/imgui_impl_opengl3.cpp
        dependencies/imgui/imgui_draw.cpp
        dependencies/imgui/imgui_tables.cpp
        dependencies/imgui/imgui_widgets.cpp
    )

    # 添加NGP_GUI宏定义
    list(APPEND NGP_DEFINITIONS -DNGP_GUI)
endif(NGP_BUILD_WITH_GUI)

# 添加其他依赖项的路径
# filesystem：文件系统库，用于处理文件和目录的操作，
#   如路径解析、文件查找、文件创建和删除等
# glm：用于数学计算和线性代数的库
# nanovdb：用于稀疏体数据结构的库
# NaturalSort：提供了一种自然排序算法，用于对字符串进行排序
# tinylogger：轻量级的日志记录库，用于在应用程序中进行日志记录
list(APPEND NGP_INCLUDE_DIRECTORIES
    "dependencies"
    "dependencies/filesystem"
    "dependencies/glm"
    "dependencies/nanovdb"
    "dependencies/NaturalSort"
    "dependencies/tinylogger"
)

# 查找并使用OpenMP：
#   是一个并行编程的标准和接口，用于编写多线程和共享内存并行程序
#   将串行代码转换为并行代码，以实现多线程的并行执行
find_package(OpenMP)
if (OPENMP_FOUND)
    # 如果找到OpenMP，则将相应的编译标志添加到C和C++编译器选项中
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 如果构建选项为NGP_BUILD_WITH_OPTIX，则添加OptiX相关依赖项
if (NGP_BUILD_WITH_OPTIX)
    set(NGP_OPTIX ON)
    # 添加OptiX依赖项的头文件路径
    list(APPEND NGP_INCLUDE_DIRECTORIES "dependencies/optix")
    # 添加NGP_OPTIX宏定义
    list(APPEND NGP_DEFINITIONS -DNGP_OPTIX)
endif()


# 如果构建选项为NGP_BUILD_WITH_PYTHON_BINDINGS，
#   则查找Python并添加pybind11依赖项
# pybind11是一个用于将C++代码绑定到Python的开源库
#   使得开发者能够直接在C++代码中定义Python模块、类和函数，
#   并实现C++代码与Python之间的无缝交互。
if (NGP_BUILD_WITH_PYTHON_BINDINGS)
    # 查找Python的Interpreter和Development组件
    find_package(Python 3.7 COMPONENTS Interpreter Development)
    if (Python_FOUND)
        # 添加pybind11的子目录进行构建
        add_subdirectory("dependencies/pybind11")
    endif()
endif()

# 编译zlib（仅在Windows平台）
# zlib是一个用于数据压缩和解压缩的开源库
if (WIN32)
    # 设置使用静态链接的zlib选项
    set(ZLIB_USE_STATIC_LIBS ON CACHE BOOL " " FORCE)
    set(ZLIB_BUILD_STATIC_LIBS ON CACHE BOOL " " FORCE)
    set(ZLIB_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
    set(SKIP_INSTALL_ALL ON CACHE BOOL " " FORCE)
    # 添加zlib的子目录进行构建
    add_subdirectory("dependencies/zlib")
    # 设置zlib的目标文件夹属性
    set_property(TARGET zlibstatic PROPERTY FOLDER "dependencies")

    # 设置zlib的头文件和库路径
    set(ZLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/zlib" CACHE PATH " " FORCE)
    set(ZLIB_LIBRARY zlibstatic)

    # 添加zlib的头文件路径
    include_directories(${ZLIB_INCLUDE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/dependencies/zlib")

    # 将zlibstatic库添加到NGP_LIBRARIES列表中
    # zlibstatic是zlib库的静态链接库（Static Linking）
    list(APPEND NGP_LIBRARIES zlibstatic)
endif()


# 添加zstr的子目录进行构建
# zstr是一个开源的C++库，
#   用于在处理文本数据时进行高效的压缩和解压缩操作。
add_subdirectory("dependencies/zstr")
# 将zstr库添加到NGP_LIBRARIES列表中
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
# 添加GUI_SOURCES变量中的源文件路径到NGP_SOURCES列表中
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
# 设置可执行文件的输出目录为CMAKE_BINARY_DIR
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# 设置Release配置下的可执行文件的输出目录为CMAKE_BINARY_DIR
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
# 设置RelWithDebInfo配置下的可执行文件的输出目录为CMAKE_BINARY_DIR
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
# 设置MinSizeRel配置下的可执行文件的输出目录为CMAKE_BINARY_DIR
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
# 设置Debug配置下的可执行文件的输出目录为CMAKE_BINARY_DIR
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
