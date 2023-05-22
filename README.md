# Instant Neural Graphics Primitives ![](https://github.com/NVlabs/instant-ngp/workflows/CI/badge.svg)

<img src="docs/assets_readme/fox.gif" height="342"/> <img src="docs/assets_readme/robot5.gif" height="342"/>

曾经想过在不到5秒钟内训练一只狐狸的NeRF模型吗？或者在由工厂机器人的照片捕捉的场景中飞行？当然你想过！

在这里，您将找到四个神经图形基元的实现，包括神经辐射场（NeRF）、有符号距离函数（SDF）、神经图像和神经体积。
在每种情况下，我们使用[__tiny-cuda-nn__](https://github.com/NVlabs/tiny-cuda-nn)框架训练和渲染一个具有多分辨率哈希输入编码的MLP。

> __使用多分辨率哈希编码的即时神经图形基元__  
> [Thomas Müller](https://tom94.net)，[Alex Evans](https://research.nvidia.com/person/alex-evans)，[Christoph Schied](https://research.nvidia.com/person/christoph-schied)，[Alexander Keller](https://research.nvidia.com/person/alex-keller)  
> _ACM计算机图形学交易（__SIGGRAPH__），2022年7月_  
> __[项目页面](https://nvlabs.github.io/instant-ngp)&nbsp;/ [论文](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)&nbsp;/ [视频](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.mp4)&nbsp;/ [演示](https://tom94.net/data/publications/mueller22instant/mueller22instant-gtc.mp4)&nbsp;/ [实时直播](https://tom94.net/data/publications/mueller22instant/mueller22instant-rtl.mp4)&nbsp;/ [BibTeX](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.bib)__

有关业务合作，请填写[NVIDIA研究许可表格](https://www.nvidia.com/en-us/research/inquiries/)。

## 安装


务必使用此命令下载完整代码

```git
git clone --recursive https://github.com/ShaySheng/instant-ngp.git
```

如果您使用的是Windows操作系统，请下载与您的显卡对应的以下发布版本，并进行解压。然后，启动`instant-ngp.exe`。

- [**RTX 3000和4000系列、RTX A4000-A6000**和其他Ampere和Ada显卡](https://github.com/NVlabs/instant-ngp/releases/download/continuous/Instant-NGP-for-RTX-3000-and-4000.zip)
- [**RTX 2000系列、Titan RTX、Quadro RTX 4000-8000**和其他图灵显卡](https://github.com/NVlabs/instant-ngp/releases/download/continuous/Instant-NGP-for-RTX-2000.zip)
- [**GTX 1000系列、Titan Xp、Quadro P1000-P6000**和其他帕斯卡显卡](https://github.com/NVlabs/instant-ngp/releases/download/continuous/Instant-NGP-for-GTX-1000.zip)

继续阅读以了解应用程序的导览，或者如果您有兴趣创建自己的NeRF，请观看[视频教程](https://www.youtube.com/watch?v=3TWxO1PftMc)或阅读[书面说明](docs/nerf_dataset_tips.md)。

如果您使用Linux，或者想要[开发者Python绑定](https://github.com/NVlabs/instant-ngp#python-bindings)，或者如果您的GPU未在上述列表中（例如Hopper、Volta或Maxwell架构），您需要自行[构建__instant-ngp__](https://github.com/NVlabs/instant-ngp#building-instant-ngp-windows--linux)。

## 使用

<img src="docs/assets_readme/testbed.png" width="100%"/>

__instant-ngp__带有一个交互式GUI，具有许多功能：
- [全面的控制](https://github.com/NVlabs/instant-ngp#keyboard-shortcuts-and-recommended-controls)，可交互地探索神经图形基元，
- [VR模式](https://github.com/NVlabs/instant-ngp#vr-controls)，通过虚拟现实头盔查看神经图形基元，
- 保存和加载“快照”，以便您可以在互联网上分享您的图形基元，
- 摄像机路径编辑器以创建视频，
- `NeRF->Mesh`和`SDF->Mesh`转换，
- 摄像机姿势和镜头优化，
- 还有许多其他功能。

### NeRF狐狸

只需启动`instant-ngp`并将`data/nerf/fox`文件夹拖放到窗口中。或者，您也可以使用命令行：

```sh
instant-ngp$ ./instant-ngp data/nerf/fox
```

<img src="docs/assets_readme/fox.png"/>

您可以使用__任何__与NeRF兼容的数据集，例如[原始NeRF数据集](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi)，[SILVR数据集](https://github.com/IDLabMedia/large-lightfields-dataset)或[DroneDeploy数据集](https://github.com/nickponline/dd-nerf-dataset)。**要创建自己的NeRF模型，请观看[视频教程](https://www.youtube.com/watch?v=3TWxO1PftMc)或阅读[书面说明](docs/nerf_dataset_tips.md)。**

### SDF豪猪

将`data/sdf/armadillo.obj`拖放到窗口中，或使用以下命令：

```sh
instant-ngp$ ./instant-ngp data/sdf/armadillo.obj
```

<img src="docs/assets_readme/armadillo.png"/>

### Einstein的图像

将`data/image/albert.exr`拖放到窗口中，或使用以下命令：

```sh
instant-ngp$ ./instant-ngp data/image/albert.exr
```

<img src="docs/assets_readme/albert.png"/>

要重现十亿像素的结果，请下载，例如，[东京图像](https://www.flickr.com/photos/trevor_dobson_inefekt69/29314390837)，并使用`scripts/convert_image.py`脚本将其转换为`.bin`格式。这种自定义格式在分辨率较高时提高了兼容性和加载速度。现在，您可以运行以下命令：

```sh
instant-ngp$ ./instant-ngp data/image/tokyo.bin
```

### 体积渲染器

下载[迪士尼云朵的nanovdb体积](https://drive.google.com/drive/folders/1SuycSAOSG64k2KLV7oWgyNWyCvZAkafK?usp=sharing)，该体积派生自[这里](https://disneyanimation.com/data-sets/?drawer=/resources/clouds/)（[CC BY-SA 3.0](https://media.disneyanimation.com/uploads/production/data_set_asset/6/asset/License_Cloud.pdf)）。然后，将`wdas_cloud_quarter.nvdb`拖放到窗口中，或使用以下命令：

```sh
instant-ngp$ ./instant-ngp wdas_cloud_quarter.nvdb
```

<img src="docs/assets_readme/cloud.png"/>

### 键盘快捷键和推荐控制方式

以下是__instant-ngp__应用程序的主要键盘控制方式。

| 键              | 意思           |
| :-------------: | ------------- |
| WASD            | 前进/向左平移/后退/向右平移。 |
| Spacebar / C    | 上移/下移。 |
| =或+ / -或_     | 增加/减少相机速度（第一人称模式）或放大/缩小（第三人称模式）。 |
| E / Shift+E     | 增加/减少曝光。 |
| Tab             | 切换菜单可见性。 |
| T               | 切换训练。大约两分钟后，训练趋于稳定，可以切换为关闭状态。 |
| { }             | 转到第一张/最后一张训练图像的相机视图。 |
| [ ]             | 转到上一个/下一个训练图像的相机视图。 |
| R               | 重新加载文件中的网络。 |
| Shift+R         | 重置相机。 |
| O               | 切换可视化或累积误差图。 |
| G               | 切换地面真实图像的可视化。 |
| M               | 切换神经模型层的多视角可视化。有关更多说明，请参阅论文中的视频。 |
| , / .           | 显示上一个/下一个可视化的层；按M键退出。 |
| 1-8             | 在各种渲染模式之间切换，其中2是标准模式。您可以在控制界面中查看渲染模式名称列表。 |

在__instant-ngp__的GUI中有许多控件。
首先，请注意此GUI可以移动和调整大小，"Camera path" GUI也可以如此（但首先必须展开才能使用）。

__instant-ngp__中推荐的用户控制方式包括：

* __Snapshot：__ 使用"Save"保存训练好的NeRF模型，使用"Load"重新加载模型。
* __Rendering -> DLSS：__ 打开此选项并将"DLSS sharpening"设置为1.0，通常可以提高渲染质量。
* __Rendering -> Crop size：__ 裁剪周围环境以聚焦于模型。"Crop aabb"可让您移动感兴趣区域的中心并进行微调。在我们的NeRF训练和数据集提示中了解更多关于此功能的信息（https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md）。

"相机路径" GUI 允许您创建用于渲染视频的相机路径。
"从相机添加" 按钮会在当前视角插入关键帧。
然后，您可以渲染相机路径的视频 `.mp4` 或将关键帧导出为 `.json` 文件。
关于 GUI 的更多信息，请参考[此文章](https://developer.nvidia.com/blog/getting-started-with-nvidia-instant-nerfs/)和[此视频指南](https://www.youtube.com/watch?v=3TWxO1PftMc)。

### VR 控制

要在 VR 中查看神经图形原语，请首先启动 VR 运行时。这可能是
- 如果您使用 Oculus Rift 或 Meta Quest（使用连接线）头戴设备，则可能是 __OculusVR__。
- 如果您使用其他头戴设备，则可能是 __SteamVR__。
- 任何兼容 OpenXR 的运行时都可以使用。

然后，在 __instant-ngp__ GUI 中点击 __连接到 VR/AR 头戴设备__ 按钮，并戴上头戴设备。
在进入 VR 之前，我们强烈建议您先完成训练（点击 "停止训练"）或加载预训练快照以获得最佳性能。

在 VR 中，您可以使用以下控制方式：

| 控制方式               | 含义       |
| :--------------------: | ------------- |
| 左摇杆 / 触摸板  | 移动 |
| 右摇杆 / 触摸板 | 旋转相机 |
| 按下摇杆 / 触摸板 | 擦除手部周围的 NeRF |
| 抓取（单手）      | 拖动神经图形原语 |
| 抓取（双手）      | 旋转和缩放（类似于手机上的捏合缩放） |

## 构建 instant-ngp（Windows和Linux）

### 要求

- 一块__NVIDIA GPU__；如果有张量核心，则性能会更好。所有显示的结果都来自于RTX 3090。
- 一个支持__C++14__的编译器。推荐以下选择并已经过测试：
  - __Windows：__ Visual Studio 2019或2022
  - __Linux：__ GCC/G++ 8或更高版本
- 最新版本的__[CUDA](https://developer.nvidia.com/cuda-toolkit)__。推荐以下选择并已经过测试：
  - __Windows：__ CUDA 11.5或更高版本
  - __Linux：__ CUDA 10.2或更高版本
- __[CMake](https://cmake.org/) v3.21或更高版本__。
- __(可选) [Python](https://www.python.org/) 3.7或更高版本__ 以进行交互式绑定。还需运行 `pip install -r requirements.txt`。
- __(可选) [OptiX](https://developer.nvidia.com/optix) 7.6或更高版本__ 以加快网格SDF训练速度。
- __(可选) [Vulkan SDK](https://vulkan.lunarg.com/)__ 以支持DLSS。

如果您使用基于Debian的Linux发行版，请安装以下软件包：
```sh
sudo apt-get install build-essential git python3-dev python3-pip libopenexr-dev libxi-dev \
                     libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev
```

或者，如果您使用的是Arch或Arch衍生发行版，请安装以下软件包：
```sh
sudo pacman -S cuda base-devel cmake openexr libxi glfw openmp libxinerama libxcursor
```

我们还建议将[CUDA](https://developer.nvidia.com/cuda-toolkit)和[OptiX](https://developer.nvidia.com/optix)安装在`/usr/local/`目录下，并将CUDA安装路径添加到PATH环境变量中。

例如，如果您有CUDA 11.4，请在`~/.bashrc`文件中添加以下内容：
```sh
export PATH="/usr/local/cuda-11.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH"
```

### 编译

首先使用以下命令克隆该存储库及其所有子模块：
```sh
$ git clone --recursive https://github.com/nvlabs/instant-ngp
$ cd instant-ngp
```

然后，使用CMake构建项目（在Windows上，必须使用[开发人员命令提示符](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-160#developer_command_prompt)）：
```sh
instant-ngp$ cmake . -B build
instant-ngp$ cmake --build build --config RelWithDebInfo -j
```

如果编译失败或耗时超过一个小时，请检查是否内存不足。在这种情况下，尝试不使用`-j`运行上述命令。
如果问题仍然存在，请在报告问题之前，请参考[此处的可能解决方法列表](https://github.com/NVlabs/instant-ngp#troubleshooting-compile-errors)。

如果编译成功，现在您可以通过`./instant-ngp`可执行文件或下面描述的`scripts/run.py`脚本来运行代码。


如果自动GPU架构检测失败（例如，如果您安装了多个GPU），请为您希望使用的GPU设置`TCNN_CUDA_ARCHITECTURES`环境变量。以下表格列出了常见GPU的值。如果您的GPU未列出，请参考[此详尽列表](https://developer.nvidia.com/cuda-gpus)。

| H100 | 40X0 | 30X0 | A100 | 20X0 | TITAN V / V100 | 10X0 / TITAN Xp | 9X0 | K80 |
|:----:|:----:|:----:|:----:|:----:|:--------------:|:---------------:|:---:|:---:|
|   90 |   89 |   86 |   80 |   75 |             70 |              61 |  52 |  37 |


## Python绑定

在构建好__instant-ngp__之后，您可以使用其Python绑定以自动化方式进行受控实验。所有与交互式GUI相同的功能（甚至更多）都具有可以轻松使用的Python绑定。有关如何从Python中实现和扩展`./instant-ngp`应用程序的示例，请参见`./scripts/run.py`，它支持`./instant-ngp`支持的命令行参数的超集。

如果您更愿意从哈希编码和快速神经网络构建新模型，请考虑使用[tiny-cuda-nn的PyTorch扩展](https://github.com/nvlabs/tiny-cuda-nn#pytorch-extension)。

祝您愉快！

## 附加资源

- [NVIDIA Instant NeRF快速入门博文](https://developer.nvidia.com/blog/getting-started-with-nvidia-instant-nerfs/)
- [用于高级NeRF数据集创建的SIGGRAPH教程](https://www.nvidia.com/en-us/on-demand/session/siggraph2022-sigg22-s-16/)。

## 常见问题（FAQ）

__问：__ 我的自定义数据集的NeRF重建效果不好，我该怎么办？

__答：__ 可能有多个问题：
- COLMAP可能无法重建相机姿态。
- 捕捉过程中可能存在运动或模糊。不要将捕捉视为一项艺术任务；将其视为摄影测量。您的数据集中应该有_尽可能少的模糊_（运动模糊、散焦或其他模糊），并且所有物体在整个捕捉过程中都必须_保持静止_。如果使用广角镜头（如iPhone广角镜头），它比窄镜头涵盖的空间更多，可以获得额外的优势。
- 数据集参数（特别是`aabb_scale`）可能被调整得不够理想。我们建议从`aabb_scale=128`开始，然后按二倍数增加或减少，直到获得最佳质量。
- 请仔细阅读[我们的NeRF训练和数据集提示](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md)。

##
__问：__ 我如何保存训练好的模型并在以后重新加载它？

__答：__ 有两个选项：
1. 使用GUI的"Snapshot"部分。
2. 使用Python绑定的`load_snapshot` / `save_snapshot`函数（请参阅`scripts/run.py`以获取示例用法）。

##
__问：__ 这个代码库可以同时使用多个GPU吗？

__答：__ 只有在VR渲染中才能使用多个GPU，每个眼睛使用一个GPU。其他情况下不能。要选择要运行的特定GPU，请使用[CUDA_VISIBLE_DEVICES](https://stackoverflow.com/questions/39649102/how-do-i-select-which-gpu-to-run-a-job-on)环境变量。要针对该特定GPU进行优化_编译_，请使用[TCNN_CUDA_ARCHITECTURES](https://github.com/NVlabs/instant-ngp#compilation-windows--linux)环境变量。

##
__问：__ 如何在无头模式下运行__instant-ngp__？

__答：__ 使用`./instant-ngp --no-gui`或`python scripts/run.py`。您还可以通过`cmake -DNGP_BUILD_WITH_GUI=off ...`编译时禁用GUI。

##
__问：__ 这个代码库能在[Google Colab](https://colab.research.google.com/)上运行吗？

__答：__ 是的。请参阅[此示例](./notebooks/instant_ngp.ipynb)，该示例受到用户[@myagues](https://github.com/NVlabs/instant-ngp/issues/6#issuecomment-1016397579)创建的笔记本的启发。注意：该代码库需要大量的GPU内存，可能无法适应您分配的GPU。在旧的GPU上运行速度也会较慢。

##
__问：__ 是否有[Docker容器](https://www.docker.com/)？

__答：__ 是的。我们提供了一个[Visual Studio Code开发容器](https://code.visualstudio.com/docs/remote/containers)，您也可以单独使用`.devcontainer/Dockerfile`。

如果您想在不使用VSCode的情况下运行容器：
```
docker-compose -f .devcontainer/docker-compose.yml build instant-ngp
xhost local:root
docker-compose -f .devcontainer/docker-compose.yml run instant-ngp /bin/bash
```
然后按照上面的构建命令正常运行。

##
__问：__ 如何编辑和训练底层的哈希编码或神经网络以用于新任务？

__答：__ 使用[__tiny-cuda-nn__的PyTorch扩展](https://github.com/nvlabs/tiny-cuda-nn#pytorch-extension)。

##
__问：__ 坐标系统的约定是什么？

__答：__ 请参阅用户@jc211的[这个有用的图示](https://github.com/NVlabs/instant-ngp/discussions/153?converting=1#discussioncomment-2187652)。

##
__问：__ 为什么在NeRF训练期间背景颜色是随机的？

__答：__ 训练数据中的透明度表示对学习模型的透明度的需求。如果使用纯色背景，模型可以通过简单地预测该背景颜色来最小化损失，而不是预测透明度（零密度）。通过随机化背景颜色，模型被_强制_学习零密度，以便让随机化的颜色“透过来”。

##
__问：__ 如何遮罩掉NeRF训练像素（例如用于动态对象去除）？

__答：__ 对于任何具有动态对象的训练图像`xyz.*`，您可以在同一文件夹中提供一个名为`dynamic_mask_xyz.png`的文件。该文件必须是PNG格式，其中_非零_像素值表示遮罩区域。

## 解决编译错误

在进一步调查之前，请确保所有子模块都是最新的，并尝试重新编译。
```sh
instant-ngp$ git submodule sync --recursive
instant-ngp$ git submodule update --init --recursive
```
如果__instant-ngp__仍然无法编译，请将CUDA和编译器更新到您的系统上可以安装的最新版本。重要的是要同时更新_两者_，因为较新的CUDA版本不一定与较早的编译器兼容，反之亦然。
如果问题仍然存在，请参考以下已知问题列表。

**\*在每个步骤之后，删除`build`文件夹并让CMake重新生成它，然后再尝试。\***

| 问题 | 解决方案 |
|---------|------------|
| __CMake错误：__ 找不到CUDA工具集/目标“cmTC_0c70f”的CUDA_ARCHITECTURES为空 | __Windows：__ Visual Studio CUDA集成未正确安装。按照[这些说明](https://github.com/mitsuba-renderer/mitsuba2/issues/103#issuecomment-618378963)修复问题，无需重新安装CUDA。([#18](https://github.com/NVlabs/instant-ngp/issues/18)) |
| | __Linux：__ 您CUDA安装的环境变量可能设置不正确。您可以通过使用```cmake . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-<your cuda version>/bin/nvcc```绕过此问题。([#28](https://github.com/NVlabs/instant-ngp/issues/28)) |
| __CMake错误：__ CXX编译器"MSVC"没有已知的功能 | 重新安装Visual Studio，并确保从开发者shell中运行CMake。在重新构建之前，请确保删除构建文件夹。([#21](https://github.com/NVlabs/instant-ngp/issues/21)) |
| __编译错误：__ 非链接阶段需要一个输入文件，但指定了输出文件 | 确保__instant-ngp__的路径中没有空格。一些构建系统似乎对其中存在的空格有问题。([#39](https://github.com/NVlabs/instant-ngp/issues/39) [#198](https://github.com/NVlabs/instant-ngp/issues/198)) |
| __编译错误：__ 对“cudaGraphExecUpdate”的未定义引用/标识符“cublasSetWorkspace”未定义 | 将CUDA安装（可能是11.0）更新到11.3或更高版本。([#34](https://github.com/NVlabs/instant-ngp/issues/34) [#41](https://github.com/NVlabs/instant-ngp/issues/41) [#42](https://github.com/NVlabs/instant-ngp/issues/42)) |
| __编译错误：__ 函数调用中参数太少 | 使用上述两个`git`命令更新子模块。([#37](https://github.com/NVlabs/instant-ngp/issues/37) [#52](https://github.com/NVlabs/instant-ngp/issues/52)) |
| __Python错误：__ 没有名为'pyngp'的模块 | 可能是因为CMake没有检测到您的Python安装，因此没有构建`pyngp`。检查CMake日志以验证此问题。如果`pyngp`在与`build`不同的文件夹中构建，Python将无法检测到它，您必须提供完整的导入语句的路径。([#43](https://github.com/NVlabs/instant-ngp/issues/43)) |

如果在表中找不到您的问题，请尝试在[讨论区](https://github.com/NVlabs/instant-ngp/discussions)和[问题区](https://github.com/NVlabs/instant-ngp/issues?q=is%3Aissue)中搜索以获取帮助。如果您仍然遇到困难，请[提交问题](https://github.com/NVlabs/instant-ngp/issues/new)寻求帮助。

## 鸣谢

非常感谢[Jonathan Tremblay](https://research.nvidia.com/person/jonathan-tremblay)和[Andrew Tao](https://developer.nvidia.com/blog/author/atao/)在测试此代码库的早期版本时的贡献，以及Arman Toorians和Saurabh Jain提供的工厂机器人数据集。
我们还感谢[Andrew Webb](https://github.com/grey-area)注意到空间哈希中的一个质数实际上不是质数；这个问题已经得到修复。

本项目使用了许多令人印象深刻的开源库，包括：
* [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) 用于快速的CUDA网络和输入编码
* [tinyexr](https://github.com/syoyo/tinyexr) 支持EXR格式
* [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) 支持OBJ格式
* [stb_image](https://github.com/nothings/stb) 支持PNG和JPEG
* [Dear ImGui](https://github.com/ocornut/imgui) 优秀的即时模式GUI库
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) 用于线性代数的C++模板库
* [pybind11](https://github.com/pybind/pybind11) 用于无缝的C++ / Python互操作性
* 其他！请查看`dependencies`文件夹。

感谢这些出色项目的作者们！

## 许可和引用

```bibtex
@article{mueller2022instant,
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {102:1--102:15},
    articleno = {102},
    numpages = {15},
    url = {https://doi.org/10.1145/3528223.3530127},
    doi = {10.1145/3528223.3530127},
    publisher = {ACM},
    address = {New York, NY, USA},
}
```

版权所有 © 2022，NVIDIA Corporation。保留所有权利。

本作品根据Nvidia源代码许可-NC授权。点击[这里](LICENSE.txt)查看此许可的副本。