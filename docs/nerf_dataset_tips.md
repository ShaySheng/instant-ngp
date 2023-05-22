# 第一部分：训练NeRF模型的技巧与即时神经图形原语

我们的NeRF实现期望以与[原始NeRF代码库](https://www.matthewtancik.com/nerf)兼容的格式，通过`transforms.json`文件提供初始相机参数。
我们提供了一个方便的脚本，[scripts/colmap2nerf.py](/scripts/colmap2nerf.py)，可以用于处理视频文件或图像序列，使用开源的[COLMAP](https://colmap.github.io/)结构运动软件提取所需的相机数据。

您还可以使用[scripts/record3d2nerf.py](/scripts/record3d2nerf.py)脚本从Record3D（基于ARKit）生成相机数据。

训练过程对数据集要求相当挑剔。
例如，数据集的覆盖范围要广，不应包含错误标记的相机数据，并且不应包含模糊的帧（运动模糊和散焦模糊都会造成问题）。
本文试图给出一些建议。
一个好的经验法则是，如果您的NeRF模型在大约20秒后似乎没有收敛，那么在长时间训练后，它不太可能有很大改善。
因此，我们建议在训练的早期阶段调整数据，以获得清晰的结果。
对于大型的现实世界场景，通过最多几分钟的训练可以稍微提高图像的清晰度。
几乎所有的收敛都发生在最初的几秒钟内。

数据集最常见的问题是相机位置的尺度或偏移不正确；下面提供更多详细信息。
其次最常见的问题是图像数量过少，或者图像的相机参数不准确（例如，如果COLMAP失败）。
在这种情况下，您可能需要获取更多图像，或者调整计算相机位置的过程。
这超出了__instant-ngp__实现的范围。

## 现有数据集

__instant-ngp__的NeRF实现默认只在从`[0, 0, 0]`到`[1, 1, 1]`的单位边界框内沿光线行进。
数据加载器默认使用输入JSON文件中的相机变换，并通过`0.33`进行位置缩放，并通过`[0.5, 0.5, 0.5]`进行偏移，以将输入数据的原点映射到该立方体的中心。
选择缩放因子是为了适应原始NeRF论文中的合成数据集，以及我们的[scripts/colmap2nerf.py](/scripts/colmap2nerf.py)脚本的输出。

值得注意的是，通过在UI的“调试可视化”下拉菜单中

同时选择“可视化相机”和“可视化单位立方体”，可以检查相机与此边界框的对齐情况，如下所示：

<img src="assets/nerfbox.jpg" width="100%"/>

对于在单位立方体之外可见背景的自然场景，需要将`transforms.json`文件中的参数`aabb_scale`设置为2的整数幂，最大为128（即1、2、4、8、...、128），在最外层作用域中（与现有的`camera_angle_x`参数的嵌套层次相同）。请参见[data/nerf/fox/transforms.json](/data/nerf/fox/transforms.json)中的示例。

效果如下图所示：

<img src="assets/nerfboxrobot.jpg" width="100%"/>

相机仍然相对于单位立方体内的“感兴趣的物体”略微居中；然而，这里设置为16的aabb_scale参数使得NeRF实现将光线追踪到一个较大的边界框（边长为16），其中包含背景元素，以`[0.5, 0.5, 0.5]`为中心。

## 调整现有数据集的尺度

如果您有一个已存在的`transforms.json`格式的数据集，它应该在原点处居中，并且与原始的NeRF合成数据集具有类似的尺度。当您将其加载到NGP中时，如果发现它不收敛，首先要检查的是相机相对于单位立方体的位置，使用上述描述的调试功能。如果数据集主要不在单位立方体内，值得将其移动到那里。您可以通过调整变换本身来实现这一点，或者，您可以将全局参数添加到JSON的外部范围。

```json
{
	"aabb_scale": 32,
	"scale": 0.33,
	"offset": [0.5, 0.5, 0.5],
	...
}
```
有关实现细节和其他选项，请参阅[nerf_loader.cu](/src/nerf_loader.cu)。

## 适应不同的曝光/白平衡/光照

许多数据集在图像之间存在曝光、白平衡或光照不一致的情况。
这可能导致重建中的问题，通常是"floaters"。
虽然避免这个问题的最好方法是重新录制数据集，但这并不总是可行的。
在这种情况下，__instant-ngp__支持通过向`transforms.json`添加以下行来学习每个图像的"latent"外观代码：
```json
{
	"n_extra_learnable_dims": 16,
}
```
其中`16`是一个表现良好的默认值，但也可以尝试其他值。

## 准备新的NeRF数据集

要对自己捕捉的数据进行训练，必须将数据处理为__instant-ngp__支持的现有格式。我们提供了脚本来支持三种方法：
- [COLMAP](#COLMAP)：从您拍摄的一组照片或视频创建数据集
- [Record3D](#Record3D)：使用iPhone 12 Pro或更新设备创建数据集（基于ARKit）
- [NeRFCapture](#NeRFCapture)：使用iOS设备直接创建数据集或将姿势图像流传输到__instant-ngp__。

所有这些方法都需要在您的计算机上安装并在PATH中可用的[Python](https://www.python.org/) 3.7或更高版本。

在Windows上，您可以[从此处下载安装程序](https://www.python.org/downloads/)。在安装过程中，请确保选中"将python.exe添加到PATH"。

如果您使用的是基于Debian的Linux发行版，请使用以下命令安装Python：
```sh
sudo apt-get install python3-dev python3-pip
```

或者，如果您使用的是Arch或基于Arch的发行版，请使用以下命令安装Python：
```sh
sudo pacman -S python python-pip
```

对于所有操作系统，在安装了Python之后，您需要通过在Windows命令提示符/ Linux终端中执行以下命令来安装所需的Python包：
```sh
pip install -r requirements.txt
```

### COLMAP

如果您使用Linux，请确保已安装[COLMAP](https://colmap.github.io/)并且在PATH中可用。如果您使用视频文件作为输入，请还确保安装了[FFmpeg](https://www.ffmpeg.org/)并且在PATH中可用。
要检查是否满足这些条件，请从终端窗口运行`colmap`和`ffmpeg

 -?`命令，并看到各自的帮助文本。

如果您使用的是Windows，您无需安装任何东西。运行以下脚本时，COLMAP和FFmpeg将自动下载。

如果您要从视频文件进行训练，请在包含视频的文件夹中运行[scripts/colmap2nerf.py](/scripts/colmap2nerf.py)脚本，并使用以下推荐参数：

```sh
data-folder$ python [path-to-instant-ngp]/scripts/colmap2nerf.py --video_in <视频文件名> --video_fps 2 --run_colmap --aabb_scale 32
```

以上假设输入为单个视频文件，然后以指定的帧率（2）提取帧。建议选择导致约50-150张图像的帧率。因此，对于一分钟的视频，`--video_fps 2`是理想的选择。

对于从图像进行训练，请将图像放在名为`images`的子文件夹中，然后使用适当的选项，例如以下选项：

```sh
data-folder$ python [path-to-instant-ngp]/scripts/colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 32
```

脚本将根据需要运行（并在Windows上安装）FFmpeg和COLMAP，然后进行转换步骤以生成所需的`transforms.json`格式，并将其写入当前目录。

默认情况下，脚本使用"sequential matcher"调用colmap，适用于从平滑变化的摄像机路径中拍摄的图像，例如视频。如果图像没有特定顺序，则应使用exhaustive matcher，如上面的图像示例所示。


要获取更多选项，您可以使用`--help`参数运行脚本。如果要使用COLMAP进行更高级的用法或处理复杂场景，请参阅[COLMAP文档](https://colmap.github.io/cli.html)；您可能需要修改[scripts/colmap2nerf.py](/scripts/colmap2nerf.py)脚本本身。

`aabb_scale`参数是最重要的__instant-ngp__特定参数。它指定场景的范围，默认为1；也就是说，场景被缩放，使得相机位置与原点的平均距离为1个单位。对于小型合成场景，例如原始NeRF数据集，1的默认`aabb_scale`是理想的选择，并且可以实现最快的训练速度。NeRF模型假设训练图像完全可以通过该边界框内的场景来解释。然而，对于自然场景，如果背景延伸到了该边界框之外，NeRF模型可能会遇到困难，并且可能在边界处产生"floaters"。通过将`aabb_scale`设置为更大的2的幂次方（最大为128），NeRF模型将扩展光线到一个更大的边界框。请注意，这可能会略微影响训练速度。如有疑问，对于自然场景，请从`aabb_scale`设置为128开始，然后根据情况逐渐减小。可以直接在`transforms.json`输出文件中编辑该值，而无需重新运行[scripts/colmap2nerf.py](/scripts/colmap2nerf.py)脚本。

您还可以选择传递对象类别（例如`--mask_categories person car`），该脚本将使用[Detectron2](https://github.com/facebookresearch/detectron2)自动生成掩码。
__instant-ngp__不会使用掩码像素进行训练。
对于希望忽略移动或敏感对象（如人、汽车或自行车）的用户，此实用程序非常有用。
有关类别列表，请参见[scripts/category2id.json](/scripts/category2id.json)。

假设一切顺利，您现在可以按如下方式训练NeRF模型，从__instant-ngp__文件夹开始：

```sh
instant-ngp$ ./instant-ngp [包含transforms.json的训练数据文件夹的路径]
```

### Record3D

如果您使用的是>=iPhone 12 Pro，则可以使用[Record3D](https://record3d.app/)收集数据并避免使用COLMAP。[Record3D](https://record3d.app/)是一款iOS应用程序，依赖于ARKit来估计每个图像的相机姿态。对于缺乏纹理或包含重复模式的场景，它比COLMAP更稳

健。要使用Record3D数据训练__instant-ngp__，请按照以下步骤进行操作：

1. 录制一个视频并以"Shareable/Internal format (.r3d)"格式导出。
2. 将导出的数据发送到计算机。
3. 将`.r3d`扩展名替换为`.zip`，然后解压缩文件以获取`path/to/data`目录。
4. 运行预处理脚本：
	```
	instant-ngp$ python scripts/record3d2nerf.py --scene path/to/data
	```
	如果您是以横向模式拍摄的场景，请添加`--rotate`选项。

5. 启动__instant-ngp__训练：
	```
	instant-ngp$ ./instant-ngp path/to/data
	```
	
### NeRFCapture

[NeRFCapture](https://github.com/jc211/NeRFCapture)是一款iOS应用程序，可在任何ARKit设备上运行，可让您直接从手机流式传输图像到__instant-ngp__，从而实现更互动的体验。它还可以收集离线数据集以供以后使用。

运行NeRFCapture脚本需要以下依赖项：
```
pip install cyclonedds
```

要进行流式传输：
1. 打开NeRFCapture应用程序。
2. 运行带有`--stream`标志的脚本。
	```
	instant-ngp$ python scripts/nerfcapture2nerf.py --stream
	```
3. 等待应用程序与脚本之间建立连接。应用程序上会显示连接状态。
4. 在应用程序上点击发送按钮。捕获的帧将被发送到__instant-ngp__。
5. 切换训练。

保存数据集：
1. 打开NeRFCapture应用程序。
2. 运行脚本并使用`--save_path`标志。`n_frames`参数指定在保存数据集之前要捕获的帧数。
	```
	instant-ngp$ python scripts/nerfcapture2nerf.py --save_path "dir1/dir2" --n_frames 20
	```
3. 等待应用程序与脚本之间建立连接。应用程序上会显示连接状态。
4. 在应用程序上点击发送按钮。捕获的帧将保存到运行脚本的计算机上的数据集文件夹中。

## NeRF训练数据的技巧

NeRF模型在训练时最适合使用50至150张图像，这些图像在场景移动、运动模糊或其他模糊伪影方面都呈现出最小的问题。重建质量取决于先前的脚本能够从图像中提取准确的相机参数。请参阅前面的部分以了解如何验证这一点。

`colmap2nerf.py`和`record3d2nerf.py`脚本假设训练图像都大致指向一个共享的兴趣点，并将其放置在原点。此点通过对所有训练图像的中心像素通过的射线的最近接近点进行加权平均得到。实际上，这意味着当训练图像被捕捉时，最好是指向感兴趣的对象的内部，尽管它们不需要完全围绕对象拍摄完整的360度视角。如果将`aabb_scale`设置为大于1的数值（如上文所述），则仍然会重建出对象后面可见的任何背景。