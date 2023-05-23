#!/usr/bin/env python3

# 导入所需的模块
import argparse # 用于解析命令行参数的模块
import os
import commentjson as json

import numpy as np

import shutil # 用于文件和文件夹操作的模块
import time

from common import *
from scenes import *

from tqdm import tqdm # 用于创建进度条的模块

import pyngp as ngp # noqa

def parse_args():
    # 创建解析命令行参数的解析器
    parser = argparse.ArgumentParser(description="Run instant neural graphics primitives with additional configuration & output options")

    # 定义命令行参数，如文件、场景、网络配置、快照等
    parser.add_argument("files", nargs="*", help="Files to be loaded. Can be a scene, network config, snapshot, camera path, or a combination of those.")
    # "files" 是一个可变位置参数，接受多个文件名作为输入。这些文件可以是场景、网络配置、快照或相机路径等。
    
    parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data. Can be NeRF dataset, a *.obj/*.stl mesh for training a SDF, an image, or a *.nvdb volume.")
    # "--scene" 或 "--training_data" 是一个可选参数，用于指定要加载的场景。它可以是场景的名称或训练数据的完整路径。
    # 场景可以是 NeRF 数据集、用于训练 SDF 的 *.obj/*.stl 网格、图像或 *.nvdb 体积数据。
    
    parser.add_argument("--mode", default="", type=str, help=argparse.SUPPRESS) # deprecated
    parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")
    # "--network" 是一个可选参数，用于指定网络配置的路径。如果未指定，则使用场景的默认配置。
    
    parser.add_argument("--load_snapshot", "--snapshot", default="", help="Load this snapshot before training. recommended extension: .ingp/.msgpack")
    # "--load_snapshot" 或 "--snapshot" 是一个可选参数，用于指定训练之前要加载的快照。推荐使用 .ingp/.msgpack 扩展名。
    
    parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .ingp/.msgpack")
    # "--save_snapshot" 是一个可选参数，用于指定训练之后要保存的快照。推荐使用 .ingp/.msgpack 扩展名。
    
    parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes, but helps with high PSNR on synthetic scenes.")
    # "--nerf_compatibility" 是一个可选的布尔参数，当指定时，会使参数与原始 NeRF 论文中的参数相匹配。
    # 这可能会导致在某些场景下速度较慢且结果较差，但对于合成场景的高 PSNR 值有帮助。
    
    parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
    # "--test_transforms" 是一个可选参数，用于指定 nerf 风格的变换 JSON 文件的路径，用于计算 PSNR。
    
    parser.add_argument("--near_distance", default=-1, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
    # "--near_distance" 是一个可选参数，用于设置训练射线从相机开始的距离。小于 0 表示使用 ngp 的默认值。
    
    parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")
    # "--exposure" 是一个可选参数，用于控制图像的亮度。正数增加亮度，负数减少亮度。
    
    parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
    # "--screenshot_transforms" 是一个可选参数，用于指定保存截图时使用的 nerf 风格变换的 JSON 文件的路径。

    parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
    # "--screenshot_frames" 是一个可选参数，用于指定要截图的帧的编号。

    parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
    # "--screenshot_dir" 是一个可选参数，用于指定保存截图的目录。

    parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")
    # "--screenshot_spp" 是一个可选参数，用于指定截图中每个像素的采样次数。

    parser.add_argument("--video_camera_path", default="", help="The camera path to render, e.g., base_cam.json.")
    # "--video_camera_path" 是一个可选参数，用于指定要渲染的相机路径，例如 base_cam.json。

    parser.add_argument("--video_camera_smoothing", action="store_true", help="Applies additional smoothing to the camera trajectory with the caveat that the endpoint of the camera path may not be reached.")
    # "--video_camera_smoothing" 是一个可选参数，当指定时，对相机轨迹应用额外的平滑处理，但要注意可能无法到达相机路径的终点。

    parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
    # "--video_fps" 是一个可选参数，用于指定每秒的帧数。

    parser.add_argument("--video_n_seconds", type=int, default=1, help="Number of seconds the rendered video should be long.")
    # "--video_n_seconds" 是一个可选参数，用于指定渲染视频的时长（秒数）。

    parser.add_argument("--video_render_range", type=int, nargs=2, default=(-1, -1), metavar=("START_FRAME", "END_FRAME"), help="Limit output to frames between START_FRAME and END_FRAME (inclusive)")
    # "--video_render_range" 是一个可选参数，用于限制输出的帧范围（包括起始帧和结束帧）。

    parser.add_argument("--video_spp", type=int, default=8, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")
    # "--video_spp" 是一个可选参数，用于指定每个像素的采样次数。较大的值意味着更少的噪点，但渲染速度较慢。

    parser.add_argument("--video_output", type=str, default="video.mp4", help="Filename of the output video (video.mp4) or video frames (video_%%04d.png).")
    # "--video_output" 是一个可选参数，用于指定输出视频的文件名（video.mp4）或视频帧（video_%%04d.png）。

    parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
    # "--save_mesh" 是一个可选参数，用于将基于 Marching Cubes 的网格输出从 NeRF 或 SDF 模型中。支持 OBJ 和 PLY 格式。

    parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")
    # "--marching_cubes_res" 是一个可选参数，用于设置 Marching Cubes 网格的分辨率。

    parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
    # "--width" 或 "--screenshot_w" 是一个可选参数，用于指定 GUI 和截图的分辨率宽度。

    parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")
    # "--height" 或 "--screenshot_h" 是一个可选参数，用于指定 GUI 和截图的分辨率高度。

    parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
    # "--gui" 是一个可选参数，当指定时，以交互方式运行测试界面。

    parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
    # "--train" 是一个可选参数，当启用 GUI 时，控制是否立即开始训练。

    parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
    # "--n_steps" 是一个可选参数，用于指定在退出之前进行的训练步数。

    parser.add_argument("--second_window", action="store_true", help="Open a second window containing a copy of the main output.")
    # "--second_window" 是一个可选参数，当指定时，打开包含主输出副本的第二个窗口。

    parser.add_argument("--vr", action="store_true", help="Render to a VR headset.")
    # "--vr" 是一个可选参数，当指定时，渲染到 VR 头显。

    parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images. Range 0.0 to 1.0.")
    # "--sharpen" 是一个可选参数，用于设置应用于 NeRF 训练图像的锐化程度。范围从 0.0 到 1.0。


    # 返回解析后的命令行参数结果
    return parser.parse_args()

def get_scene(scene):
    # 遍历场景列表
    for scenes in [scenes_sdf, scenes_nerf, scenes_image, scenes_volume]:
        # 检查给定的场景是否在当前场景列表中
        if scene in scenes:
            # 如果是，则返回对应的场景信息
            return scenes[scene]
    # 如果未找到对应的场景，返回None
    return None


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    if args.vr: # VR implies having the GUI running at the moment
        # 如果启用 VR 模式，则同时启用 GUI 模式
        args.gui = True

    if args.mode:
        # 提示警告：'--mode' 参数不再使用，它不起作用。模式会根据场景自动选择。
        print("Warning: the '--mode' argument is no longer in use. It has no effect. The mode is automatically chosen based on the scene.")

    # 创建测试实例
    testbed = ngp.Testbed()
    testbed.root_dir = ROOT_DIR

    for file in args.files:
        # 获取场景信息
        scene_info = get_scene(file)
        if scene_info:
            # 如果找到场景信息，则加载对应的文件
            file = os.path.join(scene_info["data_dir"], scene_info["dataset"])
        testbed.load_file(file)

    if args.scene:
        # 获取场景信息
        scene_info = get_scene(args.scene)
        if scene_info is not None:
            # 如果找到场景信息，则设置场景和网络路径
            args.scene = os.path.join(scene_info["data_dir"], scene_info["dataset"])
            if not args.network and "network" in scene_info:
                args.network = scene_info["network"]

        # 加载训练数据
        testbed.load_training_data(args.scene)

    if args.gui:
        # 根据参数选择合适的 GUI 分辨率
        sw = args.width or 1920
        sh = args.height or 1080
        while sw * sh > 1920 * 1080 * 4:
            sw = int(sw / 2)
            sh = int(sh / 2)
        # 初始化窗口
        testbed.init_window(sw, sh, second_window=args.second_window)
        if args.vr:
            # 如果启用 VR 模式，则初始化 VR
            testbed.init_vr()


    if args.load_snapshot:
        # 如果指定了快照文件路径，则获取快照文件所对应的场景信息
        scene_info = get_scene(args.load_snapshot)
        if scene_info is not None:
            # 如果找到了场景信息，则使用默认的快照文件名
            args.load_snapshot = default_snapshot_filename(scene_info)
        # 加载快照文件
        testbed.load_snapshot(args.load_snapshot)
    elif args.network:
        # 如果指定了网络配置文件路径，则重新加载网络配置
        testbed.reload_network_from_file(args.network)

    ref_transforms = {}
    if args.screenshot_transforms: # try to load the given file straight away
        # 如果指定了截屏变换文件路径，则尝试直接加载给定的文件
        print("Screenshot transforms from ", args.screenshot_transforms)
        with open(args.screenshot_transforms) as f:
            ref_transforms = json.load(f)

    if testbed.mode == ngp.TestbedMode.Sdf:
        # 如果测试模式是 SDF，则设置色调映射曲线为 ACES
        testbed.tonemap_curve = ngp.TonemapCurve.ACES

    testbed.nerf.sharpen = float(args.sharpen)
    # 设置 NeRF 训练图像的锐化程度

    testbed.exposure = args.exposure
    # 设置图像的曝光值

    testbed.shall_train = args.train if args.gui else True
    # 如果启用了 GUI 模式，则根据参数决定是否进行训练，否则始终进行训练


    testbed.nerf.render_with_lens_distortion = True
    # 启用镜头畸变的渲染模式

    network_stem = os.path.splitext(os.path.basename(args.network))[0] if args.network else "base"
    # 获取网络配置文件的文件名（不带扩展名）

    if testbed.mode == ngp.TestbedMode.Sdf:
        # 如果测试模式是 SDF，则根据参数设置彩色 SDF
        setup_colored_sdf(testbed, args.scene)

    if args.near_distance >= 0.0:
        # 如果指定了 NeRF 训练射线的近距离，则设置近距离参数
        print("NeRF training ray near_distance ", args.near_distance)
        testbed.nerf.training.near_distance = args.near_distance

    if args.nerf_compatibility:
        # 如果启用了 NeRF 兼容模式
        print(f"NeRF compatibility mode enabled")

        # 先前的NeRF论文在sRGB颜色空间中进行累积/混合计算。
        # 这不仅会影响背景的透明度，还会影响景深效果等。
        # 我们支持这种行为，但只在合成的NeRF数据的情况下启用，
        # 这样我们才能将PSNR数值与先前的工作结果进行比较。
        # 将颜色空间设置为 sRGB，用于与先前的 NeRF 论文中的结果进行比较
        testbed.color_space = ngp.ColorSpace.SRGB

        # 禁用指数锥追踪。稍微提高质量但降低速度。
        # 这在具有AABB 1（如合成场景）的场景中默认启用，
        # 但在较大的场景中不启用。因此在这里强制设置。
        testbed.nerf.cone_angle_constant = 0
        
        # 模拟 NeRF 论文中的行为，固定背景进行训练
        testbed.nerf.training.random_bg_color = False

    old_training_step = 0
    # 用于存储上一次训练步数的变量

    n_steps = args.n_steps
    # 从命令行参数中获取训练步数

    # 如果加载了快照、未指定训练步数，并且没有打开GUI，
    # 默认不进行训练，而是假设目标是渲染截屏、计算PSNR或生成视频。
    if n_steps < 0 and (not args.load_snapshot or args.gui):
        n_steps = 35000
        # 默认的训练步数为 35000

    tqdm_last_update = 0
    # 用于记录上次更新进度条的时间戳

    if n_steps > 0:
        # 如果设置了训练步数大于0
        with tqdm(desc="Training", total=n_steps, unit="steps") as t:
            # 创建进度条并设置总步数
            while testbed.frame():
                # 在每一帧进行训练

                if testbed.want_repl():
                    repl(testbed)
                    # 如果需要进入交互式命令行，则执行repl函数

                # 当训练步数达到指定步数时会发生什么？
                if testbed.training_step >= n_steps:
                    if args.gui:
                        testbed.shall_train = False
                        # 如果启用了GUI，则停止训练
                    else:
                        break
                        # 否则跳出训练循环

                # 更新进度条
                if testbed.training_step < old_training_step or old_training_step == 0:
                    old_training_step = 0
                    t.reset()
                    # 当训练步数回溯或为0时，重置进度条

                now = time.monotonic()
                if now - tqdm_last_update > 0.1:
                    t.update(testbed.training_step - old_training_step)
                    t.set_postfix(loss=testbed.loss)
                    old_training_step = testbed.training_step
                    tqdm_last_update = now
                    # 更新进度条并设置显示的训练损失和步数

    if args.save_snapshot:
        testbed.save_snapshot(args.save_snapshot, False)
        # 如果指定了保存快照的路径，则保存当前的快照

    if args.test_transforms:
        # 如果指定了测试转换文件路径
        print("Evaluating test transforms from ", args.test_transforms)
        # 打印正在评估的测试转换文件路径
        with open(args.test_transforms) as f:
            test_transforms = json.load(f)
            # 加载测试转换文件并存储为test_transforms变量
        data_dir=os.path.dirname(args.test_transforms)
        # 获取测试转换文件所在的目录路径
        totmse = 0
        totpsnr = 0
        totssim = 0
        totcount = 0
        minpsnr = 1000
        maxpsnr = 0
        # 用于存储计算的指标结果，包括总的MSE、PSNR、SSIM，以及最小和最大PSNR的初始值

        # 在黑色背景上评估指标
        testbed.background_color = [0.0, 0.0, 0.0, 1.0]

        # Prior nerf papers通常不进行多样本抗锯齿。
        # 因此将所有像素对齐到像素中心。
        testbed.snap_to_pixel_centers = True
        spp = 8
        # 设置每像素采样数

        testbed.nerf.render_min_transmittance = 1e-4
        # 设置最小传输率，用于渲染

        testbed.shall_train = False
        # 不进行训练
        testbed.load_training_data(args.test_transforms)
        # 加载测试转换数据

        with tqdm(range(testbed.nerf.training.dataset.n_images), unit="images", desc=f"Rendering test frame") as t:
            # 创建带有进度条的循环，用于渲染测试帧
            for i in t:
                resolution = testbed.nerf.training.dataset.metadata[i].resolution
                # 获取当前测试帧的分辨率
                testbed.render_ground_truth = True
                # 渲染带有真实图像的测试帧
                testbed.set_camera_to_training_view(i)
                # 将相机设置为训练视图
                ref_image = testbed.render(resolution[0], resolution[1], 1, True)
                # 渲染参考图像
                testbed.render_ground_truth = False
                # 不再渲染带有真实图像的测试帧
                image = testbed.render(resolution[0], resolution[1], spp, True)
                # 渲染当前测试帧的图像

                if i == 0:
                    write_image(f"ref.png", ref_image)
                    write_image(f"out.png", image)

                    diffimg = np.absolute(image - ref_image)
                    diffimg[...,3:4] = 1.0
                    write_image("diff.png", diffimg)
                    # 如果是第一帧，将参考图像、当前图像和它们的差异图像写入文件

                A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
                R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
                # 将图像数据转换为sRGB颜色空间，并进行裁剪，使其在0到1之间
                mse = float(compute_error("MSE", A, R))
                ssim = float(compute_error("SSIM", A, R))
                # 计算图像之间的MSE和SSIM
                totssim += ssim
                totmse += mse
                psnr = mse2psnr(mse)
                totpsnr += psnr
                minpsnr = psnr if psnr<minpsnr else minpsnr
                maxpsnr = psnr if psnr>maxpsnr else maxpsnr
                # 更新计算指标的累加值和最小/最大PSNR值
                totcount = totcount+1
                t.set_postfix(psnr = totpsnr/(totcount or 1))
                # 在进度条中显示当前的PSNR值

        psnr_avgmse = mse2psnr(totmse/(totcount or 1))
        psnr = totpsnr/(totcount or 1)
        ssim = totssim/(totcount or 1)
        # 计算平均MSE、PSNR和SSIM值
        print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")
        # 打印PSNR、最小/最大PSNR和SSIM的结果


    if args.save_mesh:
        # 如果指定了保存网格的路径
        res = args.marching_cubes_res or 256
        # 设置网格分辨率，默认为256
        print(f"Generating mesh via marching cubes and saving to {args.save_mesh}. Resolution=[{res},{res},{res}]")
        # 打印生成网格并保存的信息，包括保存路径和分辨率
        testbed.compute_and_save_marching_cubes_mesh(args.save_mesh, [res, res, res])
        # 通过Marching Cubes算法生成网格，并保存到指定路径

    if ref_transforms:
        # 如果存在参考转换
        testbed.fov_axis = 0
        # 设置视场角轴为0
        testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
        # 根据参考转换中的相机角度设置视场角
        if not args.screenshot_frames:
            args.screenshot_frames = range(len(ref_transforms["frames"]))
        # 如果没有指定截图帧数，则设置为参考转换中的所有帧
        print(args.screenshot_frames)
        # 打印截图帧数
        for idx in args.screenshot_frames:
            # 遍历截图帧数
            f = ref_transforms["frames"][int(idx)]
            # 获取对应索引的帧
            cam_matrix = f["transform_matrix"]
            # 获取帧的相机矩阵
            testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
            # 将相机矩阵设置为NeRF相机矩阵
            outname = os.path.join(args.screenshot_dir, os.path.basename(f["file_path"]))
            # 设置输出文件名，包括路径和基于文件路径的名称

            # Some NeRF datasets lack the .png suffix in the dataset metadata
            # 一些NeRF数据集在数据集元数据中缺少.png后缀
            if not os.path.splitext(outname)[1]:
                outname = outname + ".png"
                # 如果文件名没有后缀，则添加.png后缀

            print(f"rendering {outname}")
            # 打印正在渲染的图像路径
            image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
            # 渲染图像
            os.makedirs(os.path.dirname(outname), exist_ok=True)
            # 创建输出文件夹
            write_image(outname, image)
            # 将图像写入文件
    elif args.screenshot_dir:
        # 如果存在截图目录
        outname = os.path.join(args.screenshot_dir, args.scene + "_" + network_stem)
        # 设置输出文件名，包括截图目录、场景和网络名称
        print(f"Rendering {outname}.png")
        # 打印正在渲染的图像路径
        image = testbed.render(args.width or 1920, args.height or 1080, args.screenshot_spp, True)
        # 渲染图像
        if os.path.dirname(outname) != "":
            os.makedirs(os.path.dirname(outname), exist_ok=True)
        # 创建输出文件夹
        write_image(outname + ".png", image)
        # 将图像写入文件

    if args.video_camera_path:
        # 如果指定了视频相机路径
        testbed.load_camera_path(args.video_camera_path)
        # 加载视频相机路径

        resolution = [args.width or 1920, args.height or 1080]
        # 设置分辨率，默认为1920x1080
        n_frames = args.video_n_seconds * args.video_fps
        # 计算视频帧数，根据视频时长和帧率计算得出
        save_frames = "%" in args.video_output
        # 判断是否保存为帧图像
        start_frame, end_frame = args.video_render_range
        # 设置起始帧和结束帧的范围

        if "tmp" in os.listdir():
            shutil.rmtree("tmp")
        # 检查临时目录是否存在，如果存在则删除
        os.makedirs("tmp")
        # 创建临时目录

        for i in tqdm(list(range(min(n_frames, n_frames+1))), unit="frames", desc=f"Rendering video"):
            # 遍历帧数进行视频渲染
            testbed.camera_smoothing = args.video_camera_smoothing
            # 设置相机平滑参数

            if start_frame >= 0 and i < start_frame:
                # 如果起始帧大于等于0且当前帧小于起始帧
                # 为了使相机平滑和运动模糊起作用，我们不能从序列的中间开始渲染。
                # 相反，我们渲染一个非常小的图像并将其丢弃，对于这些初始帧来说。
                # TODO 一旦可用，将此替换为无操作的渲染方法
                frame = testbed.render(32, 32, 1, True, float(i)/n_frames, float(i + 1)/n_frames, args.video_fps, shutter_fraction=0.5)
                # 渲染一个非常小的图像，并将其丢弃
                continue
            elif end_frame >= 0 and i > end_frame:
                # 如果结束帧大于等于0且当前帧大于结束帧
                continue
                # 跳过当前帧

            frame = testbed.render(resolution[0], resolution[1], args.video_spp, True, float(i)/n_frames, float(i + 1)/n_frames, args.video_fps, shutter_fraction=0.5)
            # 渲染图像帧
            if save_frames:
                # 如果保存为帧图像
                write_image(args.video_output % i, np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)
                # 将图像帧写入文件
            else:
                write_image(f"tmp/{i:04d}.jpg", np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)
                # 将图像帧写入临时目录

        if not save_frames:
            os.system(f"ffmpeg -y -framerate {args.video_fps} -i tmp/%04d.jpg -c:v libx264 -pix_fmt yuv420p {args.video_output}")
            # 使用ffmpeg将临时目录中的图像帧合成为视频

        shutil.rmtree("tmp")
        # 删除临时目录
