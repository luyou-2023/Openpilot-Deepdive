import os
import io
import json
import torch
from math import pi
import numpy as np
from scipy.interpolate import interp1d
import cv2

cv2.setNumThreads(0)  # 禁用OpenCV多线程
cv2.ocl.setUseOpenCL(False)  # 禁用OpenCL

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import warp, generate_random_params_for_warp
from view_transform import calibration

import utils_comma2k19.orientation as orient
import utils_comma2k19.coordinates as coord


class PlanningDataset(Dataset):
    def __init__(self, root='data', json_path_pattern='p3_%s.json', split='train'):
        # 加载数据集并打印样本数量
        self.samples = json.load(open(os.path.join(root, json_path_pattern % split)))
        print('PlanningDataset: %d samples loaded from %s' %
              (len(self.samples), os.path.join(root, json_path_pattern % split)))
        self.split = split

        # 图像数据所在的根目录
        self.img_root = os.path.join(root, 'nuscenes')

        # 图像预处理的转换流程
        self.transforms = transforms.Compose(
            [
                transforms.Resize((128, 256)),  # 图像调整为128x256
                transforms.ToTensor(),  # 转换为Tensor
                transforms.Normalize([0.3890, 0.3937, 0.3851],  # 图像标准化
                                     [0.2172, 0.2141, 0.2209]),
            ]
        )

        # 是否启用数据增强（例如随机失真和翻转）
        self.enable_aug = False
        # 是否进行视角变换
        self.view_transform = False

        # 是否使用内存缓存
        self.use_memcache = False
        if self.use_memcache:
            self._init_mc_()  # 如果使用缓存，初始化Memcache

    def _init_mc_(self):
        # 初始化Memcache客户端
        from petrel_client.client import Client
        self.client = Client('~/petreloss.conf')
        print('======== Initializing Memcache: Success =======')

    def _get_cv2_image(self, path):
        # 读取图像，如果使用Memcache则从缓存中加载
        if self.use_memcache:
            img_bytes = self.client.get(str(path))
            assert (img_bytes is not None)
            img_mem_view = memoryview(img_bytes)
            img_array = np.frombuffer(img_mem_view, np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            return cv2.imread(path)

    def __len__(self):
        # 返回数据集中的样本数量
        return len(self.samples)

    def __getitem__(self, idx):
        # 获取样本
        sample = self.samples[idx]
        imgs, future_poses = sample['imgs'], sample['future_poses']

        # 处理未来位置（future_poses）
        future_poses = torch.tensor(future_poses)
        future_poses[:, 0] = future_poses[:, 0].clamp(1e-2, )  # 保证车不会倒退

        # 加载图像并转换为RGB
        imgs = list(self._get_cv2_image(os.path.join(self.img_root, p)) for p in imgs)
        imgs = list(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs)

        # 训练时启用数据增强（例如随机失真、翻转）
        if self.enable_aug and self.split == 'train':
            # 随机失真
            w_offsets, h_offsets = generate_random_params_for_warp(imgs[0], random_rate=0.1)
            imgs = list(warp(img, w_offsets, h_offsets) for img in imgs)

            # 随机翻转
            if np.random.rand() > 0.5:
                imgs = list(img[:, ::-1, :] for img in imgs)
                future_poses[:, 1] *= -1  # 翻转未来位置

        # 进行视角变换（如果设置了视角变换）
        if self.view_transform:
            camera_rotation_matrix = np.linalg.inv(np.array(sample["camera_rotation_matrix_inv"]))
            camera_translation = -np.array(sample["camera_translation_inv"])
            camera_extrinsic = np.vstack(
                (np.hstack((camera_rotation_matrix, camera_translation.reshape((3, 1)))), np.array([0, 0, 0, 1])))
            camera_extrinsic = np.linalg.inv(camera_extrinsic)
            warp_matrix = calibration(camera_extrinsic, np.array(sample["camera_intrinsic"]))
            imgs = list(
                cv2.warpPerspective(src=img, M=warp_matrix, dsize=(256, 128), flags=cv2.WARP_INVERSE_MAP) for img in
                imgs)

        # 转换为PIL图像
        imgs = list(Image.fromarray(img) for img in imgs)
        imgs = list(self.transforms(img) for img in imgs)  # 图像标准化和转换为Tensor
        input_img = torch.cat(imgs, dim=0)  # 将所有图像连接成一个张量

        return dict(
            input_img=input_img,
            future_poses=future_poses,
            camera_intrinsic=torch.tensor(sample['camera_intrinsic']),
            camera_extrinsic=torch.tensor(sample['camera_extrinsic']),
            camera_translation_inv=torch.tensor(sample['camera_translation_inv']),
            camera_rotation_matrix_inv=torch.tensor(sample['camera_rotation_matrix_inv']),
        )


class SequencePlanningDataset(PlanningDataset):
    def __init__(self, root='data', json_path_pattern='p3_%s.json', split='train'):
        print('Sequence', end='')
        self.fix_seq_length = 18  # 固定序列长度
        super().__init__(root=root, json_path_pattern=json_path_pattern, split=split)

    def __getitem__(self, idx):
        seq_samples = self.samples[idx]
        seq_length = len(seq_samples)

        # 如果序列长度小于固定长度，随机选择其他样本
        if seq_length < self.fix_seq_length:
            return self.__getitem__(np.random.randint(0, len(self.samples)))

        # 如果序列长度超过固定长度，则随机裁剪
        if seq_length > self.fix_seq_length:
            seq_length_delta = seq_length - self.fix_seq_length
            seq_length_delta = np.random.randint(0, seq_length_delta + 1)
            seq_samples = seq_samples[seq_length_delta:self.fix_seq_length + seq_length_delta]

        seq_future_poses = list(smp['future_poses'] for smp in seq_samples)
        seq_imgs = list(smp['imgs'] for smp in seq_samples)

        seq_input_img = []
        for imgs in seq_imgs:
            imgs = list(self._get_cv2_image(os.path.join(self.img_root, p)) for p in imgs)
            imgs = list(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs)  # RGB
            imgs = list(Image.fromarray(img) for img in imgs)
            imgs = list(self.transforms(img) for img in imgs)
            input_img = torch.cat(imgs, dim=0)
            seq_input_img.append(input_img[None])
        seq_input_img = torch.cat(seq_input_img)

        return dict(
            seq_input_img=seq_input_img,  # 序列输入图像
            seq_future_poses=torch.tensor(seq_future_poses),  # 序列的未来位置
            camera_intrinsic=torch.tensor(seq_samples[0]['camera_intrinsic']),
            camera_extrinsic=torch.tensor(seq_samples[0]['camera_extrinsic']),
            camera_translation_inv=torch.tensor(seq_samples[0]['camera_translation_inv']),
            camera_rotation_matrix_inv=torch.tensor(seq_samples[0]['camera_rotation_matrix_inv']),
        )


class Comma2k19SequenceDataset(PlanningDataset):
    def __init__(self, split_txt_path, prefix, mode, use_memcache=True, return_origin=False):
        # 初始化数据集路径、前缀、模式等信息
        self.split_txt_path = split_txt_path  # 用于指定样本的列表文件路径
        self.prefix = prefix  # 用于指定样本的前缀路径

        # 读取样本列表，并去除每行的换行符
        self.samples = open(split_txt_path).readlines()
        self.samples = [i.strip() for i in self.samples]

        # 确保模式是 'train', 'val' 或 'demo'
        assert mode in ('train', 'val', 'demo')
        self.mode = mode
        if self.mode == 'demo':
            print('Comma2k19SequenceDataset: DEMO mode is on.')

        # 根据模式设置固定的序列长度
        self.fix_seq_length = 800 if mode == 'train' else 800

        # 定义数据转换流程：调整尺寸、转为Tensor、标准化处理
        self.transforms = transforms.Compose(
            [
                # transforms.Resize((900 // 2, 1600 // 2)),  # 被注释掉的大小调整
                # transforms.Resize((9 * 32, 16 * 32)),  # 被注释掉的大小调整
                transforms.Resize((128, 256)),  # 调整大小到128x256
                transforms.ToTensor(),  # 转为Tensor
                transforms.Normalize([0.3890, 0.3937, 0.3851],  # 标准化数据
                                     [0.2172, 0.2141, 0.2209]),
            ]
        )

        # 定义用于图像校准的外部矩阵和相机内外参数
        self.warp_matrix = calibration(extrinsic_matrix=np.array([[0, -1, 0, 0],
                                                                  [0, 0, -1, 1.22],
                                                                  [1, 0, 0, 0],
                                                                  [0, 0, 0, 1]]),
                                       cam_intrinsics=np.array([[910, 0, 582],
                                                                [0, 910, 437],
                                                                [0, 0, 1]]),
                                       device_frame_from_road_frame=np.hstack(
                                           (np.diag([1, -1, -1]), [[0], [0], [1.22]])))

        self.use_memcache = use_memcache  # 是否使用内存缓存
        if self.use_memcache:
            self._init_mc_()  # 如果启用缓存，初始化缓存客户端

        self.return_origin = return_origin  # 是否返回原始图像

        # 定义时间锚点和帧数
        self.num_pts = 10 * 20  # 10秒 * 20Hz = 200帧
        self.t_anchors = np.array(
            (0., 0.00976562, 0.0390625, 0.08789062, 0.15625,
             0.24414062, 0.3515625, 0.47851562, 0.625, 0.79101562,
             0.9765625, 1.18164062, 1.40625, 1.65039062, 1.9140625,
             2.19726562, 2.5, 2.82226562, 3.1640625, 3.52539062,
             3.90625, 4.30664062, 4.7265625, 5.16601562, 5.625,
             6.10351562, 6.6015625, 7.11914062, 7.65625, 8.21289062,
             8.7890625, 9.38476562, 10.)
        )
        self.t_idx = np.linspace(0, 10, num=self.num_pts)  # 时间索引

    def _get_cv2_vid(self, path):
        # 使用OpenCV打开视频文件，如果使用缓存，生成预签名URL
        if self.use_memcache:
            path = self.client.generate_presigned_url(str(path), client_method='get_object', expires_in=3600)
        return cv2.VideoCapture(path)

    def _get_numpy(self, path):
        # 如果使用缓存，获取数据并返回NumPy数组
        if self.use_memcache:
            bytes = io.BytesIO(memoryview(self.client.get(str(path))))
            return np.lib.format.read_array(bytes)
        else:
            return np.load(path)

    def __getitem__(self, idx):
        # 获取当前序列的路径并打开视频文件
        seq_sample_path = self.prefix + self.samples[idx]
        cap = self._get_cv2_vid(seq_sample_path + '/video.hevc')
        if (cap.isOpened() == False):
            raise RuntimeError  # 如果视频打开失败，抛出错误
        imgs = []  # 用于存储所有的图像帧
        origin_imgs = []  # 用于存储原始图像帧
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                imgs.append(frame)  # 将读取到的帧添加到图片列表
                if self.return_origin:
                    origin_imgs.append(frame)  # 如果需要返回原始图像，保存原始图像
            else:
                break
        cap.release()  # 释放视频资源

        # 计算序列的长度
        seq_length = len(imgs)

        # 在DEMO模式下，调整固定序列长度
        if self.mode == 'demo':
            self.fix_seq_length = seq_length - self.num_pts - 1

        # 如果序列长度小于所需的长度，则递归调用获取下一个样本
        if seq_length < self.fix_seq_length + self.num_pts:
            print('The length of sequence', seq_sample_path, 'is too short',
                  '(%d < %d)' % (seq_length, self.fix_seq_length + self.num_pts))
            return self.__getitem__(idx + 1)

        # 随机选择一个开始索引，保证生成的序列长度
        seq_length_delta = seq_length - (self.fix_seq_length + self.num_pts)
        seq_length_delta = np.random.randint(1, seq_length_delta + 1)

        # 计算序列的开始和结束位置
        seq_start_idx = seq_length_delta
        seq_end_idx = seq_length_delta + self.fix_seq_length

        # 对图像序列进行裁剪和透视变换
        imgs = imgs[seq_start_idx - 1: seq_end_idx]  # 获取指定范围的帧
        imgs = [cv2.warpPerspective(src=img, M=self.warp_matrix, dsize=(512, 256), flags=cv2.WARP_INVERSE_MAP) for img
                in imgs]
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]  # 转为RGB格式
        imgs = list(Image.fromarray(img) for img in imgs)  # 转为PIL图像
        imgs = list(self.transforms(img)[None] for img in imgs)  # 应用预定义的转换
        input_img = torch.cat(imgs, dim=0)  # [N+1, 3, H, W]
        del imgs  # 删除中间变量节省内存
        input_img = torch.cat((input_img[:-1, ...], input_img[1:, ...]), dim=1)  # 拼接图像

        # 获取位姿数据
        frame_positions = self._get_numpy(self.prefix + self.samples[idx] + '/global_pose/frame_positions')[
                          seq_start_idx: seq_end_idx + self.num_pts]
        frame_orientations = self._get_numpy(self.prefix + self.samples[idx] + '/global_pose/frame_orientations')[
                             seq_start_idx: seq_end_idx + self.num_pts]

        future_poses = []
        for i in range(self.fix_seq_length):
            # 计算未来的位姿
            ecef_from_local = orient.rot_from_quat(frame_orientations[i])
            local_from_ecef = ecef_from_local.T
            frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef,
                                              frame_positions - frame_positions[i]).astype(np.float32)

            # 对每个坐标轴进行插值
            fs = [interp1d(self.t_idx, frame_positions_local[i: i + self.num_pts, j]) for j in range(3)]
            interp_positions = [fs[j](self.t_anchors)[:, None] for j in range(3)]
            interp_positions = np.concatenate(interp_positions, axis=1)

            future_poses.append(interp_positions)  # 将插值后的位姿添加到列表中
        future_poses = torch.tensor(np.array(future_poses), dtype=torch.float32)

        rtn_dict = dict(
            seq_input_img=input_img,  # torch.Size([N, 6, 128, 256]) # 包含变换后的图像
            seq_future_poses=future_poses,  # torch.Size([N, num_pts, 3]) # 包含未来的位姿
            # camera_intrinsic=torch.tensor(seq_samples[0]['camera_intrinsic']), # 相机内参（可选）
            # camera_extrinsic=torch.tensor(seq_samples[0]['camera_extrinsic']), # 相机外参（可选）
            # camera_translation_inv=torch.tensor(seq_samples[0]['camera_translation_inv']), # 相机逆位移（可选）
            # camera_rotation_matrix_inv=torch.tensor(seq_samples[0]['camera_rotation_matrix_inv']), # 相机逆旋转矩阵（可选）
        )

        # For DEMO 模式的处理
        if self.return_origin:
            # 获取原始图像并处理
            origin_imgs = origin_imgs[seq_start_idx: seq_end_idx]  # 选择需要的帧范围
            # 转换原始图像为RGB格式，并转换为Tensor
            origin_imgs = [torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[None] for img in origin_imgs]
            # 将所有原始图像连接在一起，形成一个Tensor
            origin_imgs = torch.cat(origin_imgs, dim=0)  # N, H_ori, W_ori, 3，原始图像的尺寸
            # 将原始图像加入返回字典
            rtn_dict['origin_imgs'] = origin_imgs

        # 返回最终结果字典
        return rtn_dict
