from __future__ import division, print_function, absolute_import
import glob
import re
import warnings
import os.path as osp
from .bases import BaseImageDataset
import json


class MARS_Text(BaseImageDataset):
    dataset_dir = 'marslite'

    def __init__(self, root='', verbose=True, cfg=None, **kwargs):
        super(MARS_Text, self).__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.prompt = cfg.MODEL.TEXT_PROMPT * 'X ' if cfg.MODEL.TEXT_PROMPT > 0 else ''
        self.prefix = cfg.MODEL.PREFIX
        if self.prefix:
            print('~~~~~~~【We use modality prefix Here!】~~~~~~~')
        else:
            print('~~~~~~~【We do not use modality prefix Here!】~~~~~~~')
        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated.'
            )

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'querymm')
        self.gallery_dir = osp.join(self.data_dir, 'test')

        self.train_text_dir = osp.join(self.data_dir, 'text_update')
        self.query_text_dir = osp.join(self.data_dir, 'text_update')
        self.gallery_text_dir = osp.join(self.data_dir, 'text_update')

        self._check_before_run()

        train = self._process_dir(self.train_dir, self.train_text_dir, relabel=True)
        query = self._process_dir(self.query_dir, self.query_text_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, self.gallery_text_dir, relabel=False)
        if verbose:
            print("=> MARSLITE loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, text_dir_path, relabel=False):
        # 根据路径判断是处理训练集还是测试集
        prefix = 'train' if 'train' in dir_path else 'test'
        json_file_path = osp.join(text_dir_path, prefix + '_annotations.json')

        # 1. 一次性加载JSON文件
        try:
            with open(json_file_path, 'r') as f:
                annotations = json.load(f)
        except FileNotFoundError:
            print(f"错误：在路径 {json_file_path} 未找到标注文件")
            return []

        # 2. 创建从【纯图片文件名】到标注文本的映射字典
        #    注意：我们知道这里只有RGB (_m1) 的标注
        caption_map = {
            osp.basename(item['img_path']): item['captions'][0]
            for item in annotations if 'captions' in item and item['captions']
        }

        # --- 获取所有RGB图片的路径 ---
        img_paths_RGB = sorted(glob.glob(osp.join(dir_path, 'RGB', '*.jpg')))
        
        # --- 标准的PID处理流程 ---
        pid_container = set()
        pattern = re.compile(r'([a-z\d]+)_c(\d)')

        for img_path_RGB in img_paths_RGB:
            basename = osp.basename(img_path_RGB)
            match = pattern.search(basename)
            if match:
                pid, _ = match.groups()
                if pid != -1:
                    pid_container.add(pid)
        
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        # --- 处理并整合数据集 ---
        dataset = []
        for img_path_RGB in img_paths_RGB:
            jpg_name_RGB = osp.basename(img_path_RGB)
            
            # --- 【核心逻辑修正】 ---
            # 1. 只查找RGB图片的标注
            caption = caption_map.get(jpg_name_RGB)

            # 2. 如果RGB图片本身就没有标注，说明数据有问题，直接跳过这张图
            if not caption:
                warnings.warn(f"图片 {jpg_name_RGB} 在JSON文件中没有找到对应的标注，已跳过。")
                continue

            # 构造其他模态的图片路径
            jpg_name_NI = jpg_name_RGB.replace("_m1", "_m2")
            jpg_name_TI = jpg_name_RGB.replace("_m1", "_m3")
            img_path_NI = osp.join(dir_path, 'IR', jpg_name_NI)
            img_path_TI = osp.join(dir_path, 'Thermal', jpg_name_TI)

            # 确保所有模态的物理图片文件都存在
            if not (osp.exists(img_path_RGB) and osp.exists(img_path_NI) and osp.exists(img_path_TI)):
                continue

            img_paths = [img_path_RGB, img_path_NI, img_path_TI]

            match = pattern.search(jpg_name_RGB)
            if not match: continue

            pid_str, camid_str = match.groups()
            
            if pid_str not in pid_container: continue
            
            camid = int(camid_str) - 1
            pid = pid2label[pid_str]

            # 3. 让所有模态共用这一个找到的RGB标注
            if self.prefix:
                text_annotation_RGB = f"An image of a {self.prompt}person in the visible spectrum: {caption}"
                text_annotation_NI = f"An image of a person in the near infrared spectrum: {caption}"
                text_annotation_TI = f"An image of a person in the thermal infrared spectrum: {caption}"
            else:
                text_annotation_RGB = caption
                text_annotation_NI = caption
                text_annotation_TI = caption
            
            trackid = -1
            dataset.append((img_paths, pid, camid, trackid, text_annotation_RGB, text_annotation_NI, text_annotation_TI))

        return dataset