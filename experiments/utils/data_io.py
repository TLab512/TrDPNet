import os
import json
from collections import OrderedDict
import numpy as np

try:
    from experiments.config.structured import PointCloudDatasetConfig
except ImportError:
    from config.structured import PointCloudDatasetConfig


def id_to_name(id, category_list):
    for k, v in category_list.items():
        if v[0] <= id < v[1]:
            return k, id - v[0]


def category_model_id_pair(dataset_portion=None, cfg: PointCloudDatasetConfig = None):
    '''
    Load category, model names from a shapenet dataset.
    '''
    category_name_pair = []  # full path of the objs files

    cats = json.load(open(cfg.dataset))
    cats = OrderedDict(sorted(cats.items(), key=lambda x: x[0]))
    list_prefix = "softras_"
    if dataset_portion == "train":
        file_list = list_prefix + "train.lst"
    elif dataset_portion == "val" or dataset_portion == "vis":
        file_list = list_prefix + "val.lst"
    elif dataset_portion == "test":
        file_list = list_prefix + "test.lst"
    else:
        file_list = list_prefix + "train.lst"
    for k, cat in cats.items():  # load by categories
        if cfg.category_id is not None and cat["id"] not in cfg.category_id:
            continue
        model_path = os.path.join(cfg.dataset_path, f"{cat['id']}/{file_list}")
        with open(model_path, "r") as f:
            portioned_models = [x.strip() for x in f.readlines()]

        category_name_pair.extend([(cat['id'], model_id) for model_id in portioned_models])

    print('lib/data_io.py: model paths from %s' % (cfg.dataset))

    return category_name_pair


def write_list_to_lst_file(lst, filename):
    # 提取目录路径
    directory = os.path.dirname(filename)

    # 如果目录不存在，则创建目录
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            print(f"创建目录时出错: {e.strerror}")
            return

    # 删除旧文件（如果存在）
    try:
        if os.path.exists(filename):
            os.remove(filename)
    except OSError as e:
        print(f"删除文件时出错: {e.strerror}")

    # 立即以写入模式打开文件，确保文件存在
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for item in lst:
                file.write("%s\n" % item)
        print(f"成功写入文件 {filename}")
    except IOError as e:
        print(f"写入文件时出错: {e.strerror}")

def get_point_cloud_file(category, model_id, cfg: PointCloudDatasetConfig = None):
    return cfg.pcd_path % (category, model_id)


def get_voxel_file(category, model_id, cfg: PointCloudDatasetConfig = None):
    return cfg.voxel_path % (category, model_id)


def get_rendering_file(category, model_id, rendering_id, cfg: PointCloudDatasetConfig = None):
    return os.path.join(cfg.rendering_path % (category, model_id), '%02d.png' % rendering_id)

def get_text_file(category, model_id, rendering_id, cfg: PointCloudDatasetConfig = None):
    return os.path.join(cfg.text_path % (category, model_id), '%02d.txt' % rendering_id)

def get_camera_info(category, model_id, rendering_ids, cfg: PointCloudDatasetConfig = None):
    meta_file = os.path.join(cfg.camera_path % (category, model_id))
    cameras = []
    with open(meta_file, 'r') as f:
        for i, line in enumerate(f):
            ls = line.strip().split()
            cameras.append(list(map(float, ls)))
    res = []
    for idx in rendering_ids:
        res.append(cameras[idx])
    return res

from pytorch3d.io import IO
io = IO()

def read_point_set(fp):
    pointcloud = io.load_pointcloud(fp)
    return pointcloud.points_padded().flatten(0, 1)
