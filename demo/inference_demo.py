#%%
import sys, os
from mmdet3d.apis import init_model, inference_detector, show_result_meshlab
from mmdet3d.datasets import build_dataset
from mmcv import Config
#%%
# change the workdir 
os.chdir('/home/lazurite/code/mmdetection3d')
config_file = '/home/lazurite/code/mmdetection3d/configs/centerpoint/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus.py'
cfg = Config.fromfile(config_file)
#%%
# visualize the dataset
# python tools/misc/browse_dataset.py configs/_base_/datasets/kitti-3d-3class.py --task det --output-dir vis --online

# visualize the prediction result
# python demo/pcd_demo.py demo/data/kitti/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth --out-dir=output/ --show

#%%
model = init_model(config_file, checkpoint=None, device='cuda:0')
#%%
train_dataset = build_dataset(cfg.data.train)
test_dataset = build_dataset(cfg.data.test)
eval_dataset = build_dataset(cfg.data.val)
# %%
data_sample = train_dataset[0]
data_keys = data_sample.keys()
print("The Keys of the data sample are:", data_keys)
# The Keys of the data sample are: 
# ['img_metas', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])

# %%
pcd = './data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin'
result, data = inference_detector(model, pcd)
show_result_meshlab(data, result, show=False, out_dir='out_nus')

# %%
# %%
