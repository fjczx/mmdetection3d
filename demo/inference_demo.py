#%%
from cgi import test
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
