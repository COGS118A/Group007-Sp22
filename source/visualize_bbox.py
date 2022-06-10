import matplotlib.pyplot as plt
import numpy as np
import torch
import os
# import pyautogui  # simulate key presses

from tqdm import tqdm
import qdtrack.apis as api
from qdtrack.apis import show_result_pyplot, inference_model

# Load constants
cfg = 'configs/bdd100k/qdtrack-frcnn_r50_fpn_12e_bdd100k.py'
# checkpoint = 'models/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed.pth'
checkpoint = 'models/epoch_12_0.005.pth'
# img_folder_path = 'data/bdd/images/track/val/b1c9c847-3bda4659'   # val: traffic
# img_folder_path = 'data/bdd/images/track/test/cad7fdff-d9946f73'   # test: pedestrians
# img_folder_path = 'data/bdd/images/track/test/cabe1040-c59cb390'   # test: foggy traffic
# img_folder_path = 'data/bdd/images/track/test/cac47e88-3227e13a'  # test: highway with sunset
# img_folder_path = 'data/bdd/images/track/test/cb078020-b82ebd77'  # test: local
img_folder_path = 'data/bdd/images/track/test/cabc9045-d91ecb66'  # test: night local
out_folder_name = os.path.basename(img_folder_path)
out_path = 'reports'
imgs_path = [os.path.join(img_folder_path, img) for img in os.listdir(img_folder_path)]
imgs = [plt.imread(img) for img in imgs_path]
print(imgs_path)

# Init model
model = api.init_model(cfg, checkpoint, device="cpu")
model.init_tracker()
# Visualize the bbox
print('Visualizing bbox ...')
for i, img in tqdm(enumerate(imgs), total=len(imgs)):
    # Forward model (we visualize img one-by-one as the inference model is not functioning correctly)
    results = inference_model(model=model, imgs=[img], frame_id=1)

    # Show results
    out_file = os.path.join(out_path, out_folder_name, os.path.basename(imgs_path[i]))
    show_result_pyplot(model, img, results, out_file=out_file, show=False)




