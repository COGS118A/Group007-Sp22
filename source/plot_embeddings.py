import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import qdtrack.apis as api
from qdtrack.apis import inference_model, show_result_pyplot


def main():
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('--config', help='path to config file of the model')
    parser.add_argument('--ckpt', help='path to checkpoint of the model')
    parser.add_argument('--imgs', help='path to the images')
    parser.add_argument('--pca', help='path/file name to the PCA plot')
    parser.add_argument('--tsne', help='path/file name to the TSNE plot')

    args = parser.parse_args()

    #initialize the model 
    model = api.init_model(args.config, args.ckpt, device="cpu")
    model.init_tracker() 

    imgs = os.listdir(args.imgs)

    #get the embeddings for all the images
    all_embeddings = np.array([])
    all_labels = np.array([])
    print("Getting embeddings")
    for i, img in tqdm(enumerate(imgs), total=len(imgs)):
        img_path = os.path.join(args.imgs, img)
        
        pic = plt.imread(img_path)
        embeddings, labels = get_embeddings(model, pic)
        
        if i == 0:
            all_embeddings = embeddings
            all_labels = labels
        else:
            all_embeddings = np.concatenate((all_embeddings, embeddings), axis=0)
            all_labels = np.concatenate((all_labels, labels), axis=0)
       
    print(all_embeddings.shape) 
    print(all_labels.shape)

    all_embeddings = all_embeddings[all_labels>=0]
    all_labels = all_labels[all_labels>=0]
    
    #save the embeddings for test
    with open('track_feat.npy', "wb") as f:
        np.save(f, all_embeddings)
    with open('labels.npy', "wb") as f2:
        np.save(f2, all_labels)

    pca_plot = plt_pca(all_embeddings, all_labels)
    pca_plot.savefig(args.pca)
    tsne_plot = plt_tsne(all_embeddings, all_labels)
    tsne_plot.savefig(args.tsne)
    
def plt_pca(data, labels):
    #scale the embeddings
    scaler = StandardScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    
    #PCA with 2 components
    pca = PCA(n_components=2, random_state=1)
    pca.fit(data)
    data_pca = pca.transform(data)

    plt.figure(figsize=(15,10))
    figure = sns.scatterplot(x=data_pca[:,0], y=data_pca[:,1], hue=labels, palette='tab20')
    return figure.get_figure()

def plt_tsne(data, labels):
    #TSNE with 2 components
    tsne = TSNE(n_components = 2, random_state=1)
    transformed_data = tsne.fit_transform(data)
    
    plt.figure(figsize=(15,10))
    figure = sns.scatterplot(x=transformed_data[:,0], y=transformed_data[:,1], hue=labels, palette='tab20')
    return figure.get_figure()
    
def get_embeddings(model, imgs):
    #get embeddings 
    frame_id = 1
    
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False
        
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    
    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    
    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img, frame_id=frame_id)
        else:
            # add information into dict
            data = dict(
                img_info=dict(filename=img, frame_id=frame_id),
                img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)
        
    data = collate(datas, samples_per_gpu=len(imgs))

    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    with torch.no_grad():
        _, results= model.get_embeddings(rescale=True, **data)
    
    if results['embeds'] is not None:
        track_embeddings = results['embeds'].cpu().detach().numpy()
    else:
        track_embeddings = np.array([np.zeros(256)])
        
    if results['ids'] is not None:
        ids = results['ids'].cpu().detach().numpy()
    else:
        ids = np.array([-1])
    #ids = results['bbox'].cpu().detach().numpy()[:, 0].astype(np.int64).astype('str')
    
    return track_embeddings, ids


if __name__ == '__main__':
    main()
