## Instructions
This page provide instructions to plot the feature embeddings of model.

Made modification to qdtrack.py and quasi_dense_embed_tracker.py to extract feature embeddings and object ids. 

Replace qdtrack/qdtrack/models/mot/qdtrack.py in the QDTrack repo with qdtrack.py in this folder.

Replace qdtrack/qdtrack/models/trackers/quasi_dense_embed_tracker.py in the QDTrack repo with quasi_dense_embed_tracker.py in this folder.

Run the plot_embeddings.py to plot the embeddings. 
```bash
python plot_embeddings.py --config PATH_TO_THE_CONFIG_FILE --ckpt PATH_TO_THE_CHECKPOINT_FILE --imgs PATH_TO_THE_FOLDER_OF_A_VIDEO --pca PATH_TO_SAVE_PCA_PLOT --tsne PATH_TO_SAVE_TSNE_PLOT
```