import logging
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import streamlit as st

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import MergeModel, MergeModel_ConvNeXt, MergeModel_Enet
from load_dataset import load_datasets, load_weights
from utils import score, create_training_solution

MODEL_DICT = {"regnety": MergeModel, "ori": MergeModel, 
              "convnext": MergeModel_ConvNeXt, 
              "enet": MergeModel_Enet}

class InferModel:
    def __init__(self, label, backbone = "regnety", pretrain = "byol", fold_id = "1"):
        self.label = label
        self.backbone = backbone
        self.pretrain = pretrain
        self.fold_id = fold_id

        # Create model and load model weights
        self.model = MODEL_DICT[backbone]()
        self.weights = load_weights(backbone = self.backbone, 
                                    pretrain = self.pretrain, 
                                    fold_id = self.fold_id)
        self.model.load_state_dict(self.weights)

        # Generate GramCam visualization for each organ
        self.visuals = []
    
    def infer(self, X):
        with torch.no_grad():
            pred = self.model(*X)[: -1] # Exclude any_injury, pred: (5, 1, 2 | 3)
        pred = [F.softmax(out, dim = 1) for out in pred]
        pred = np.concatenate([np.array(out.detach().numpy()) for out in pred], axis = -1).flatten() # (13, )

        return pred
    
    def get_score(self, pred, true_df):
        # Load organ ROI videos and label for the scan
        pred_df = true_df.copy()

        for idx, col_name in enumerate(pred_df.columns[1 : -1]):
            pred_df[col_name] = pred[idx]

        infer_score = score(create_training_solution(true_df), pred_df, "patient_id", reduction = 'none')
        organ_names = ["bowel", "extravasation", "kidney", "liver", "spleen"]

        assert len(infer_score) == len(organ_names)
        
        score_df = pd.DataFrame(dict(zip(organ_names, infer_score)), index = [0])

        avg_score = sum(infer_score) / len(infer_score)

        return pred_df, true_df, score_df, avg_score

    
    def get_gradcam(self, X):
        logging.info("Start generating GradCam visualization...")

        for idx, organ in enumerate(["full", "kidney", "liver", "spleen"]):
            # Get backbone (timm CNN model) and its last convolutional layer
            organ_backbone = getattr(self.model.extractor, f"{organ}_extractor")
            organ_backbone_last_conv = [organ_backbone.cnn.s4.b7.conv2.conv]

            images = X[idx] # full/kidney/liver/spleen video

            # Check video's dimension
            if len(images.shape) == 3: # images: (D, H, W)
                images = images.unsqueeze(1).repeat(1, 3, 1, 1)

            if not (len(images.shape) == 4 and images.shape[1] == 3):
                raise Exception(f"Expect images shape (D, 3, H, W), but get {images.shape}!")
            
            # Generate GradCAM mask
            cam = GradCAM(model = organ_backbone, target_layers = organ_backbone_last_conv)
            masks = cam(input_tensor = images, targets = None) # gray_cam: (96, 256, 256)
            rgb_images = images.squeeze(0).permute(0, 2, 3, 1).numpy() # (96, 256, 256, 3)

            # Apply heatmap on original images
            visual = [show_cam_on_image(rgb_image, mask, use_rgb = True) for rgb_image, mask in zip(rgb_images, masks)]
            
            self.visuals.append(visual)

        logging.info("GradCam visualization generation complete!")

# Use `@st.cache_resource` for functions that load complex objects like models
@st.cache_resource(show_spinner = False)
def cached_load_model(selected_model):
    """Initializes and caches the model."""
    pretrain, *backbone = selected_model.split("/")
    backbone = "regnety" if len(backbone) == 0 else backbone[0]
    return InferModel(label="placeholder", backbone=backbone, pretrain=pretrain)

@st.cache_data(show_spinner = False)
def cached_infer(_infer_model, _X):
    return _infer_model.infer(X = _X)

@st.cache_data(show_spinner = False)
def cached_get_score(_infer_model, _pred, _true_df):
    return _infer_model.get_score(pred = _pred, true_df = _true_df)

# Use `@st.cache_data` for functions that perform heavy computation
@st.cache_data(show_spinner = False)
def cached_get_gradcam_visuals(_infer_model, _X):
    """Generates and caches Grad-CAM visualizations."""
    _infer_model.get_gradcam(X = _X)
    return _infer_model.visuals


if __name__ == "__main__": 
    (X, label, true_df) = load_datasets()

    infer_model = InferModel(label = label)

    infer_model.get_gradcam(X = X)
    visuals = infer_model.visuals

    fig, axs = plt.subplots(6, 16, figsize = (20, 20))
    for idx, img in enumerate(visuals[1]):
        ax = axs[idx // 16][idx % 16]
        ax.imshow(img)
        ax.axis("off")

    fig.tight_layout()
    plt.show()
