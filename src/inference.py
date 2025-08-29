import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from model import MergeModel, MergeModel_ConvNeXt, MergeModel_Enet
from load_dataset import load_datasets, load_weights
from utils import score, create_training_solution

MODEL_DICT = {"regnety": MergeModel, "ori": MergeModel, 
              "convnext": MergeModel_ConvNeXt, 
              "enet": MergeModel_Enet}

def infer(X, label,
          backbone = "regnety", pretrain = "byol", 
          fold_id = "1"):

    # Initialize model
    model = MODEL_DICT[backbone]()

    # Load trained model weights
    weights = load_weights(pretrain = pretrain, fold_id = fold_id)
    # weights = torch.load("./weight.bin", weights_only = True, map_location = torch.device('cpu'))
    # for key in list(weights.keys()):
    #     if "projector" in key: del weights[key]
    model.load_state_dict(weights)

    # Inference
    with torch.no_grad():
        pred = model(*X)[: -1] # Exclude any_injury, pred: (5, 1, 2 | 3)
    pred = [F.softmax(out, dim = 1) for out in pred]
    pred = np.concatenate([np.array(out.detach().numpy()) for out in pred], axis = -1).flatten() # (13, )

    return pred

def get_score(pred, true_df):
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

if __name__ == "__main__":
    # result = infer()
    # print(result)
     # Initialize model
    backbone = "regnety"
    pretrain = "byol"
    fold_id = "1"
    
    model = MODEL_DICT[backbone]()

    # Load trained model weights
    weights = load_weights(pretrain = pretrain, fold_id = fold_id)
    # weights = torch.load("./weight1.bin", weights_only = True, map_location = torch.device('cpu'))
    # print(weights.keys())
    # for key in list(weights.keys()):
    #     if "projector" in key: del weights[key]
    model.load_state_dict(weights)
    print("Successfully load model weights!")
    # model.extractor.load_state_dict(weights["extractor"])
    # model.classifier.load_state_dict(weights["classifier"])
