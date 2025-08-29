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

def infer(patient_id = "10007", series_id = "47578", 
          backbone = "regnety", pretrain = "byol", 
          fold_id = "1"):
    # Load organ ROI videos and label for the scan
    (X, label, true_df) = load_datasets(patient_id = patient_id, series_id = series_id)
    pred_df = true_df.copy()

    # Initialize model
    model = MODEL_DICT[backbone]()

    # Load trained model weights
    weights = torch.load("./weight.bin", weights_only = True, map_location = torch.device('cpu'))
    for key in list(weights.keys()):
        if "projector" in key: del weights[key]
    # weights = load_weights(pretrain = pretrain, fold_id = fold_id)
    model.load_state_dict(weights)
    # model.extractor.load_state_dict(weights["extractor"])
    # model.classifier.load_state_dict(weights["classifier"])

    # Inference
    with torch.no_grad():
        output = model(*X)[: -1] # Exclude any_injury, output: (5, 1, 2 | 3)
    output = [F.softmax(out, dim = 1) for out in output]
    output = np.concatenate([np.array(out.detach().numpy()) for out in output], axis = -1).flatten() # (13, )

    for idx, col_name in enumerate(pred_df.columns[1 : -1]):
        pred_df[col_name] = output[idx]

    infer_score = score(create_training_solution(true_df), pred_df, "patient_id", reduction = 'none')
    organ_names = ["bowel", "extravasation", "kidney", "liver", "spleen"]

    assert len(infer_score) == len(organ_names)
    
    print(true_df.values)
    print(pred_df.values)

    return dict(zip(organ_names, infer_score))

def infer(X, label,
          backbone = "regnety", pretrain = "byol", 
          fold_id = "1"):

    # Initialize model
    model = MODEL_DICT[backbone]()

    # Load trained model weights
    weights = torch.load("./weight.bin", weights_only = True, map_location = torch.device('cpu'))
    for key in list(weights.keys()):
        if "projector" in key: del weights[key]
    # weights = load_weights(pretrain = pretrain, fold_id = fold_id)
    model.load_state_dict(weights)
    # model.extractor.load_state_dict(weights["extractor"])
    # model.classifier.load_state_dict(weights["classifier"])

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
    result = infer()
    print(result)
