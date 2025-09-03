import os
import torch
import pandas as pd
import numpy as np
import datasets
import logging
import huggingface_hub
import streamlit as st
from time import time
from tqdm import tqdm
from model import MergeModel
from PIL import Image

logging.basicConfig(level = logging.INFO, format = "%(asctime)s-%(levelname)s-%(message)s")

DATASET_REPO_ID = "Morris-is-taken/RSNA_23_SSL_SCL_Demo_Data"
WEIGHTS_REPO_ID = "Morris-is-taken/RSNA_23_3D_Pretrain_weight"
IMG_SIZE = 256

@st.cache_data
def load_weights(backbone = "regnety", pretrain = "byol", fold_id = "1"):
    if backbone != "regnety":
        pretrain = '_'.join([pretrain, backbone])

    filename = f"{pretrain}/fold{fold_id}/weight.bin"
    try:
        cache_path = huggingface_hub.hf_hub_download(repo_id = WEIGHTS_REPO_ID, 
                                                     filename = filename,
                                                     repo_type = "dataset")
        logging.info(f"Succesfully load weights: {filename}")
    except Exception as e:
        logging.error(f"Failed to load weights: {filename}")
        raise(e)
    
    weights = torch.load(cache_path, weights_only = True, map_location = torch.device('cpu'))
    for key in list(weights.keys()):
        if "projector" in key: del weights[key]
    return weights

@st.cache_data
def load_label(patient_id = "10007", series_id = "47578"):
    train_df = pd.read_csv("data/train.csv")
    meta_df = pd.read_csv("data/train_series_meta.csv")
    
    merge_df = train_df.merge(meta_df, how = "inner", on = "patient_id") # Merge patient_id, series_id, and label information
    merge_df.drop(["aortic_hu", "incomplete_organ"], axis = 1, inplace = True) # Drop unused column
    merge_df[["patient_id", "series_id"]] = merge_df[["patient_id", "series_id"]].astype(str) # Convert ID from int to str
    # Move column
    series_id_col = merge_df.pop("series_id")
    merge_df.insert(1, series_id_col.name, series_id_col)

    row = merge_df.loc[(merge_df["patient_id"] == f"{patient_id}") & (merge_df["series_id"] == f"{series_id}")]
    logging.debug(f"Current row for {patient_id}-{series_id}: {row}")
    bowel, extra, kidney, liver, spleen = row.iloc[:, 2 : 4], row.iloc[:, 4 : 6], row.iloc[:, 6 : 9], row.iloc[:, 9 : 12], row.iloc[:, 12 : 15]
    bowel, extra, kidney, liver, spleen = np.argmax(bowel), np.argmax(extra), np.argmax(kidney), np.argmax(liver), np.argmax(spleen)

    true_df = row.drop("series_id", axis = 1)

    return (bowel, extra, kidney, liver, spleen), true_df

@st.cache_data
def load_datasets(patient_id = "10007", series_id = "47578"):
    # Load organ ROI videos of the scan
    try:
        X = []
        for organ in ["images", "kidney", "liver", "spleen"]:
            logging.info(f"Loading scan {patient_id}-{series_id}-{organ}...")
            ds = datasets.load_dataset(DATASET_REPO_ID,
                                       data_files = f"{patient_id}/{series_id}/{organ}/*.png",
                                    #    streaming = True
                                       )
            ds = ds["train"]
            logging.info(f"Converting {organ} dataset to PyTorch tensor...")
            # ds_numpy = ds.with_format("numpy")
            # ds_numpy = np.concat([example["image"] for example in tqdm(ds_numpy)])
            ds_numpy = np.array([np.array(example["image"]) / 255.0 for example in ds])
            logging.info(f"Conversion complete!")

            # Sample images to 96
            sample_idx = np.linspace(start = 0, stop = len(ds_numpy) - 1, num = 96, endpoint = True, dtype = np.uint8)
            ds_numpy = np.array([ds_numpy[idx] for idx in sample_idx])
            
            X.append(torch.tensor(ds_numpy, dtype = torch.float32))

        logging.info(f"Successfully load scan {patient_id}-{series_id}!")

    except Exception as e:
        logging.error("Failed to load datasets!")
        raise(e)
    
    # Load the label of the scan
    try:
        logging.info(f"Loading label of {patient_id}-{series_id}...")
        label, true_df = load_label(patient_id = patient_id, series_id = series_id)
        logging.info(f"Successfully load label {patient_id}-{series_id}!")

    except Exception as e:
        logging.error(f"Failed to load labels!")
        raise(e)
        
    return (X, label, true_df) # X: (4, 96, 256, 256), label: (5, ), true_df: (1, 14)

@st.cache_data 
def get_avail_ids(dest_path = "data/available_ids.csv"):
    if os.path.exists(dest_path):
        df = pd.read_csv(dest_path)
    
    else:
        api = huggingface_hub.HfApi()
        files = api.list_repo_files(repo_id = DATASET_REPO_ID, repo_type = "dataset")
        
        ids = set((file.split("/")[0], file.split("/")[1]) for file in files if file.endswith(".png"))

        df = pd.DataFrame(data = ids, columns = ["patient_id", "series_id"])
        df.to_csv(dest_path, index = False)

    return df.values.tolist()

@st.cache_data
def get_avail_models(dest_path = "data/available_models.csv"):
    if os.path.exists(dest_path):
        df = pd.read_csv(dest_path)
    
    else:
        api = huggingface_hub.HfApi()
        files = api.list_repo_files(repo_id = WEIGHTS_REPO_ID, repo_type = "dataset")
        
        ids = set(file.split("/")[0] for file in files if file.endswith(".bin"))

        df = pd.DataFrame(data = ids, columns = ["model"])
        df.to_csv(dest_path, index = False)

    return df["model"].values.tolist()
    # return df.values.tolist()

if __name__ == "__main__":
    start_t = time()
    # print(get_avail_ids())
    # print(df.head())
    # avail_list = pd.read_csv("data/available_ids.csv")
    # print(avail_list.values.tolist())
    print(get_avail_models())
    end_t = time()
    logging.info(f"Time to load a scan: {end_t - start_t}")
    
    