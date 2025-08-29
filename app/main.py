import os
import streamlit as st
import torch
import sys
sys.path.append("src")

from load_dataset import load_datasets, get_avail_ids, get_avail_models
from inference import infer, get_score

st.markdown("<div style='text-align:center;font-size:48px'>Abdominal Trauma Scans Analyzer</div>", unsafe_allow_html=True)
st.subheader("Step 1. Select a scan.")

scan_options = get_avail_ids()
model_options = get_avail_models()
selected_scan = st.selectbox("Please select a scan", scan_options)
selected_model = st.selectbox("Please select a model", model_options)

if st.button("Start analyze"):
    with st.spinner("Fetching the scans from dataset..."):
        (X, label, true_df) = load_datasets(*selected_scan)

    with st.spinner("Analyzing..."):
        pretrain, *backbone = selected_model.split("/")
        backbone = "regnety" if len(backbone) == 0 else backbone[0]
        pred = infer(X = X, label = label, 
                     backbone = backbone, pretrain = pretrain)

    with st.spinner("Calculating the score for each organ"):
        pred_df, true_df, score_df, avg_score = get_score(pred = pred, true_df = true_df)
    
    st.subheader("Analyze result")
    st.write("Model prediction")
    st.write(pred_df)
    st.write("Ground Truth")
    st.write(true_df)
    st.markdown("<div style='text-align:center;font-size:42px'>Final Score</div>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;font-size:36px;background-image:linear-gradient(to right,#4CAF50, #40fff5);color:transparent;background-clip:text;background-size: 100% auto;'>{avg_score}</p>", unsafe_allow_html=True)
    st.write("Score details")
    st.write(score_df)