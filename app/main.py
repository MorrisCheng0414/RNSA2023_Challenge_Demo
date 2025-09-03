import streamlit as st
import sys
import torch
import altair as alt
import pandas as pd
sys.path.append("src")

from load_dataset import load_datasets, get_avail_ids, get_avail_models
from inference import (cached_infer, 
                       cached_get_score,
                       cached_get_gradcam_visuals, 
                       cached_load_model)
# from utils import plot_result

ORGANS = ["full", "kidney", "liver", "spleen"]
CLASSES = ["Bowel", "Extravasation", "Kidney", "Liver", "Spleen"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

if "is_analyzed" not in st.session_state:
    st.session_state.is_analyzed = False

def set_analyzed_state():
    st.session_state.is_analyzed = True

def plot_result(pred_df, true_df):
    merge_df = pd.merge(pred_df.T, true_df.T, left_index = True, right_index = True, how = "inner")
    merge_df = merge_df.iloc[1 : -1, :]
    merge_df.columns = ["Prediction", "Ground Truth"]
    # merge_df.to_csv("./merge_df.csv")
    merge_df = merge_df.astype(float).round(4)

    merge_df = merge_df.reset_index()
    
    tabs = st.tabs(CLASSES)

    for idx, cls in enumerate(CLASSES):
        with tabs[idx]:
            class_df = merge_df[merge_df["index"].str.contains(cls.lower())]
            
            # Define chart
            chart = alt.Chart(class_df.melt('index'))
            chart = chart.mark_bar()
            chart = chart.encode(
                alt.X(
                    'value:Q', 
                    axis = alt.Axis(
                        title = 'probability', 
                        grid = True)
                    ),
                alt.Y(
                    'variable:N', 
                    axis = alt.Axis(
                        title = '', 
                        grid = False)
                    ),
                color = alt.Color(
                    'variable:N', 
                    scale=alt.Scale(
                        domain=['Ground Truth', 'Prediction'], 
                        range=['#1F77B4', '#FF7F0E']), title = ""
                    ),
                )
            # Define text
            text = chart.mark_text(
                align = "left",
                baseline = "middle",
                dx = 5,
                size = 16,
                )
            text = text.encode(
                text = "value:Q",
                )
            
            # Add values to bars
            combine_chart = chart + text
            
            # Chart configuration
            combine_chart = combine_chart.properties(height = 80, width = 500) # set chart size (width = "container" isn't working)
            combine_chart = combine_chart.transform_calculate(y = "split(datum.index, '_')") # create new line in class names
            combine_chart = combine_chart.facet(row = alt.Row('y:O', title = "",)) # grouped bars by class
            combine_chart = combine_chart.configure_view(stroke = "transparent")
            combine_chart = combine_chart.configure_axisY(labels = False)

            st.altair_chart(combine_chart)

def plot_pie(avg_score, score_df, radius = 120):
    avg_score = round(avg_score, 6)

    pie_df = score_df.T
    pie_df.columns = ["score"]
    pie_df["score"] = pie_df["score"].astype(float)
    total_df = pd.DataFrame({"total": [avg_score]})
    
    pie_df = pie_df.reset_index(names = "organ")
    
    chart = alt.Chart(pie_df).encode(theta = alt.Theta("score:Q").stack(True), 
                                     color = alt.Color("organ:N", scale=alt.Scale(
                        range= COLORS[ : pie_df.shape[0]]), 
                        title = "",
                        sort = "descending",
                    ).legend(None))
    
    chart = chart.mark_arc(innerRadius = radius // 2, outerRadius = radius)
    
    text = chart.mark_text(radius = radius + 30, size = 14, baseline = "middle")
    text = text.encode(text = "organ:N")
    
    mid_text = alt.Chart(total_df)
    mid_text = mid_text.mark_text(
        align = "center", 
        baseline = "middle", 
        color = "white",
        size = radius // 6,
        )
    mid_text = mid_text.encode(
        text = "total:Q",
        )

    combine_chart = chart + text + mid_text

    col1, col2 = st.columns([2, 3])
    with col1:
        # st.markdown(f"<div style='text-align:center;font-size:24px'>Final Score: {avg_score}</div>", unsafe_allow_html=True)
        st.markdown(f"""
                    <p style='
                    text-align:center;
                    font-size:30px;
                    background-image:linear-gradient(to right,#4CAF50, #40fff5);
                    color:transparent;
                    background-clip:text;
                    background-size: 100% auto;
                    '>Final Score: {avg_score}</p>
                    """, unsafe_allow_html=True)
        # st.markdown("<style>.col_heading{text-align: center;}</style>", unsafe_allow_html = True)
        # pie_df.columns = ['<div class="col_heading">'+col+'</div>' for col in pie_df.columns]
        # st.write(pie_df.to_html(escape = False), unsafe_allow_html = True)
        st.dataframe(pie_df, hide_index = True)
    with col2:
        st.altair_chart(combine_chart)


st.markdown("<div style='text-align:center;font-size:48px'>Abdominal Trauma Scans Analyzer</div>", unsafe_allow_html=True)
st.divider()
st.subheader("Please select a scan")

scan_options = get_avail_ids()
model_options = get_avail_models() 
selected_scan = st.selectbox("Please select a scan", scan_options)
selected_model = st.selectbox("Please select a model", model_options)


st.button("Analyze", on_click = set_analyzed_state)
st.divider()

if st.session_state.is_analyzed:
    visuals = [] # GradCAM images
    with st.spinner("Fetching the scans from dataset..."):
        (X, label, true_df) = load_datasets(*selected_scan)
        # (X, label, true_df) = cached_load_datasets(*selected_scan)

    with st.spinner("Creating model..."):
        infer_model = cached_load_model(selected_model = selected_model)

    with st.spinner("Analyzing..."):
        pred = cached_infer(infer_model, X)

    with st.spinner("Calculating the score for each organ"):
        pred_df, true_df, score_df, avg_score = cached_get_score(_infer_model = infer_model, _pred = pred, _true_df = true_df)
    
    st.subheader("Analysis result")
    
    plot_result(pred_df = pred_df, true_df = true_df)
    
    st.write("Score details")

    # st.dataframe(score_df, hide_index = True)
    plot_pie(avg_score = avg_score, score_df = score_df)

    # st.write(score_df)
    st.divider()
    st.subheader("GradCAM visualization")

    with st.spinner("Generate GradCam visualization"):
        visuals = cached_get_gradcam_visuals(infer_model, X)

    scan_types = ["Original", "Kidney", "Liver", "Spleen"]

    tabs = st.tabs(scan_types)

    for idx, tab in enumerate(tabs):
        gradcam_imgs = visuals[idx]
        ori_imgs = X[idx]
        with tab:
            # st.header(f"{scan_types[idx]} video")
            num_images = len(gradcam_imgs)
            selected_frame = st.slider("Choose a video frame", 0, num_images - 1, 0, key = f"slider_{idx}")
            
            gradcam_to_show = gradcam_imgs[selected_frame]
            ori_imgs = ori_imgs[selected_frame]

            _ , col, _ = st.columns([1, 2, 1])
            with col:
                st.image(gradcam_to_show, caption = "GradCAM visualization", use_container_width = True)


