# RNSA2023_Challenge_Demo

This project is an interactive [`Streamlit`](https://streamlit.io/) application designed to assist in the analysis of abdominal trauma CT images. The application integrates training result from [my thesis](https://github.com/MorrisCheng0414/RSNA2023-Challenge-SSL-SCL) with [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) visualization, providing an intuitive interface for exploring model predictions and understanding the reasoning behind them.

## Key Features

+ **Interactive UI**: Built with `Streamlit`, the application provides a clean and intuitive user interface for exploring model outputs.

+ **Model Inference**: Supports various model weights for predicting injuries outcomes from CT scans.

+ **Grad-CAM Heatmaps**: Generates visual heatmaps that highlight the specific regions of an image the 2D CNN in the model focused on when inferencing.

+ **Data Visualization**: Utilizes interactive bar plots and pie charts to present model predictions and scores, moving away from dense, hard-to-read tables.

+ **Performance Optimization**: Employs `Streamlit`'s caching system to avoid costly operations, such as fetching images, reloading models and regenerating heatmaps, ensuring a smooth and responsive user experience.

## Demo video



## Requirements
- `altair`
- `datasets`
- `grad-cam`
- `huggingface_hub`
- `numpy`
- `pandas`
- `streamlit`
- `timm`
- `torch`
- `torchvision`
- `transformers`

## Usage
Please follow these steps to set up and run the application from the source code.
1.  **Clone the repository**
  ```bash
  git clone https://github.com/MorrisCheng0414/RNSA2023_Challenge_Demo.git
  cd RNSA2023_Challenge_Demo
  ```

2.  **Install Python Libraries**

    Install all necessary dependencies by running the following command:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**

    Start the application with the `streamlit run` command. The application will be opened in your browser.
    ```bash
    streamlit run app/main.py
    ```

## Folder Structure

```
.
├── app/
│   └── main.py          # Streamlit main application
├── data/
│   ├── train.csv
│   └── train_series_meta.csv
├── src/
│   ├── inference.py     # Model inference
│   ├── load_dataset.py  # Fetch images, weights from Hugging Face
│   ├── model.py         # Model definition
│   └── utils.py         
├── requirements.txt
└── README.md
```
