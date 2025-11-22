# AI Dermatological Diagnosis Tool

## Project Description
This project is an AI-powered dermatological diagnosis tool that leverages deep learning models for skin lesion classification. The tool uses TensorFlow/Keras to predict and analyze dermatoscopic images, identifying key skin conditions to assist in early detection and diagnosis. The frontend application is built using Streamlit, providing a user-friendly web interface for image upload, real-time prediction, Grad-CAM-based explainability, and personalized clinical recommendations.

## Features
- Image upload or webcam capture for dermatoscopic lesion images
- Deep learning-based classification of 7 common skin lesion types
- Grad-CAM heatmap visualization for AI model explainability
- Personalized preliminary diagnosis and urgency recommendations
- Comparative lesion image analysis and skincare formulation suggestions
- History tracking with analytics dashboard
- Support for custom TensorFlow model upload

## Dataset
The model is trained on the HAM10000 dataset, a large collection of multi-source dermatoscopic images representing seven diagnostic categories:
- Actinic keratoses (akiec)
- Basal cell carcinoma (bcc)
- Benign keratosis-like lesions (bkl)
- Dermatofibroma (df)
- Melanocytic nevi (nv)
- Melanoma (mel)
- Vascular lesions (vas)

Dataset link: [HAM10000 on Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)  
For more information, see: [Dataset paper](https://arxiv.org/ftp/arxiv/papers/1803/1803.10417.pdf)

## Installation

1. Clone the repository to your local machine.
2. It is recommended to use a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit web app with:

```bash
streamlit run app.py
```

- Upload a dermatoscopic image or capture one via your webcam.
- The AI model will analyze the image and display predicted skin lesion type with confidence scores.
- View Grad-CAM visualization to understand model focus areas.
- Access personalized preliminary diagnostics and clinical recommendations.
- Explore comparative image analysis, history dashboard, advanced tools, and educational content through the web interface.

To use a custom model, upload your TensorFlow `.h5` model file in the advanced settings panel.

## Model Details

- Framework: TensorFlow / Keras
- Model file: `cifar_model.h5` (default)
- Input image size: 32x32 RGB images
- Classes: 7 skin lesion types as per HAM10000 dataset
- Explainability: Grad-CAM heatmaps showing attention in lesion areas

## Medical Disclaimer

This tool is intended for educational and research purposes only and provides a preliminary AI assessment. It is not a substitute for professional medical diagnosis. Always consult a qualified dermatologist or healthcare professional for proper evaluation and treatment.

## Developer

Developed by: Keeistu M S  
For questions or support, please contact the developer.
