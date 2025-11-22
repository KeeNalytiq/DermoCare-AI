# 🧴 AI Dermatological Diagnosis Tool

## 🧪 Overview

This project is a **full-fledged AI-powered dermatological diagnosis system** designed to analyze dermatoscopic images and provide **skin lesion classification**, **explainability visualizations**, and **clinical decision-support insights**. The tool combines a deep learning model (TensorFlow/Keras) with an interactive **Streamlit web application**, making skin lesion screening accessible, intuitive, and scalable.

The goal is to assist clinicians, researchers, and users with **preliminary skin health assessment**, not to replace professional diagnosis.

---

## 🚀 Key Features

### 🌄 Image Input Options

* Upload dermatoscopic images (JPG/PNG)
* Capture images directly using the **webcam**

### 🤖 AI-Powered Classification

* Predicts **7 classes** of skin lesions from the HAM10000 dataset:

  * Actinic keratoses (akiec)
  * Basal cell carcinoma (bcc)
  * Benign keratosis-like lesions (bkl)
  * Dermatofibroma (df)
  * Melanocytic nevi (nv)
  * Melanoma (mel)
  * Vascular lesions (vas)

### 🔍 Explainability With Grad-CAM

* Generates heatmaps showing **where the model is looking** during prediction.
* Helps users and clinicians visually interpret AI decisions.

### 📊 Dashboard & Analytics

* Track prediction history
* Visual insights for lesion trends and model confidence

### 🧠 Custom Model Loading

* Upload your own `.h5` TensorFlow model
* Automatically integrates with the Streamlit UI

### 🩺 Preliminary Clinical Support

* Urgency recommendation (routine / early check-up / immediate consultation)
* Skincare advice and lesion characteristic insights

---

## 📁 Dataset

This project uses the **HAM10000 (Human Against Machine)** dataset — one of the most popular and diverse dermatoscopic image datasets.

### 🔗 Dataset Links

* Kaggle dataset: [HAM10000 on Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
* Research paper: [HAM10000 Dataset Paper](https://arxiv.org/ftp/arxiv/papers/1803/1803.10417.pdf)

The dataset includes **10,000+ high-quality dermatoscopic images** across **7 diagnostic categories**, enabling the model to learn real clinical variability.

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone <your_repo_link>
cd <project_folder>
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🧠 How the Model Works

This project uses a **Convolutional Neural Network (CNN)** trained on the HAM10000 dataset.

### 🔹 Model Input

* Images are resized to **32×32 RGB**
* Normalized and fed into the CNN

### 🔹 CNN Architecture (Typical Flow)

```
Input Image → Convolution Layers → ReLU Activation → MaxPooling →
Additional Conv Layers → Dense Layers → Softmax Output
```

### 🔹 Output

* A probability vector of 7 classes
* Highest probability = predicted class

### 🔹 Explainability (Grad-CAM)

Grad-CAM overlays a heatmap on the image:

* Highlights regions contributing most to the prediction
* Shows whether model focused on lesion or background

### 🔹 Why Grad-CAM?

* Increases trust
* Detects model bias
* Supports medical interpretability

---

## 🧬 Workflow (Enhanced)

```
User uploads/captures image
        ↓
Image is preprocessed (resize, normalize)
        ↓
Model predicts skin lesion type
        ↓
Grad-CAM heatmap is generated
        ↓
Streamlit displays:
  • Predicted class
  • Confidence score
  • Heatmap
  • Clinical suggestions
```

---

## 🔧 Project Structure (Recommended)

```
project/
│── app.py                # Streamlit application
│── model.py              # Model loading + preprocessing
│── utils.py              # Helper functions
│── cifar_model.h5        # Default pre-trained model
│── README.md             # Project documentation
│── requirements.txt
└── assets/               # Sample images, icons
```

---

## ⚠️ Medical Disclaimer

This system provides **AI-based preliminary assessment** only.
It is **not** a medical diagnostic tool. Always consult a certified dermatologist.

---

## 👨‍💻 Developer & Contact

**Developed by:** Keeistu M S

### 📬 Contact

* **Email:** [keeistums@gmail.com](mailto:keeistu25@gmail.com)
* **LinkedIn:** [https://www.linkedin.com/in/keeistu-ms](https://www.linkedin.com/in/keeistu17/)

Feel free to reach out for collaboration, improvements, or support.
