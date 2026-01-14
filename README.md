# Monkeypox Disease Classification Using Deep Learning

This project implements a deep learning–based system for multi-class classification of viral skin diseases with a primary focus on Monkeypox. The system uses a hybrid deep learning architecture trained on skin lesion images and is deployed as a Streamlit web application for easy interaction.

This project is developed purely for **academic and educational purposes** and is not intended for real-world medical diagnosis.

---

## 📌 Project Overview

Monkeypox is a viral infectious disease that causes visible skin lesions similar to other viral skin diseases such as Chickenpox, Measles, and Cowpox. Manual diagnosis based on visual inspection can be challenging due to high similarity between diseases.

This project aims to:
- Automatically classify Monkeypox from skin lesion images
- Apply deep learning techniques for medical image classification
- Analyze model performance using multiple evaluation metrics
- Deploy the trained model as a web application

---

## 🦠 Disease Classes

The system performs **multi-class classification** for the following categories:

- Monkeypox  
- Chickenpox  
- Cowpox  
- Hand, Foot and Mouth Disease (HFMD)  
- Measles  
- Healthy Skin  

---

## 📊 Dataset Details

- **Dataset Name:** Multi-Class Viral Skin Lesion Dataset (MCVSLD)
- **Published:** December 2024
- **Total Classes:** 6
- **Images Used:** 100 images per class (balanced)
- **Image Size:** 128 × 128 pixels
- **License:** CC BY 4.0

📎 Dataset Link:  
https://doi.org/10.17632/dfztdtfsxz.1

⚠️ **Note:**  
Due to size and license constraints, the dataset is **not uploaded** to this repository. Please download it from the official source and organize it as described below.

---

## 📂 Dataset Folder Structure
```text
dataset/
├── train/
│ ├── Monkeypox/
│ ├── Chickenpox/
│ ├── Cowpox/
│ ├── HFMD/
│ ├── Measles/
│ └── Healthy/
│
├── val/
└── test/
```
---


## 🧠 Model Architecture

- Backbone Network: **ResNet50 (Pretrained)**
- Feature Extraction: Convolutional Neural Network (CNN)
- Classification Head: Transformer Encoder
- Output Layer: Softmax (Multi-class Classification)
- Loss Function: Cross Entropy Loss
- Optimizer: Adam / AdamW
- Learning Rate Scheduler: Cosine Annealing

The hybrid architecture helps in capturing both **local lesion features** and **global contextual information**.

---

## ⚙️ Technologies Used

- Python  
- PyTorch  
- Albumentations  
- OpenCV  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Optuna  
- Streamlit  

---

## 🧪 Experimental Setup

- Image preprocessing: resizing, normalization, enhancement
- Data augmentation: rotation, flipping, brightness adjustment
- Train–Validation–Test split
- Early stopping and learning rate scheduling
- Evaluation on unseen test data

---

## 📈 Evaluation Metrics

The model performance is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC–AUC (Per Class & Macro Average)  
- Precision–Recall Curves  
- Confusion Matrix  
- Cohen’s Kappa  
- Matthews Correlation Coefficient (MCC)  

---

## 🔍 Model Interpretability

To understand model decisions, the following techniques are used:

- **Grad-CAM**: Highlights important skin lesion regions
- **t-SNE**: Visualizes feature space clustering
- **Misclassified Samples Analysis**: Identifies error patterns

All evaluation outputs are stored in the `outputs/evaluation_results/` directory.

---

## 🌐 Streamlit Web Application

The trained model is deployed using **Streamlit**, which provides:

- Image upload functionality
- Image preview
- Disease prediction output
- Confidence score display

📸 Screenshots of the web interface are available in:
```text
outputs/streamlit_screenshots/
```

---

## 🏗️ Project Structure
```text
monkeypox-disease-classification-using-deep-learning/
│
├── code/
│ ├── best_model.pth
│ ├── project.ipynb
│ ├── streamlit_app.py
│
├── streamlit_app/
│ └── app.py
│
├── dataset/
│ └── README.md
│
├── outputs/
│ ├── evaluation_results/
│ └── streamlit_screenshots/
│
├── requirements.txt
├── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yogeswaran-apparaj/monkeypox-disease-classification-using-deep-learning.git
cd monkeypox-disease-classification-using-deep-learning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run Streamlit App
```bash
streamlit run streamlit_app/app.py
```
---

## ⚠️ Disclaimer

This project is developed only for academic and educational purposes.
It is not intended for clinical diagnosis, treatment, or medical decision-making.

--- 

## 👨‍🎓 Author
#### YOGESWARAN APPARAJ
B.Tech – Artificial Intelligence & Data Science


--- 

## 📜 License

This project is released for academic use only.
Dataset license follows Creative Commons CC BY 4.0.



