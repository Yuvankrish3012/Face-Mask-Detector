# Face-Mask-Detector

# 😷 Face Mask Detection using Deep Learning

This project aims to detect whether a person is:
- ✅ Wearing a mask correctly  
- ❌ Not wearing a mask  
- ⚠️ Wearing a mask incorrectly  

using **Convolutional Neural Networks (CNN)** with real-time detection capabilities.

---

## 📁 Dataset

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- **Classes**:
  1. With mask 😷  
  2. Without mask 😮  
  3. Mask worn incorrectly 😕

- **Format**:
  - Images: `images/`
  - Bounding Boxes: `annotations/` in **Pascal VOC format**
- **Total images**: 853

---

## 🔍 Exploratory Data Analysis (EDA)

Before building the model, we performed visual and statistical exploration of the dataset.

### 1️⃣ Sample Distribution per Class  

![image](https://github.com/user-attachments/assets/5fefe22f-766f-4664-8c76-718371ea4013)

### 2️⃣ Sample Images from Each Class  

![image](https://github.com/user-attachments/assets/6e409978-dc4e-42c5-a2c4-024c04e13a59)
![image](https://github.com/user-attachments/assets/367a191b-f9d5-4624-afe6-51077b6c8a8f)
![image](https://github.com/user-attachments/assets/d381fdf7-16ff-4ec3-9a60-cac472d07a3b)
![image](https://github.com/user-attachments/assets/90e84d71-b531-4db1-9a07-272ac2012ced)

![image](https://github.com/user-attachments/assets/594db93b-f29b-4356-b30c-34cc580b51c4)


### 3️⃣ Image Dimension Stats  

![image](https://github.com/user-attachments/assets/d50e64f5-6a96-446f-9c93-ebfb74630bbd)

---

## 🧠 Model Architecture

We used a **custom CNN model** with the following layers:

```text
Input -> Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Conv2D(128) -> Flatten -> Dense -> Output(3)
```

- 3 Convolutional layers (with ReLU, MaxPooling)
- 1 Flatten layer
- 2 Dense layers (with dropout)
- Final Softmax for 3-class output

---

## 🏋️ Training Summary

- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  
- **Epochs**: 10  
- **Train/Val Split**: 80/20

### 📈 Accuracy & F1-Score

```
✅ Accuracy: 0.95
✅ F1 Score: 0.94
```

### 📊 Classification Report

| Class               | Precision | Recall | F1-Score |
|---------------------|-----------|--------|----------|
| Incorrect Mask (0)  | 0.86      | 0.24   | 0.38     |
| With Mask (1)       | 0.97      | 0.97   | 0.97     |
| Without Mask (2)    | 0.86      | 0.95   | 0.90     |

---

## 💾 Saved Artifacts

- `mask_detector_model.h5` → Trained CNN Model  
- `label_encoder.pkl` → Encodes class labels  
- `app.py` → Streamlit UI

---

## 🌐 Streamlit Frontend

We built a **beautiful and responsive UI** using Streamlit.

### 🎨 Features:
- Upload any image
- Detect all visible faces
- Show bounding boxes and mask status with confidence %
- Color-coded detection:  
  - 🟢 With Mask  
  - 🔴 Without Mask  
  - 🟡 Mask Incorrect

📌 **Placeholder**: _Insert screenshot of Streamlit UI here_

### ▶️ Run App

```bash
streamlit run "D:\ML PROJECTS\Face Mask Detection\app.py"
```

---

## 📌 Folder Structure

```
📁 Face Mask Detection
│
├── archive/
│   ├── annotations/
│   └── images/
├── app.py
├── mask_detector_model.h5
├── label_encoder.pkl
└── README.md
```

---

## 📚 Libraries Used

- Python, TensorFlow / Keras  
- OpenCV, Streamlit  
- Seaborn / Matplotlib  
- scikit-learn  

---

## 🚀 Future Improvements

- Improve recall for `Incorrect Mask` class  
- Real-time webcam detection  
- Deploy to HuggingFace or Streamlit Cloud  
