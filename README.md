```markdown
# 🖋️ Deep Learning Writer Identification

A deep learning-based writer identification system that classifies handwritten document images using segmented line patches and probability aggregation at page level.

---

## 🚀 Project Overview

This project implements a CNN-based writer identification model using **TensorFlow/Keras**.

Instead of directly classifying entire pages, the system follows a structured pipeline:

1. Segment each handwritten page into multiple line patches  
2. Preprocess patches (normalization + inversion)  
3. Predict writer probabilities for each patch  
4. Aggregate patch probabilities to generate final page-level prediction  

This approach improves robustness by leveraging local handwriting features.

---

## 🧠 Model Details

- **Framework:** TensorFlow (Mac M-series compatible)
- **Backend:** `tensorflow-macos` + `tensorflow-metal`
- **Patch size:** 128 × 256
- **Segments per line:** 8
- **Aggregation method:** Sum of patch probabilities
- **Multi-class classification**

---

## 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Macro F1 Score
- Macro AUC (One-vs-Rest)

### Example Results

```

Test Accuracy : 0.0643
Macro F1      : 0.0379
Macro AUC     : 0.8064

```

> Note: Low accuracy may indicate class imbalance or the need for improved architecture.

---

## 📁 Project Structure

```

deep-learning-writer-identification/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── models/
│   └── writer_model.keras
│
├── outputs/
│   └── result.csv
│
├── train.py
├── run.py
├── patching.py
├── utils.py
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ (Mac M-series Only)

```bash
pip install tensorflow-macos tensorflow-metal
```

---

## ▶️ Run Evaluation

```bash
python run.py
```

Results will be saved to:

```
outputs/result.csv
```

---

## 📌 Key Features

* Line-based patch extraction
* Patch-level probability aggregation
* Multi-class writer classification
* GPU acceleration using Apple Metal
* End-to-end evaluation pipeline

---

## 📈 Future Improvements

* Replace CNN with ResNet / EfficientNet
* Apply data augmentation
* Implement attention-based patch aggregation
* Explore Transformer-based handwriting modeling
* Improve class balancing strategies

---

## 👩‍💻 Author

**Khadiga Idris**
Computer Science (Data Science Track)
Albukhary International University

GitHub: [https://github.com/doji0x0](https://github.com/doji0x0)

```
```



