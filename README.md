# 🌱 TerrAgro: Soil Classification Using Machine Learning
### **Case Study 139 | Machine Learning Fundamentals & Final Project Evaluation**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6+-orange.svg)](https://scikit-learn.org/)
[![Top Model Accuracy](https://img.shields.io/badge/Top%20Model%20Accuracy-99.29%25-brightgreen.svg)]()
[![5-Fold CV](https://img.shields.io/badge/5--Fold%20CV-98.54%25%20%C2%B1%200.86%25-success.svg)]()
[![Case Study](https://img.shields.io/badge/Case%20Study-139%20(Soil%20Classification)-purple.svg)]()

> **Final Project Title:** *TerrAgro: An Intelligent Soil & Agronomic Intelligence System Using Supervised Machine Learning*  
> **Course:** Machine Learning Fundamentals (CS401 / ML202)  
> **Assigned Problem Domain:** Case Study 139: Soil Classification Using Machine Learning  

---

## 📌 Executive Summary & Quick Navigation
**TerrAgro** is an end-to-end Machine Learning precision agriculture platform that predicts optimal soil taxonomic classification and agronomic crop suitability from 9 fundamental geochemical and physical indicators: **pH, Nitrogen (N), Phosphorus (P), Potassium (K), Moisture (%), Organic Matter (%), Electrical Conductivity (EC), Temperature (°C), and Humidity (%)** across **8 distinct soil categories**.

| Resource | File Link | Description |
| :--- | :--- | :--- |
| 📝 **Academic Project Report** | [`docs/markdown/PROJECT_REPORT.md`](file:///Users/rayhaansiledar/Desktop/ML%20Final%20Project/docs/markdown/PROJECT_REPORT.md) / [PDF](file:///Users/rayhaansiledar/Desktop/ML%20Final%20Project/docs/pdf/PROJECT_REPORT.pdf) | Full 16-chapter technical documentation formatted for 100-mark evaluation. |
| 🎓 **Viva Examination Guide** | [`docs/markdown/VIVA_PREPARATION_GUIDE.md`](file:///Users/rayhaansiledar/Desktop/ML%20Final%20Project/docs/markdown/VIVA_PREPARATION_GUIDE.md) / [PDF](file:///Users/rayhaansiledar/Desktop/ML%20Final%20Project/docs/pdf/VIVA_PREPARATION_GUIDE.pdf) | 25 high-yield viva questions, formulas, examiner traps, and oral strategy. |
| 📘 **ML Topics Guide** | [`docs/markdown/ML_TOPICS_EXPLAINED.md`](file:///Users/rayhaansiledar/Desktop/ML%20Final%20Project/docs/markdown/ML_TOPICS_EXPLAINED.md) / [PDF](file:///Users/rayhaansiledar/Desktop/ML%20Final%20Project/docs/pdf/ML_TOPICS_EXPLAINED.pdf) | Plain-English explanation of all ML concepts (HOW, WHY, WHEN). |
| 📁 **Project Files Guide** | [`docs/markdown/PROJECT_FILES_EXPLAINED.md`](file:///Users/rayhaansiledar/Desktop/ML%20Final%20Project/docs/markdown/PROJECT_FILES_EXPLAINED.md) / [PDF](file:///Users/rayhaansiledar/Desktop/ML%20Final%20Project/docs/pdf/PROJECT_FILES_EXPLAINED.pdf) | Comprehensive file-by-file breakdown. |
| 🧠 **ML Models Guide** | [`docs/markdown/ML_MODELS_EXPLAINED.md`](file:///Users/rayhaansiledar/Desktop/ML%20Final%20Project/docs/markdown/ML_MODELS_EXPLAINED.md) / [PDF](file:///Users/rayhaansiledar/Desktop/ML%20Final%20Project/docs/pdf/ML_MODELS_EXPLAINED.pdf) | Deep dive into the 5 ML models (Logistic Regression, KNN, DT, RF, GB). |
| 📓 **Jupyter Notebook** | [`Soil_Classification_Case_Study_139.ipynb`](file:///Users/rayhaansiledar/Desktop/ML%20Final%20Project/Soil_Classification_Case_Study_139.ipynb) | Complete, executed interactive Jupyter notebook with inline plots. |
| 🚀 **Streamlit Web Application** | [`app.py`](file:///Users/rayhaansiledar/Desktop/ML%20Final%20Project/app.py) | Interactive web dashboard running locally at `http://localhost:8503`. |
| 🤖 **Training Pipeline Script** | [`train_and_evaluate.py`](file:///Users/rayhaansiledar/Desktop/ML%20Final%20Project/train_and_evaluate.py) | Reproducible script generating models, EDA plots, and benchmark CSVs. |
| 📂 **Dataset Directory** | [`dataset/`](file:///Users/rayhaansiledar/Desktop/ML%20Final%20Project/dataset) | 2,800 balanced soil observations and batch upload test templates. |

---

## 🔄 Project Architecture & Workflow

```
[ Problem Formulation & Case Study 139 Objectives ]
                       │
                       ▼
[ Dataset Ingestion: 2,800 Observations across 8 Soil Orders ]
                       │
                       ▼
[ Exploratory Data Analysis: Correlation Heatmap, Boxplots, NPK Distributions ]
                       │
                       ▼
[ Preprocessing Impact Benchmark: Raw vs. StandardScaler Normalized ]
                       │
                       ▼
[ Model Training & 5-Fold Stratified CV: LR, KNN, DT, RF, GB ]
                       │
                       ▼
[ Comparative Evaluation: Accuracy, Macro F1, Recall, Confusion Matrix ]
                       │
                       ▼
[ Model Serialization (joblib) & Feature Importance Analysis ]
                       │
                       ▼
[ Streamlit Deployment: Live Classifier, Batch CSV, Advisory Hub, Viva Hub ]
```

---

## 🏆 Quantitative Model Benchmark Leaderboard

| Machine Learning Algorithm | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score | 5-Fold Stratified CV | Preprocessing Applied |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Logistic Regression** | **99.29%** | **99.29%** | **99.29%** | **99.29%** | **99.29%** | **98.54 ± 0.86%** | `StandardScaler` |
| 🥈 **Random Forest Classifier** | **99.11%** | **99.11%** | **99.11%** | **99.11%** | **99.11%** | **98.50 ± 0.58%** | None (Scale Invariant) |
| 🥉 **Gradient Boosting Classifier** | **98.39%** | **98.44%** | **98.39%** | **98.40%** | **98.40%** | **98.36 ± 0.51%** | None (Scale Invariant) |
| 4️⃣ **K-Nearest Neighbors (KNN)** | **98.04%** | **98.06%** | **98.04%** | **98.03%** | **98.03%** | **98.11 ± 0.54%** | `StandardScaler` (+5.54% gain) |
| 5️⃣ **Decision Tree Classifier** | **95.89%** | **95.91%** | **95.89%** | **95.88%** | **95.88%** | **95.68 ± 0.63%** | None (Scale Invariant) |

---

## 🌟 Key Research Questions Answered (Case Study 139)

1. **Can soil type be automatically classified?**  
   **Yes (> 99% accuracy).** Machine learning models readily capture geochemical and physical soil fingerprints across 8 orders.
2. **Which soil parameter contributes most to classification?**  
   **Electrical Conductivity (18.84%)** and **Organic Matter (17.56%)**, followed by **Moisture (16.10%)** and **pH (10.14%)**.
3. **Which algorithm provides the highest F1-score?**  
   **Logistic Regression (99.29%)** and **Random Forest (99.11%)**.
4. **Which soil categories are frequently confused?**  
   **Sandy Loam vs. Red Soil** (minor overlap due to shared low moisture and low organic matter).
5. **Does preprocessing improve classification?**  
   **Yes!** KNN gained **+5.54% accuracy** (from 92.50% to 98.04%), and Logistic Regression achieved smooth gradient convergence.
6. **Can the model classify a completely new soil sample?**  
   **Yes.** Serialized pipeline processes arbitrary raw test inputs in real time with confidence estimates.

---

## 🚀 Installation & Local Execution

### 1. Clone & Navigate to the Project Directory
```bash
cd "/Users/rayhaansiledar/Desktop/ML Final Project"
```

### 2. Install Dependencies
```bash
python3 -m pip install -r requirements.txt
```

### 3. Run the Training Pipeline (Optional - Pretrained models already included)
```bash
python3 train_and_evaluate.py
```

### 4. Launch the Interactive Streamlit Web Application
```bash
streamlit run app.py --server.port 8503
```
Open your browser at `http://localhost:8503`.

---

## 📂 Organized Project Structure

```
.
├── app.py                               # Main Streamlit Web Application
├── train_and_evaluate.py                # Master Training & Benchmark Pipeline
├── Soil_Classification_Case_Study_139.ipynb # Interactive Jupyter Notebook
├── requirements.txt                     # Dependencies file
├── README.md                            # Main project overview documentation
│
├── docs/                                # 📚 All Documentation & PDFs
│   ├── markdown/                        # Markdown source files
│   │   ├── PROJECT_REPORT.md            # Formal 16-Chapter Academic Report
│   │   ├── VIVA_PREPARATION_GUIDE.md    # 25+ Viva Voce Q&As
│   │   ├── ML_TOPICS_EXPLAINED.md       # ML Concepts (HOW, WHY, WHEN)
│   │   ├── PROJECT_FILES_EXPLAINED.md   # File-by-File Breakdown
│   │   └── ML_MODELS_EXPLAINED.md       # 5 ML Models Deep Dive
│   └── pdf/                             # Exported Publication-Grade PDFs
│       ├── PROJECT_REPORT.pdf
│       ├── VIVA_PREPARATION_GUIDE.pdf
│       ├── ML_TOPICS_EXPLAINED.pdf
│       ├── PROJECT_FILES_EXPLAINED.pdf
│       ├── ML_MODELS_EXPLAINED.pdf
│       └── README.pdf
│
├── dataset/                             # 📂 Raw & Processed Soil Data
│   ├── soil_classification.csv          # 2,800 Multi-Parameter Observations
│   ├── online_raw_soil_measures.csv     # Live downloaded online agricultural dataset
│   └── test_samples.csv                 # 25-Sample Test Batch CSV Template
│
├── models/                              # 💾 Serialized Models & Benchmarks
│   ├── logistic_regression.pkl          # Trained Logistic Regression
│   ├── knn.pkl                          # Trained KNN Model
│   ├── decision_tree.pkl                # Trained Decision Tree Model
│   ├── random_forest.pkl                # Trained Random Forest Ensemble
│   ├── gradient_boosting.pkl            # Trained Gradient Boosting Model
│   ├── scaler.pkl                       # Trained StandardScaler
│   ├── label_encoder.pkl                # LabelEncoder mapping classes
│   ├── model_metadata.json              # Full project metadata & metrics
│   ├── benchmark_comparison.csv         # Comparative leaderboard CSV
│   ├── preprocessing_impact.csv         # Preprocessing impact data
│   ├── feature_importance.csv           # Gini & Boosting feature weights
│   └── confused_soil_pairs.csv          # Error analysis class pairs
│
├── eda_plots/                           # 🎨 7 High-resolution EDA charts
├── model_metrics/                       # 📊 6 Evaluation & benchmark plots
└── scripts/                             # 🛠️ Helper & Utility Scripts
    └── convert_docs_to_pdf.py           # Markdown-to-PDF batch converter
```

---

## 👥 Authors & Academic Credits
- **Project Title:** Soil Classification Using Machine Learning (Case Study 139)
- **Academic Program:** B.Tech Computer Science & Engineering / Data Science & AI
- **Evaluation Criteria:** 100-Mark Project & Viva Voce Examination
