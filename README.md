# 🧠 EEG-SchizoDetect: Intelligent Diagnosis of Schizophrenia via Multivariate Entropy and Machine Learning

> _Reimagining psychiatric diagnosis through entropy, EEG signals, and powerful AI models._

## 📌 Abstract

**EEG-SchizoDetect** presents a novel diagnostic framework for schizophrenia based on multichannel EEG analysis, entropy measures, and cutting-edge machine learning. Our system leverages entropy's sensitivity to neural complexity and integrates five powerful classifiers to identify schizophrenia with remarkable accuracy — potentially redefining clinical approaches.

## 🚀 Highlights

- ✅ **97.39% Accuracy** using Ensemble Deep RVFL on 16 EEG channels  
- 🔍 Multi-lobe entropy profiling across **frontal, temporal, parietal, occipital** regions  
- 🧠 Signal complexity decoded via **Shannon, Sample, Approximate, Spectral, Tsallis entropy**  
- 🤖 Models tested: `GB`, `ELM`, `TELM`, `RVFL`, `edRVFL`  
- 📊 Evaluated under k-fold cross-validation (5–10 folds)

## 🎯 Research Goals

- Detect schizophrenia using non-invasive EEG signals  
- Assess how brain regions (via channel configurations) affect model accuracy  
- Benchmark multiple models to find optimal accuracy/performance trade-offs  
- Build a reproducible ML pipeline for early diagnosis in real-world scenarios


## 🚀 Pipeline Overview

```mermaid
flowchart TD
    A[📂 EEG .edf Data] --> B[🔧 Preprocessing & Epoching]
    B --> C[📊 Entropy Feature Extraction]
    C --> D[🧮 Feature Vector Creation]
    D --> E[🤖 ML Model Training & Testing]
## 🧠 Feature Engineering

| Feature             | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **Shannon Entropy** | Quantifies signal unpredictability                                           |
| **Sample Entropy**  | Measures signal complexity over time                                         |
| **Approx. Entropy** | Detects signal irregularity, suitable for short, noisy EEG sequences         |
| **Spectral Entropy**| Assesses energy distribution across EEG frequencies                         |
| **Tsallis Entropy** | Generalized entropy measure for non-linear, long-range dependent signals     |

## 🧪 Experimental Configuration

- Dataset: **RepOD EEG**, 28 subjects, 19 electrodes  
- Epoch Length: **2 seconds**, 500 samples per epoch  
- Preprocessing: Artifact filtering, epoch selection, normalization  
- Tools: `MNE`, `pyEDFlib`, `NumPy`, `scikit-learn`, `TensorFlow`, `Pandas`

## 💻 Model Benchmarking

| Model     | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| Gradient Boosting (GB)        | 97.00%   | 0.98      | 0.96   | 0.97     |
| Extreme Learning Machine (ELM)| 90.79%   | 0.91      | 0.90   | 0.91     |
| Twin ELM (TELM)               | 93.64%   | 0.96      | 0.91   | 0.93     |
| Random VFL (RVFL)            | 93.64%   | 0.96      | 0.91   | 0.93     |
| **Ensemble Deep RVFL**        | **97.39%**| **0.97**  | **0.97**| **0.97** |

📈 *Best results achieved on 16-channel configuration (frontal, temporal, parietal lobes).*

## 📁 Folder Structure

```bash
├── data/               # EEG .edf files
├── preprocessing/      # Epoching & cleaning scripts
├── features/           # Entropy feature extraction
├── models/             # Classifier training & comparison
├── evaluation/         # ROC, metrics, result tables
├── results/            # Saved models, logs, charts
└── README.md           # This file

@project{eegschizodetect2025,
  title={Multivariate EEG Analysis for Schizophrenia Detection: Leveraging Entropy Measures and Machine Learning Techniques across Channel Configurations},
  author={Goyal, Tushar and Mishra, Utkarsh and Sharma, Vasant Kr and Shendre, Sayog},
  year={2025},
  institution={Motilal Nehru National Institute of Technology Allahabad}
}



---

✅ Copy the code above into a file named `README.md` in your GitHub project.

Let me know if you want to add badges, setup instructions, demo GIFs, or deploy it as a hosted web app.

