# Movie Recommendation System with AutoInt and AutoInt+MLP

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Public%20Domain-green.svg)](#license)

> **AutoInt** (Automatic Feature Interaction Learning)와 **AutoInt+MLP** 모델을 활용한 영화 추천 시스템

MovieLens 1M 데이터셋을 기반으로 사용자의 과거 시청 이력과 평점 데이터를 분석하여 개인화된 영화 추천을 제공합니다.

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/AutoInt_MLP.git
cd AutoInt_MLP

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run movie_rec_app.py
```

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Training Configuration](#training-configuration)
- [Evaluation Results](#evaluation-results)
- [Web Application](#web-application)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Overview

본 프로젝트는 **Multi-Head Self-Attention** 메커니즘을 활용한 추천 시스템입니다.

### 주요 특징

- **AutoInt 모델**: Self-Attention 기반 feature interaction 자동 학습
- **AutoInt+MLP 모델**: DNN 레이어를 결합한 하이브리드 모델
- **실시간 추천**: 사용자별 Top-10 영화 추천
- **웹 인터페이스**: Streamlit 기반 대화형 UI

---

## System Architecture

### 1. Model Architecture

#### 1.1 AutoInt Model

Multi-Head Self-Attention 메커니즘을 활용하여 feature 간의 고차원 상호작용을 자동으로 학습합니다.

**Key Components:**

```
Input Features
    ↓
Features Embedding Layer (dim=16)
    ↓
Multi-Head Self-Attention × 3 layers
  - Attention heads: 2
  - Residual connections
    ↓
Flatten & Dense Layer
    ↓
Sigmoid Activation
    ↓
CTR Prediction
```

#### 1.2 AutoInt+MLP Model

AutoInt 구조에 Deep Neural Network를 결합한 하이브리드 모델입니다.

**Key Components:**

```
                Input Features
                      ↓
              Embedding Layer
                   /    \
                  /      \
        AutoInt Path    DNN Path
         (Attention)    (32→32)
                  \      /
                   \    /
              Fusion Layer
                    ↓
            Sigmoid Activation
                    ↓
            CTR Prediction
```

**DNN Branch Specifications:**
- Hidden units: (32, 32)
- Activation: ReLU
- Dropout rate: 0.4
- Batch Normalization: Optional

### 2. Data Pipeline

```
Raw Data (MovieLens 1M)
         ↓
Data Preprocessing
         ↓
Feature Engineering
         ↓
Label Encoding
         ↓
Train/Test Split (80:20)
         ↓
Model Training
         ↓
Evaluation & Inference
```

---

## Dataset

### MovieLens 1M Dataset

| Metric | Value |
|--------|-------|
| Users | 6,040명 |
| Movies | 3,706편 |
| Ratings | 1,000,209개 |
| Rating Scale | 1-5 (정수) |
| Time Period | 2000-2003 |

### Feature Schema

| Feature | Type | Description | Cardinality |
|---------|------|-------------|-------------|
| `user_id` | Categorical | 사용자 식별자 | 6,040 |
| `movie_id` | Categorical | 영화 식별자 | 3,706 |
| `rating_year` | Categorical | 평점 부여 연도 | 4 |
| `rating_month` | Categorical | 평점 부여 월 | 12 |
| `rating_decade` | Categorical | 평점 부여 연대 | 1 |
| `movie_decade` | Categorical | 영화 제작 연대 | 10 |
| `movie_year` | Categorical | 영화 제작 연도 | 81 |
| `genre1`, `genre2`, `genre3` | Categorical | 영화 장르 (최대 3개) | 18 |
| `gender` | Categorical | 사용자 성별 | 2 |
| `age` | Categorical | 사용자 연령대 | 7 |
| `occupation` | Categorical | 사용자 직업 | 21 |
| `zip` | Categorical | 사용자 우편번호 | 3,439 |

**Total Field Dimensions**: `[6040, 3706, 10, 81, 4, 12, 1, 18, 18, 16, 2, 7, 21, 3439]`

---

## Training Configuration

### Hyperparameters

```python
# Model Parameters
embedding_dim = 16
att_layer_num = 3
att_head_num = 2
att_res = True
dnn_hidden_units = (32, 32)
dnn_activation = 'relu'
dnn_dropout = 0.4

# Training Parameters
epochs = 5
batch_size = 2048
learning_rate = 0.0001
optimizer = Adam
loss_function = BinaryCrossentropy

# Regularization
l2_reg_dnn = 0
l2_reg_embedding = 1e-5
```

### Training Results

#### AutoInt Model

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1/5 | 0.6813 | 0.6505 |
| 2/5 | 0.6221 | 0.5944 |
| 3/5 | 0.5707 | 0.5543 |
| 4/5 | 0.5487 | 0.5467 |
| 5/5 | 0.5430 | 0.5446 |

**Loss Reduction:**
- Training loss: `0.6813 → 0.5430` (**20.3% improvement**)
- Validation loss: `0.6505 → 0.5446` (**16.3% improvement**)

#### AutoInt+MLP Model

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1/5 | 0.6760 | 0.6468 |
| 2/5 | 0.6180 | 0.5896 |
| 3/5 | 0.5660 | 0.5500 |
| 4/5 | 0.5434 | 0.5435 |
| 5/5 | 0.5377 | 0.5411 |

**Loss Reduction:**
- Training loss: `0.6760 → 0.5377` (**20.5% improvement**)
- Validation loss: `0.6468 → 0.5411` (**16.3% improvement**)

---

## Evaluation Results

### Performance Metrics

| Model | NDCG@10 | Hit Rate@10 |
|-------|---------|-------------|
| AutoInt | 0.66201 | 0.63026 |
| AutoInt+MLP | 0.66196 | 0.63058 |

### Performance Analysis

- **NDCG** (Normalized Discounted Cumulative Gain): 두 모델 모두 약 **0.662**로 거의 동일
- **Hit Rate**: AutoInt+MLP가 **0.00032** (0.05%) 더 높으나 통계적으로 유의미한 차이는 아님
- 두 모델의 성능이 **실질적으로 동등**하며, 작업 특성에 따라 선택 가능

---

## Web Application

### Streamlit Interface

#### 주요 기능

1. **사용자 정보 입력**
   - 사용자 ID 직접 입력
   - 추천 타겟 연도/월 선택
   - 모델 선택 (AutoInt / AutoInt+MLP / 두 모델 비교)

2. **사용자 프로필 표시**
   - 성별, 나이, 직업, 지역 정보

3. **과거 시청 이력**
   - 평점 4점 이상 영화 목록
   - 영화 제목, 장르, 평점, 시청 시간

4. **추천 결과**
   - Top-10 영화 추천
   - 영화 ID, 제목, 장르 정보
   - 두 모델 비교 시 나란히 표시

### Application Screenshots

#### 실행 예시 1
- **사용자 ID**: 3
- **연도**: 2001, **월**: 5
- **사용자 정보**: M, 25세, 직업 15, 지역 55117
- **선호 영화 (9개)**: Animal House, Raising Arizona, Happy Gilmore 등 코미디 장르 선호
- **AutoInt+MLP 추천**: M, Cape Fear, Terror in a Texas Town 등 드라마/스릴러 장르 10개

#### 실행 예시 2
- **사용자 ID**: 2
- **연도**: 2000, **월**: 5
- **사용자 정보**: M, 56세, 직업 16, 지역 70072
- **선호 영화 (73개)**: Shine, Verdict 등 드라마 장르 다수
- **AutoInt 추천**: Umbrellas of Cherbourg, Aparajito, Murder My Sweet 등 클래식/드라마 10개

---

## Installation & Usage

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-username/AutoInt_MLP.git
cd AutoInt_MLP

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preprocessing (First time only)

```bash
# Run notebook
jupyter notebook notebook/data_prepro.ipynb
```

**Input:**
- `data/ml-1m/*.dat` (raw data)

**Output:**
- `data/ml-1m/*_prepro.csv` (preprocessed data)
- `data/ml-1m/movielens_rcmm_v2.csv` (integrated data)

### 3. Model Training (First time only)

#### Option A: Train AutoInt

```bash
jupyter notebook notebook/autoint_train.ipynb
```

#### Option B: Train AutoInt+MLP

```bash
jupyter notebook notebook/autoint_mlp_train.ipynb
```

**Output:**
- `model/autoInt_model_weights.weights.h5`
- `model/autoIntMLP_model_weights.weights.h5`
- `data/field_dims.npy`
- `model/label_encoders.pkl`

### 4. Run Application

```bash
streamlit run movie_rec_app.py
```

**Required Files:**
- `data/field_dims.npy`
- `data/label_encoders.pkl`
- `data/ml-1m/*_prepro.csv`
- `model/autoInt_model_weights.weights.h5`
- `model/autoIntMLP_model_weights.weights.h5`
- `autoint.py`
- `autointmlp.py`

---

## Project Structure

```
AutoInt_MLP/
├── data/
│   ├── field_dims.npy              # Feature dimension info
│   ├── label_encoders.pkl          # Fitted label encoders
│   └── ml-1m/
│       ├── users.dat               # Raw user data
│       ├── movies.dat              # Raw movie data
│       ├── ratings.dat             # Raw rating data
│       ├── *_prepro.csv            # Preprocessed data
│       └── movielens_rcmm_v*.csv   # Integrated data
│
├── model/
│   ├── autoInt_model_weights.weights.h5
│   ├── autoIntMLP_model_weights.weights.h5
│   └── label_encoders.pkl
│
├── notebook/
│   ├── data_EDA.ipynb              # Exploratory data analysis
│   ├── data_prepro.ipynb           # Data preprocessing
│   ├── autoint_train.ipynb         # AutoInt training
│   ├── autoint_mlp_train.ipynb     # AutoInt+MLP training
│   └── model_load_test.ipynb       # Model testing
│
├── autoint.py                       # AutoInt implementation
├── autointmlp.py                    # AutoInt+MLP implementation
├── movie_rec_app.py                 # Main Streamlit app
├── show_st*.py                      # App variations
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## Key Findings

### 1. Model Performance

- 두 모델 모두 **NDCG@10 약 0.662**, **Hit Rate@10 약 0.63**으로 우수한 성능
- AutoInt+MLP의 추가 DNN layer가 성능 향상에 **유의미한 영향을 주지 않음**
- 데이터셋 특성상 **attention mechanism만으로도 충분한** feature interaction 학습 가능

### 2. Training Stability

- **5 epoch 내에 안정적인 수렴**
- Validation loss가 **epoch 4부터 plateau** 도달
- **Overfitting 징후 없음** (train/val loss 차이 < 0.02)

### 3. Inference Efficiency

- **Batch prediction (2048)** 활용으로 효율적인 추론
- 6,000명 사용자에 대한 전체 추천 생성 시간 **< 10초**
- **Real-time 추천 가능**한 수준의 latency

---

## Future Work

### Model Enhancement
- Extended AutoInt (XDeepFM) 적용
- Attention mechanism variant 실험 (sparse attention, local attention)
- Multi-task learning (rating prediction + ranking)

### Feature Engineering
- User/Item embedding pre-training (Word2Vec, BERT4Rec)
- Temporal features 확장 (time of day, day of week)
- Social features (collaborative filtering signals)

### System Optimization
- Model quantization for faster inference
- Distributed training for larger datasets
- A/B testing framework 구축

### Production Deployment
- Docker containerization
- REST API development (FastAPI)
- Model serving with TensorFlow Serving
- Monitoring and logging system

---

## References

1. Song, W., Shi, C., Xiao, Z., Duan, Z., Xu, Y., Zhang, M., & Tang, J. (2019). **AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks**. *CIKM 2019*.

2. Harper, F. M., & Konstan, J. A. (2015). **The MovieLens Datasets: History and Context**. *ACM Transactions on Interactive Intelligent Systems*, 5(4), 1-19.

3. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). **Attention is all you need**. *NeurIPS 2017*.

---

## License

This project is released into the **public domain**.

You can freely use, modify, and distribute this code for any purpose, including commercial use, without any restrictions or attribution requirements.

### 한국어
이 프로젝트는 **퍼블릭 도메인**으로 공개됩니다.

상업적 용도를 포함하여 어떤 목적으로든 자유롭게 사용, 수정, 배포할 수 있으며, 별도의 제약이나 출처 표기 요구사항이 없습니다.

---

## Contact

For questions or collaboration inquiries, please contact:

**Email**: soflywithai@gmail.com

---

## Note

본 프로젝트는 **학술적 목적**으로 개발되었으며, MovieLens 데이터셋 사용 정책을 준수합니다.

---

<div align="center">

**If you find this project useful, please consider giving it a star!**

Made with love by [Your Name]

</div>
