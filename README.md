## Movie Recommendation System with AutoInt and AutoInt+MLP

$\color{green}{\text{AutoInt, AutoInt+MLP}}$
$\color{green}{\text{streamlit run movie_rec_app.py}}$

[ 1 ]  Overview

본 프로젝트는 AutoInt(Automatic Feature Interaction Learning)와 AutoInt+MLP 모델을 활용한 영화 추천 시스템입니다. 
MovieLens 1M 데이터셋을 기반으로 사용자의 과거 시청 이력과 평점 데이터를 분석하여 개인화된 영화 추천을 제공합니다.

[ 2 ] System Architecture

1. Model Architecture
   
1.1 AutoInt Model
AutoInt는 Multi-Head Self-Attention 메커니즘을 활용하여 feature 간의 고차원 상호작용을 자동으로 학습하는 모델입니다.

Key Components:
Features Embedding Layer: 범주형 features를 dense embedding 벡터로 변환

Multi-Head Self-Attention Layers: 3개 층의 attention 메커니즘으로 feature 간 상호작용 학습

Attention heads: 2
Embedding dimension: 16
Residual connections 적용

Output Layer: Sigmoid activation을 통한 CTR(Click-Through Rate) 예측

1.2 AutoInt+MLP Model
AutoInt 구조에 Deep Neural Network를 결합하여 성능을 향상시킨 하이브리드 모델입니다.

Key Components:
AutoInt의 모든 구성요소 포함

추가 DNN Branch:
Hidden units: (32, 32)
Activation: ReLU
Dropout rate: 0.4
Batch Normalization (optional)

Fusion Layer: AutoInt 출력과 DNN 출력을 결합하여 최종 예측

2. Data Pipeline
   
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

Dataset
MovieLens 1M Dataset

Users: 6,040명
Movies: 3,706편
Ratings: 1,000,209개
Rating Scale: 1-5 (정수)
Time Period: 2000-2003

Feature Schema
Feature                    Type             Description            Cardinality  
user_id                    Categorical      사용자 식별자           6,040
movie_id                   Categorical      영화 식별자             3,706
rating_year                Categorical      평점 부여 연도          4  
rating_month               Categorical      평점 부여 월            12  
rating_decade              Categorical      평점 부여 연대          -
movie_decade               Categorical      영화 제작 연대          10    
movie_year                 Categorical      영화 제작 연도          81
genre1, genre2, genre3     Categorical      영화 장르 (최대 3개)    18
gender                     Categorical      사용자 성별             2
age                        Categorical      사용자 연령대           7occupationCategorical사용자 직업21zipCategorical사용자 우편번호3,439
Total Field Dimensions: [6040, 3706, 10, 81, 4, 12, 1, 18, 18, 16, 2, 7, 21, 3439]
Training Configuration
Hyperparameters
python# Model Parameters
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
```
Epoch 1/5: loss: 0.6813, val_loss: 0.6505
Epoch 2/5: loss: 0.6221, val_loss: 0.5944
Epoch 3/5: loss: 0.5707, val_loss: 0.5543
Epoch 4/5: loss: 0.5487, val_loss: 0.5467
Epoch 5/5: loss: 0.5430, val_loss: 0.5446
```

**Loss Reduction**: 
- Training loss: 0.6813 → 0.5430 (20.3% improvement)
- Validation loss: 0.6505 → 0.5446 (16.3% improvement)

#### AutoInt+MLP Model
```
Epoch 1/5: loss: 0.6760, val_loss: 0.6468
Epoch 2/5: loss: 0.6180, val_loss: 0.5896
Epoch 3/5: loss: 0.5660, val_loss: 0.5500
Epoch 4/5: loss: 0.5434, val_loss: 0.5435
Epoch 5/5: loss: 0.5377, val_loss: 0.5411
Loss Reduction:

Training loss: 0.6760 → 0.5377 (20.5% improvement)
Validation loss: 0.6468 → 0.5411 (16.3% improvement)

Evaluation Metrics
ModelNDCG@10Hit Rate@10AutoInt0.662010.63026AutoInt+MLP0.661960.63058
Performance Analysis:

NDCG (Normalized Discounted Cumulative Gain): 두 모델 모두 약 0.662로 거의 동일
Hit Rate: AutoInt+MLP가 0.00032 (0.05%) 더 높으나 통계적으로 유의미한 차이는 아님
두 모델의 성능이 실질적으로 동등하며, 작업 특성에 따라 선택 가능

Implementation Details
1. Data Preprocessing
python# Label Encoding for categorical features
label_encoders = {
    'user_id': LabelEncoder(),
    'movie_id': LabelEncoder(),
    'genre1': LabelEncoder(),
    # ... 기타 features
}

# Train/Test Split
train_size = 0.8
train_data = data[:int(len(data) * train_size)]
test_data = data[int(len(data) * train_size):]
2. Model Training
python# Model compilation
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=BinaryCrossentropy(),
    metrics=[BinaryAccuracy()]
)

# Model fitting
history = model.fit(
    X_train, y_train,
    batch_size=2048,
    epochs=5,
    validation_data=(X_val, y_val),
    verbose=1
)
3. Inference Pipeline
pythondef get_recommendations(user_id, year, month, model, top_k=10):
    # 1. 사용자가 시청하지 않은 영화 필터링
    unseen_movies = filter_unseen_movies(user_id)
    
    # 2. Feature 구성
    features = build_features(user_id, unseen_movies, year, month)
    
    # 3. 모델 예측
    predictions = model.predict(features, batch_size=2048)
    
    # 4. Top-K 추출
    top_k_movies = get_top_k(predictions, k=top_k)
    
    return top_k_movies
```

## Web Application

### Streamlit Interface

**주요 기능**:
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

**실행 예시 1**: 사용자 ID 3, 연도 2001, 월 5
- **사용자 정보**: M, 25세, 직업 15, 지역 55117
- **선호 영화 (9개)**: Animal House, Raising Arizona, Happy Gilmore 등 코미디 장르 선호
- **AutoInt+MLP 추천**: M, Cape Fear, Terror in a Texas Town 등 드라마/스릴러 장르 10개

**실행 예시 2**: 사용자 ID 2, 연도 2000, 월 5
- **사용자 정보**: M, 56세, 직업 16, 지역 70072
- **선호 영화 (73개)**: Shine, Verdict 등 드라마 장르 다수
- **AutoInt 추천**: Umbrellas of Cherbourg, Aparajito, Murder My Sweet 등 클래식/드라마 10개

## Project Structure
```
AutoInt_MLP/
├── 📁 data/
│   ├── field_dims.npy              # Feature dimension info
│   ├── label_encoders.pkl          # Fitted label encoders
│   └── ml-1m/
│       ├── users.dat               # Raw user data
│       ├── movies.dat              # Raw movie data
│       ├── ratings.dat             # Raw rating data
│       ├── *_prepro.csv           # Preprocessed data
│       └── movielens_rcmm_v*.csv  # Integrated data
│
├── 📁 model/
│   ├── autoInt_model_weights.weights.h5
│   ├── autoIntMLP_model_weights.weights.h5
│   └── label_encoders.pkl
│
├── 📁 notebook/
│   ├── data_EDA.ipynb             # Exploratory data analysis
│   ├── data_prepro.ipynb          # Data preprocessing
│   ├── autoint_train.ipynb        # AutoInt training
│   ├── autoint_mlp_train.ipynb    # AutoInt+MLP training
│   └── model_load_test.ipynb      # Model testing
│
├── autoint.py                      # AutoInt implementation
├── autointmlp.py                   # AutoInt+MLP implementation
├── movie_rec_app.py                # Main Streamlit app
├── show_st*.py                     # App variations
└── requirements.txt                # Dependencies
Installation & Usage
1. Environment Setup
bash# Clone repository
git clone https://github.com/your-username/AutoInt_MLP.git
cd AutoInt_MLP

# Install dependencies
pip install -r requirements.txt
2. Data Preprocessing (First time only)
bash# Run notebook/data_prepro.ipynb
jupyter notebook notebook/data_prepro.ipynb
Input:

data/ml-1m/*.dat (raw data)

Output:

data/ml-1m/*_prepro.csv (preprocessed data)
data/ml-1m/movielens_rcmm_v2.csv (integrated data)

3. Model Training (First time only)
Option A: Train AutoInt
bash# Run notebook/autoint_train.ipynb
jupyter notebook notebook/autoint_train.ipynb
Option B: Train AutoInt+MLP
bash# Run notebook/autoint_mlp_train.ipynb
jupyter notebook notebook/autoint_mlp_train.ipynb
Output:

model/autoInt_model_weights.weights.h5
model/autoIntMLP_model_weights.weights.h5
data/field_dims.npy
model/label_encoders.pkl

4. Run Application
bashstreamlit run movie_rec_app.py
Required Files:

✅ data/field_dims.npy
✅ data/label_encoders.pkl
✅ data/ml-1m/*_prepro.csv
✅ model/autoInt_model_weights.weights.h5
✅ model/autoIntMLP_model_weights.weights.h5
✅ autoint.py
✅ autointmlp.py

Key Findings
1. Model Performance

두 모델 모두 NDCG@10 약 0.662, Hit Rate@10 약 0.63으로 우수한 성능
AutoInt+MLP의 추가 DNN layer가 성능 향상에 유의미한 영향을 주지 않음
데이터셋 특성상 attention mechanism만으로도 충분한 feature interaction 학습 가능

2. Training Stability

5 epoch 내에 안정적인 수렴
Validation loss가 epoch 4부터 plateau 도달
Overfitting 징후 없음 (train/val loss 차이 < 0.02)

3. Inference Efficiency

Batch prediction (2048) 활용으로 효율적인 추론
6,000명 사용자에 대한 전체 추천 생성 시간 < 10초
Real-time 추천 가능한 수준의 latency

Future Work

Model Enhancement

Extended AutoInt (XDeepFM) 적용
Attention mechanism variant 실험 (sparse attention, local attention)
Multi-task learning (rating prediction + ranking)


Feature Engineering

User/Item embedding pre-training (Word2Vec, BERT4Rec)
Temporal features 확장 (time of day, day of week)
Social features (collaborative filtering signals)


System Optimization

Model quantization for faster inference
Distributed training for larger datasets
A/B testing framework 구축


Production Deployment

Docker containerization
REST API development (FastAPI)
Model serving with TensorFlow Serving
Monitoring and logging system



References

Song, W., Shi, C., Xiao, Z., Duan, Z., Xu, Y., Zhang, M., & Tang, J. (2019). AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks. CIKM 2019.
Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), 1-19.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. NeurIPS 2017.

License
This project is licensed und
