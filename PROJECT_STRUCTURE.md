# AutoInt_MLP 프로젝트 구조 및 파일 연결 관계

## 📁 폴더 구조 및 주요 역할

### 1. **루트 디렉토리** (`/`)
- **역할**: 프로젝트의 메인 디렉토리
- **주요 파일**:
  - `autoint.py`: AutoInt 모델 구현 (순수 어텐션 기반)
  - `autointmlp.py`: AutoInt+MLP 하이브리드 모델 구현
  - `movie_rec_app.py`: Streamlit 웹 애플리케이션 (메인 앱)
  - `show_st*.py`: Streamlit 앱의 다양한 버전들
  - `requirements.txt`: 프로젝트 의존성 패키지 목록

### 2. **`data/` 폴더**
- **역할**: 모든 데이터 파일 저장
- **구조**:
  ```
  data/
  ├── ml-1m/                    # MovieLens 1M 원본 데이터셋
  │   ├── users.dat            # 원본 사용자 데이터
  │   ├── movies.dat           # 원본 영화 데이터
  │   ├── ratings.dat          # 원본 평점 데이터
  │   ├── users_prepro.csv     # 전처리된 사용자 데이터
  │   ├── movies_prepro.csv    # 전처리된 영화 데이터
  │   ├── ratings_prepro.csv    # 전처리된 평점 데이터
  │   ├── movielens_rcmm_v1.csv # 추천용 통합 데이터 v1
  │   └── movielens_rcmm_v2.csv # 추천용 통합 데이터 v2
  ├── field_dims.npy           # 각 필드의 차원 정보 (임베딩용)
  ├── label_encoders.pkl       # 라벨 인코더 (전처리용)
  └── ml-1m.npy                # 전처리된 NumPy 배열
  ```

### 3. **`model/` 폴더**
- **역할**: 학습된 모델 가중치 및 구조 저장
- **구조**:
  ```
  model/
  ├── autoInt_model.keras              # AutoInt 모델 구조
  ├── autoInt_model_weights.weights.h5 # AutoInt 모델 가중치
  ├── autoIntMLP_model_weights.weights.h5 # AutoIntMLP 모델 가중치
  └── label_encoders.pkl               # 라벨 인코더 (백업)
  ```

### 4. **`notebook/` 폴더**
- **역할**: 데이터 분석, 전처리, 모델 학습을 위한 Jupyter 노트북
- **구조**:
  ```
  notebook/
  ├── data_EDA.ipynb          # 데이터 탐색적 분석
  ├── data_prepro.ipynb        # 데이터 전처리 파이프라인
  ├── autoint_train.ipynb      # AutoInt 모델 학습
  ├── autoint_mlp_train.ipynb  # AutoIntMLP 모델 학습
  └── model_load_test.ipynb    # 모델 로드 및 테스트
  ```

---

## 🔄 파일 간 연결 구조 및 데이터 흐름

### **Phase 1: 데이터 전처리 파이프라인**

```
data/ml-1m/
├── users.dat ──────┐
├── movies.dat ─────┼──> notebook/data_prepro.ipynb
└── ratings.dat ────┘
                            │
                            ▼
                    [전처리 작업]
                    - 연도/년대 추출
                    - 장르 분리
                    - 타임스탬프 변환
                    - 라벨 생성 (3점 이상 = 1)
                            │
                            ▼
data/ml-1m/
├── users_prepro.csv ────────┐
├── movies_prepro.csv ───────┼──> data/ml-1m/movielens_rcmm_v2.csv
└── ratings_prepro.csv ──────┘    (통합 추천 데이터)
```

### **Phase 2: 모델 학습 파이프라인**

#### **AutoInt 모델 학습** (`notebook/autoint_train.ipynb`)
```
data/ml-1m/movielens_rcmm_v2.csv
            │
            ▼
    [데이터 로드 및 인코딩]
            │
            ├──> LabelEncoder 생성
            │    (각 컬럼별 인코딩)
            │
            ▼
    [field_dims 계산]
    (각 필드의 최대값 + 1)
            │
            ├──> data/field_dims.npy ──────┐
            └──> model/label_encoders.pkl ─┤
                                            │
                                            ▼
                                    [모델 학습]
                                    autoint.py
                                    - FeaturesEmbedding
                                    - MultiHeadSelfAttention
                                    - AutoIntModel
                                            │
                                            ▼
                            model/autoInt_model_weights.weights.h5
```

#### **AutoIntMLP 모델 학습** (`notebook/autoint_mlp_train.ipynb`)
```
data/ml-1m/movielens_rcmm_v2.csv
            │
            ▼
    [동일한 전처리 과정]
            │
            ▼
    [모델 학습]
    autointmlp.py
    - FeaturesEmbedding
    - MultiHeadSelfAttention
    - MultiLayerPerceptron (DNN 추가)
    - AutoIntMLPModel
            │
            ▼
    model/autoIntMLP_model_weights.weights.h5
```

### **Phase 3: 애플리케이션 실행** (`movie_rec_app.py`)

```
[앱 시작]
    │
    ▼
load_data() 함수
    │
    ├──> data/field_dims.npy ──────────────┐
    ├──> data/label_encoders.pkl ──────────┤
    ├──> data/ml-1m/users_prepro.csv ──────┤
    ├──> data/ml-1m/movies_prepro.csv ─────┤
    ├──> data/ml-1m/ratings_prepro.csv ────┤
    │                                      │
    ├──> model/autoInt_model_weights.weights.h5 ──┐
    └──> model/autoIntMLP_model_weights.weights.h5 ┘
            │
            ▼
    [모델 인스턴스 생성]
    - AutoIntModel (autoint.py)
    - AutoIntMLPModel (autointmlp.py)
            │
            ▼
    [사용자 입력]
    - user_id
    - 추천 타겟 연도/월
    - 모델 선택
            │
            ▼
    [추천 생성]
    - 사용자가 보지 않은 영화 필터링
    - 피처 인코딩 (label_encoders 사용)
    - 모델 예측 (predict_model 함수)
            │
            ▼
    [결과 표시]
    - 상위 10개 영화 추천
```

---

## 📊 주요 파일별 역할

### **모델 구현 파일**

#### `autoint.py`
- **역할**: AutoInt 모델 구현
- **주요 클래스**:
  - `FeaturesEmbedding`: 피처 임베딩 레이어
  - `MultiHeadSelfAttention`: 멀티헤드 셀프 어텐션
  - `AutoInt`: AutoInt 레이어 (어텐션만 사용)
  - `AutoIntModel`: 완전한 AutoInt 모델
- **사용 위치**: 
  - `notebook/autoint_train.ipynb` (학습)
  - `movie_rec_app.py` (추론)

#### `autointmlp.py`
- **역할**: AutoInt+MLP 하이브리드 모델 구현
- **주요 클래스**:
  - `FeaturesEmbedding`: 피처 임베딩 레이어
  - `MultiHeadSelfAttention`: 멀티헤드 셀프 어텐션
  - `MultiLayerPerceptron`: DNN 레이어
  - `AutoIntMLP`: AutoInt + MLP 결합 레이어
  - `AutoIntMLPModel`: 완전한 AutoIntMLP 모델
- **사용 위치**:
  - `notebook/autoint_mlp_train.ipynb` (학습)
  - `movie_rec_app.py` (추론)

### **애플리케이션 파일**

#### `movie_rec_app.py` (메인 앱)
- **역할**: Streamlit 기반 웹 애플리케이션
- **주요 함수**:
  - `load_data()`: 데이터 및 모델 로드
  - `get_user_seen_movies()`: 사용자가 본 영화 목록
  - `get_recommendations()`: 영화 추천 생성
- **의존성**:
  - `autoint.py` → AutoIntModel, predict_model
  - `autointmlp.py` → AutoIntMLPModel, predict_model
  - `data/` 폴더의 모든 전처리된 데이터
  - `model/` 폴더의 학습된 모델 가중치

### **노트북 파일**

#### `notebook/data_prepro.ipynb`
- **역할**: 원본 데이터 전처리
- **입력**: `data/ml-1m/*.dat` 파일들
- **출력**: 
  - `data/ml-1m/*_prepro.csv` 파일들
  - `data/ml-1m/movielens_rcmm_v2.csv`

#### `notebook/autoint_train.ipynb`
- **역할**: AutoInt 모델 학습
- **입력**: 
  - `data/ml-1m/movielens_rcmm_v2.csv`
  - `autoint.py` (모델 정의)
- **출력**:
  - `data/field_dims.npy`
  - `model/label_encoders.pkl`
  - `model/autoInt_model_weights.weights.h5`

#### `notebook/autoint_mlp_train.ipynb`
- **역할**: AutoIntMLP 모델 학습
- **입력**:
  - `data/ml-1m/movielens_rcmm_v2.csv`
  - `autointmlp.py` (모델 정의)
- **출력**:
  - `model/autoIntMLP_model_weights.weights.h5`

#### `notebook/model_load_test.ipynb`
- **역할**: 모델 로드 및 테스트
- **입력**: 모든 학습된 모델 및 데이터 파일
- **목적**: 모델이 제대로 로드되는지 검증

---

## 🔗 핵심 데이터 흐름 요약

```
1. 원본 데이터 (users.dat, movies.dat, ratings.dat)
   ↓
2. 전처리 (data_prepro.ipynb)
   ↓
3. 통합 데이터 (movielens_rcmm_v2.csv)
   ↓
4. 모델 학습 (autoint_train.ipynb / autoint_mlp_train.ipynb)
   ↓
5. 모델 저장 (model/*.h5, data/field_dims.npy, data/label_encoders.pkl)
   ↓
6. 애플리케이션 로드 (movie_rec_app.py)
   ↓
7. 사용자 입력 → 추천 생성 → 결과 표시
```

---

## 📝 중요 파일 의존성

### 학습 시 필요한 파일 순서:
1. `data/ml-1m/movielens_rcmm_v2.csv` (전처리된 데이터)
2. `autoint.py` 또는 `autointmlp.py` (모델 정의)
3. 학습 후 생성: `field_dims.npy`, `label_encoders.pkl`, `*.h5` (가중치)

### 추론 시 필요한 파일:
1. `data/field_dims.npy` (필드 차원 정보)
2. `data/label_encoders.pkl` (인코딩용)
3. `data/ml-1m/*_prepro.csv` (사용자/영화/평점 데이터)
4. `model/*_weights.weights.h5` (학습된 가중치)
5. `autoint.py` 또는 `autointmlp.py` (모델 구조)

---

## 🎯 실행 순서

1. **데이터 준비**: `notebook/data_prepro.ipynb` 실행
2. **모델 학습**: 
   - `notebook/autoint_train.ipynb` 실행 (AutoInt)
   - `notebook/autoint_mlp_train.ipynb` 실행 (AutoIntMLP)
3. **애플리케이션 실행**: `streamlit run movie_rec_app.py`
