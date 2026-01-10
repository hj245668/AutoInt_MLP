# 실행(Execution)과 평가(Evaluation) 구분 가이드

## 📋 개요

이 프로젝트는 **실행(학습/추론)**과 **평가(성능 측정)** 부분이 명확히 구분되어 있습니다.

---

## 🚀 실행(Execution) 부분

### 1. **데이터 전처리 실행** (`notebook/data_prepro.ipynb`)

#### 목적
- 원본 데이터를 모델 학습에 적합한 형태로 변환

#### 실행 단계
```python
# 1. 원본 데이터 로드
users = pd.read_csv('data/ml-1m/users.dat', ...)
movies = pd.read_csv('data/ml-1m/movies.dat', ...)
ratings = pd.read_csv('data/ml-1m/ratings.dat', ...)

# 2. 데이터 전처리
- 연도/년대 추출
- 장르 분리 (genre1, genre2, genre3)
- 타임스탬프 변환 (rating_year, rating_month, rating_decade)
- 라벨 생성 (3점 이상 = 1)

# 3. 전처리된 데이터 저장
movies.to_csv('data/ml-1m/movies_prepro.csv', ...)
ratings.to_csv('data/ml-1m/ratings_prepro.csv', ...)
users.to_csv('data/ml-1m/users_prepro.csv', ...)

# 4. 통합 추천 데이터 생성
merge_mlens_data.to_csv('data/ml-1m/movielens_rcmm_v2.csv', ...)
```

#### 출력 파일
- `data/ml-1m/*_prepro.csv` (전처리된 개별 데이터)
- `data/ml-1m/movielens_rcmm_v2.csv` (통합 추천 데이터)

---

### 2. **모델 학습 실행** (`notebook/autoint_train.ipynb` / `autoint_mlp_train.ipynb`)

#### 목적
- 모델을 학습시켜 가중치를 생성

#### 실행 단계

##### 2-1. 데이터 준비
```python
# 통합 데이터 로드
movielens_rcmm = pd.read_csv('data/ml-1m/movielens_rcmm_v2.csv', dtype=str)

# 라벨 인코더 생성 및 적용
label_encoders = {col: LabelEncoder() for col in movielens_rcmm.columns[:-1]}
for col, le in label_encoders.items():
    movielens_rcmm[col] = le.fit_transform(movielens_rcmm[col])

# 학습/테스트 분할
train_df, test_df = train_test_split(movielens_rcmm, test_size=0.2, random_state=42)

# field_dims 계산 (임베딩 차원 정의)
field_dims = np.max(movielens_rcmm[u_i_feature + meta_features].astype(np.int64).values, axis=0) + 1
```

##### 2-2. 모델 정의
```python
# AutoInt 모델 생성
autoInt_model = AutoIntModel(
    field_dims=field_dims,
    embedding_size=embed_dim,
    att_layer_num=3,
    att_head_num=2,
    att_res=True,
    ...
)

# 또는 AutoIntMLP 모델 생성
autoIntMLP_model = AutoIntMLPModel(
    field_dims=field_dims,
    embedding_size=embed_dim,
    dnn_hidden_units=(32, 32),
    ...
)
```

##### 2-3. 모델 컴파일
```python
optimizer = Adam(learning_rate=learning_rate)
loss_fn = BinaryCrossentropy(from_logits=False)
autoInt_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['binary_crossentropy'])
```

##### 2-4. 모델 학습
```python
# 학습 실행
history = autoInt_model.fit(
    train_df[u_i_feature + meta_features], 
    train_df[label], 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_split=0.1
)
```

##### 2-5. 모델 저장
```python
# 필드 차원 정보 저장
np.save('data/field_dims.npy', field_dims)

# 모델 가중치 저장
autoInt_model.save_weights('model/autoInt_model_weights.weights.h5')

# 라벨 인코더 저장
joblib.dump(label_encoders, 'model/label_encoders.pkl')
```

#### 출력 파일
- `data/field_dims.npy` (필드 차원 정보)
- `model/autoInt_model_weights.weights.h5` (AutoInt 가중치)
- `model/autoIntMLP_model_weights.weights.h5` (AutoIntMLP 가중치)
- `model/label_encoders.pkl` (라벨 인코더)

---

### 3. **추론 실행** (`movie_rec_app.py`)

#### 목적
- 학습된 모델을 사용하여 실제 추천 생성

#### 실행 단계

##### 3-1. 데이터 및 모델 로드
```python
@st.cache_resource
def load_data():
    # 데이터 로드
    field_dims = np.load('data/field_dims.npy')
    ratings_df = pd.read_csv('data/ml-1m/ratings_prepro.csv')
    movies_df = pd.read_csv('data/ml-1m/movies_prepro.csv')
    user_df = pd.read_csv('data/ml-1m/users_prepro.csv')
    label_encoders = joblib.load('data/label_encoders.pkl')
    
    # 모델 로드
    model_autoint = AutoIntModel(...)
    model_autoint.load_weights('model/autoInt_model_weights.weights.h5')
    
    model_autointmlp = AutoIntMLPModel(...)
    model_autointmlp.load_weights('model/autoIntMLP_model_weights.weights.h5')
    
    return user_df, movies_df, ratings_df, model_autoint, model_autointmlp, label_encoders
```

##### 3-2. 추천 생성
```python
def get_recommendations(user, user_non_seen_dict, user_df, movies_df, 
                       r_year, r_month, model, label_encoders, predict_fn):
    # 사용자가 보지 않은 영화 필터링
    user_non_seen_movie = user_non_seen_dict.get(user)
    
    # 피처 데이터 준비
    merge_data = pd.concat([user_non_seen_movie_df, user_info], axis=1)
    
    # 인코딩
    for col, le in label_encoders.items():
        merge_data[col] = le.fit_transform(merge_data[col])
    
    # 예측 (predict_model 함수 사용)
    recom_top = predict_fn(model, merge_data)
    
    return movies_df[movies_df['movie_id'].isin(origin_m_id)]
```

#### 실행 방법
```bash
streamlit run movie_rec_app.py
```

---

## 📊 평가(Evaluation) 부분

### 1. **평가 지표 함수 정의**

#### 위치
- `notebook/autoint_train.ipynb` (Cell 4-5)
- `notebook/autoint_mlp_train.ipynb` (Cell 6-7)

#### 평가 함수들

##### NDCG (Normalized Discounted Cumulative Gain)
```python
def get_DCG(ranklist, y_true):
    """DCG 계산"""
    dcg = 0.0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item in y_true:
            dcg += 1.0 / math.log(i + 2)
    return dcg

def get_IDCG(ranklist, y_true):
    """Ideal DCG 계산"""
    idcg = 0.0
    i = 0
    for item in y_true:
        if item in ranklist:
            idcg += 1.0 / math.log(i + 2)
            i += 1
    return idcg

def get_NDCG(ranklist, y_true):
    """NDCG 평가 지표"""
    ranklist = np.array(ranklist).astype(int)
    y_true = np.array(y_true).astype(int)
    dcg = get_DCG(ranklist, y_true)
    idcg = get_IDCG(y_true, y_true)
    if idcg == 0:
        return 0
    return round((dcg / idcg), 5)
```

##### Hit Rate
```python
def get_hit_rate(ranklist, y_true):
    """hitrate 평가 지표"""
    c = 0
    for y in y_true:
        if y in ranklist:
            c += 1
    return round(c / len(y_true), 5)
```

---

### 2. **테스트 예측 함수**

#### 위치
- `notebook/autoint_train.ipynb` (Cell 6)
- `notebook/autoint_mlp_train.ipynb` (Cell 8)

#### 함수 정의
```python
def test_model(model, test_df, batch_size):
    """모델 테스트 - 예측 수행"""
    user_pred_info = defaultdict(list)
    total_rows = len(test_df)
    
    for i in range(0, total_rows, batch_size):
        features = test_df.iloc[i:i + batch_size, :-1].values
        y_pred = model.predict(features, verbose=False)
        
        for feature, p in zip(features, y_pred):
            u_i = feature[:2]  # user_id, movie_id
            user_pred_info[int(u_i[0])].append((int(u_i[1]), float(p)))
    
    return user_pred_info
```

---

### 3. **평가 실행 단계**

#### 위치
- `notebook/autoint_train.ipynb` (Cell 20-22)
- `notebook/autoint_mlp_train.ipynb` (Cell 25-27)

#### 평가 프로세스

##### 3-1. 테스트 데이터 예측
```python
# 사용자별 예측 정보 저장
user_pred_info = {}
top = 10  # 상위 10개 추천

# 모델로 테스트 데이터 예측
mymodel_user_pred_info = test_model(autoInt_model, test_df, batch_size)

# 사용자별로 상위 top개만 추출
for user, data_info in tqdm(mymodel_user_pred_info.items()):
    ranklist = sorted(data_info, key=lambda s: s[1], reverse=True)[:top]
    ranklist = list(dict.fromkeys([r[0] for r in ranklist]))
    user_pred_info[str(user)] = ranklist

# 실제 테스트 데이터에서 label=1인 영화 추출 (정답)
test_data = test_df[test_df['label']==1].groupby('user_id')['movie_id'].apply(list)
```

##### 3-2. NDCG 계산
```python
mymodel_ndcg_result = {}

# 각 사용자별 NDCG 계산
for user, data_info in tqdm(test_data.items()):
    mymodel_pred = user_pred_info.get(str(user))
    testset = list(set(np.array(data_info).astype(int)))
    mymodel_pred = mymodel_pred[:top]
    
    # NDCG 값 구하기
    user_ndcg = get_NDCG(mymodel_pred, testset)
    mymodel_ndcg_result[user] = user_ndcg
```

##### 3-3. Hit Rate 계산
```python
mymodel_hitrate_result = {}

# 각 사용자별 Hit Rate 계산
for user, data_info in tqdm(test_data.items()):
    mymodel_pred = user_pred_info.get(str(user))
    testset = list(set(np.array(data_info).astype(int)))
    mymodel_pred = mymodel_pred[:top]
    
    # hitrate 값 구하기
    user_hitrate = get_hit_rate(mymodel_pred, testset)
    mymodel_hitrate_result[user] = user_hitrate
```

##### 3-4. 평가 결과 출력
```python
# 전체 평균 성능 출력
print("mymodel ndcg : ", round(np.mean(list(mymodel_ndcg_result.values())), 5))
print("mymodel hitrate : ", round(np.mean(list(mymodel_hitrate_result.values())), 5))
```

#### 예상 출력
```
mymodel ndcg :  0.6619
mymodel hitrate :  0.63049
```

---

## 🔄 실행과 평가의 관계

### 학습 노트북에서의 흐름

```
[실행 부분]
1. 데이터 로드 및 전처리
2. 모델 정의 및 컴파일
3. 모델 학습 (fit)
4. 모델 저장
   ↓
[평가 부분]
5. 테스트 데이터 예측 (test_model)
6. 평가 지표 계산 (NDCG, Hit Rate)
7. 결과 출력
```

### 주요 차이점

| 구분 | 실행(Execution) | 평가(Evaluation) |
|------|----------------|-----------------|
| **목적** | 모델 학습 및 추론 | 모델 성능 측정 |
| **데이터** | train_df (학습) | test_df (평가) |
| **함수** | `model.fit()`, `model.predict()` | `test_model()`, `get_NDCG()`, `get_hit_rate()` |
| **출력** | 모델 가중치 파일 | 성능 지표 (NDCG, Hit Rate) |
| **위치** | 학습 노트북 전반부 | 학습 노트북 후반부 |

---

## 📍 평가가 수행되는 위치

### 1. **학습 노트북 내 평가**
- `notebook/autoint_train.ipynb`: AutoInt 모델 평가
- `notebook/autoint_mlp_train.ipynb`: AutoIntMLP 모델 평가
- **목적**: 학습된 모델의 성능 검증

### 2. **애플리케이션에서의 평가**
- `movie_rec_app.py`는 **평가를 수행하지 않음**
- 단순히 추론만 수행하여 사용자에게 추천 결과 제공

### 3. **모델 테스트 노트북**
- `notebook/model_load_test.ipynb`: 모델 로드 및 기본 테스트
- **목적**: 저장된 모델이 제대로 로드되는지 검증

---

## 🎯 실행 순서 요약

### 전체 파이프라인
```
1. [실행] data_prepro.ipynb
   → 전처리된 데이터 생성

2. [실행] autoint_train.ipynb / autoint_mlp_train.ipynb
   → 모델 학습 (fit)
   → [평가] 테스트 데이터로 성능 측정
   → 모델 저장

3. [실행] movie_rec_app.py
   → 모델 로드 및 추론
   → 사용자에게 추천 제공
```

---

## 💡 핵심 포인트

1. **실행과 평가는 분리되어 있음**
   - 학습은 `fit()` 함수로 실행
   - 평가는 별도의 `test_model()` 및 평가 함수로 수행

2. **평가는 학습 후에만 수행**
   - 학습이 완료된 후 테스트 데이터로 평가
   - 평가 결과는 모델 성능을 확인하는 용도

3. **애플리케이션은 평가 없이 추론만 수행**
   - `movie_rec_app.py`는 평가 지표를 계산하지 않음
   - 단순히 사용자에게 추천 결과만 제공

4. **평가 지표**
   - **NDCG**: 순위 품질 측정 (0~1, 높을수록 좋음)
   - **Hit Rate**: 추천 정확도 측정 (0~1, 높을수록 좋음)
