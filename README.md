# PJT
# AutoInt Recommendation System 

**AutoInt** 및 **AutoInt_MLP** 적용 영화/콘텐츠 추천 시스템 구현
**Streamlit** 대시보드로 시각화

---

## 1. Environment Setup(가상환경)

**Python 3.11** 기반
```bash
# 1. 가상환경 생성 (이름: ds6_rcmm)
conda create -n ds6_rcmm python=3.11 -y
# 2. 가상환경 활성화
conda activate ds6_rcmm
# 필수 패키지 설치
pip install -r requirements.txt

```

---

## 2. Usage (실행 방법)

프로젝트 폴더 내에서 아래 명령어를 입력하면 웹 인터페이스(Streamlit)를 통해 추천 시스템을 확인할 수 있습니다.

* **`show_st.py`**: Lecture 모델 (**AutoInt**) 스트림릿 실행 코드
* **`show_st_plus.py`**: Project 모델 (**AutoInt_MLP**) 스트림릿 실행 코드

```bash
# 기본 AutoInt 모델 실행
streamlit run show_st.py

# 성능 개선된 AutoInt_MLP 모델 실행
streamlit run show_st_plus.py

```

---

##  3. Notebooks (노트북 코드)

학습 및 데이터 처리는 아래 순서대로 구성되어 있습니다.

| 순서 | 파일명 | 설명 |
| --- | --- | --- |
| 1 | `data_EDA.ipynb` | 데이터를 로드하고 기초적인 탐색적 데이터 분석(EDA) 수행 |
| 2 | `data_prepro.ipynb` | 학습용 데이터 전처리 (기본 전처리 데이터는 이미 제공됨) |
| 3 | `autoint_train.ipynb` | **AutoInt** 모델 학습 및 가중치 저장 |
| 4 | `autoint_mlp_train.ipynb` | **AutoInt_MLP** 모델 학습 및 가중치 저장 |
| 5 | `model_load_test.ipynb` | **모델 로드 및 정상 작동 여부 디버깅 코드** |

> Streamlit 실행 중 에러 발생 시 디버깅이 어려울 수 있음.
> 모델 수정 후 `model_load_test.ipynb` 가중치가 올바르게 로드되는지 먼저 확인.

---

## 4. 주의사항 및 참고

### (1) 모델 구조 일치

학습 파일(`autoint_mlp_train.ipynb`)에서 모델의 아키텍처(Layer 수, Embedding 차원 등)를 변경했다면, 반드시 **`autointmlp.py` 파일 내의 모델 정의도 동일하게 수정**해야 가중치 로드가 가능합니다.

### (2) TensorFlow 버전 및 가중치 파일

* 사용 환경에 따라 가중치 저장 방식이 다를 수 있습니다. 현재 코드는 가중치 파일 뒤에 `.weights` 확장자가 붙는 형식을 지원하도록 설정되어 있습니다.

### (3) 데이터 타입 설정 (`dtype`)

* 코드 내에서 `dtype.longlong` 부분은 라이브러리 버전에 따라 에러가 발생할 수 있습니다. 이 경우 시스템 환경에 맞춰 `int64` 또는 `int32`로 변경하여 사용하시기 바랍니다.
* **PyTorch**: `torch.long` 또는 `torch.int64`
* **NumPy/TF**: `np.int64`



---

## 5. Next Steps (추가 과제)

* **성능 향상**: 다양한 하이퍼파라미터 튜닝 및 Feature Engineering을 시도
* **UI 개선**: Streamlit에서 제공하는 다양한 위젯 활용 화면 꾸미기

---
