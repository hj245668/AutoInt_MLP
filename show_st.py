import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="영화 추천 시스템",
    page_icon="🎬",
    layout="wide"
)

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent if '__file__' in locals() else Path.cwd()
data_path = PROJECT_ROOT / "data" / "ml-1m"
model_path = PROJECT_ROOT / "model"

FEATURE_COLS = [
    'user_id', 'movie_id', 'movie_decade', 'movie_year',
    'rating_year', 'rating_month', 'rating_decade',
    'genre1', 'genre2', 'genre3', 'gender', 'age', 'occupation', 'zip'
]

def normalize_inputs(movies_df, users_df, ratings_df):
    # 타입 통일: 여기서 한 번에 고정해두면 이후 버그가 크게 줄어요.
    users_df['user_id'] = users_df['user_id'].astype(int)
    movies_df['movie_id'] = movies_df['movie_id'].astype(int)
    ratings_df['user_id'] = ratings_df['user_id'].astype(int)
    ratings_df['movie_id'] = ratings_df['movie_id'].astype(int)
    return movies_df, users_df, ratings_df

def safe_label_encode(df: pd.DataFrame, label_encoders: dict, feature_cols=FEATURE_COLS) -> pd.DataFrame:
    """
    추론(inference)에서는 transform만.
    unknown 처리: 'no'가 classes에 있으면 'no'로 치환, 아니면 에러로 중단(전처리 불일치).
    """
    encoded = df.copy()
    for col in feature_cols:
        le = label_encoders.get(col)
        if le is None:
            raise KeyError(f"label_encoders에 '{col}'이 없습니다. 학습/추론 컬럼 구성이 다를 수 있어요.")

        classes = set(getattr(le, "classes_", []))
        if len(classes) == 0:
            raise ValueError(f"'{col}' encoder에 classes_가 없습니다. pkl이 깨졌을 수 있어요.")

        if "no" in classes:
            encoded[col] = encoded[col].apply(lambda x: x if x in classes else "no")
        else:
            unknowns = set(encoded[col].unique()) - classes
            if unknowns:
                raise ValueError(f"[{col}] 학습에 없던 값(unknown)이 있습니다: {list(sorted(unknowns))[:10]} ...")

        encoded[col] = le.transform(encoded[col])
    return encoded

def build_pred_df_for_user_movies(user_row: pd.Series, movies_subset: pd.DataFrame, target_year: int, target_month: int) -> pd.DataFrame:
    """
    user_row(단일 사용자) + 영화 후보(movies_subset)로 모델 입력 df 생성
    """
    r_decade = f"{(int(target_year)//10)*10}s"
    # month는 1~12 int로 통일 (csv가 int라면 int 유지가 더 안전)
    target_month = int(target_month)

    base = movies_subset[['movie_id', 'movie_decade', 'movie_year', 'genre1', 'genre2', 'genre3']].copy()
    base['user_id'] = int(user_row['user_id'])
    base['gender'] = user_row['gender']
    base['age'] = int(user_row['age'])
    base['occupation'] = int(user_row['occupation'])
    base['zip'] = str(user_row['zip'])

    base['rating_year'] = int(target_year)
    base['rating_month'] = int(target_month)
    base['rating_decade'] = r_decade

    # 결측/빈값 정리 (genre2/genre3가 NaN일 수 있음)
    for g in ['genre2', 'genre3']:
        base[g] = base[g].fillna('no')
        base[g] = base[g].replace('', 'no')

    # 모델 컬럼 순서 보장
    base = base[FEATURE_COLS]
    return base



# ===== 모델 및 데이터 로딩 함수 =====
@st.cache_resource
def load_model(model_type="autoint"):
    """모델 로딩"""
    if model_type == "autoint":
        from autoint import AutoIntModel
        weights_file = "autoint_model_weights.weights.h5"
    else:  # autointmlp
        from autointmlp import AutoIntMLPModel
        weights_file = "autointmlp_model_weights.weights.h5"
    
    # field_dims 로드 (없으면 생성)
    field_dims_path = model_path / "field_dims.npy"
    if not field_dims_path.exists():
        st.warning("field_dims.npy를 찾을 수 없어 자동 생성합니다...")
        field_dims = generate_field_dims()
        np.save(field_dims_path, field_dims)
    
    field_dims = np.load(field_dims_path)
    
    # 모델 생성
    if model_type == "autoint":
        model = AutoIntModel(
            field_dims=field_dims,
            embedding_size=16,
            att_layer_num=3,
            att_head_num=2,
            att_res=True,
            l2_reg_dnn=0,
            l2_reg_embedding=1e-5,
            dnn_use_bn=False,
            dnn_dropout=0.4,
            init_std=0.0001
        )
    else:
        model = AutoIntMLPModel(
            field_dims=field_dims,
            embedding_size=16,
            att_layer_num=3,
            att_head_num=2,
            att_res=True,
            dnn_hidden_units=(32, 32),
            dnn_activation='relu',
            l2_reg_dnn=0,
            l2_reg_embedding=1e-5,
            dnn_use_bn=False,
            dnn_dropout=0.4,
            init_std=0.0001
        )
    
    # 더미 입력으로 빌드
    dummy_x = tf.constant([[0] * len(field_dims)], dtype=tf.int64)
    _ = model(dummy_x)
    
    # 가중치 로드
    model.load_weights(str(model_path / weights_file))
    
    return model

@st.cache_data
def load_data():
    """데이터 및 인코더 로딩"""
    # CSV 파일 로드
    movies_df = pd.read_csv(data_path / "movies_prepro.csv")
    users_df = pd.read_csv(data_path / "users_prepro.csv")
    ratings_df = pd.read_csv(data_path / "ratings_prepro.csv")
    
    # Label Encoders 로드
    label_encoders = joblib.load(model_path / "label_encoders.pkl")
    
    return movies_df, users_df, ratings_df, label_encoders

def generate_field_dims():
    """field_dims 자동 생성"""
    # movielens_rcmm_v2.csv 로드
    movielens_rcmm = pd.read_csv(data_path / "movielens_rcmm_v2.csv", dtype=str)
    
    # field_dims 계산
    u_i_feature = ['user_id', 'movie_id']
    meta_features = ['movie_decade', 'movie_year', 'rating_year', 'rating_month', 
                     'rating_decade', 'genre1','genre2', 'genre3', 'gender', 
                     'age', 'occupation', 'zip']
    
    field_dims = np.max(movielens_rcmm[u_i_feature + meta_features].astype(np.int64).values, axis=0) + 1
    
    return field_dims

# ===== 추천 함수 =====
def recommend_movies_for_new_user(
    model, user_info, movies_df, users_df, label_encoders,
    target_year: int, target_month: int,
    top_k=10,
    proxy_users_n: int = 20,
    enforce_year_filter: bool = False,
):
    """
    신규 사용자는 user_id가 없어서 모델 입력이 어려움.
    해결: 비슷한 사용자(성별/나이/직업)들의 user_id를 proxy로 뽑아 점수를 평균.
    """

    # 후보 영화: 전체(또는 연도 정책 필터 적용)
    candidates = movies_df.copy()
    if enforce_year_filter and 'movie_year' in candidates.columns:
        candidates = candidates[candidates['movie_year'] <= int(target_year)].copy()
        if candidates.empty:
            st.warning("선택한 연도 이전에 후보 영화가 없습니다.")
            return pd.DataFrame()

    # 유사 사용자 찾기 (필요하면 조건을 완화)
    gender = user_info['gender']
    age = int(user_info['age'])
    occ = int(user_info['occupation'])

    similar = users_df[
        (users_df['gender'] == gender) &
        (users_df['age'] == age) &
        (users_df['occupation'] == occ)
    ]

    if similar.empty:
        # 완화 1: 성별+나이만
        similar = users_df[(users_df['gender'] == gender) & (users_df['age'] == age)]
    if similar.empty:
        # 완화 2: 성별만
        similar = users_df[(users_df['gender'] == gender)]
    if similar.empty:
        # 최후: 전체에서 샘플
        similar = users_df

    proxy_users = similar.sample(n=min(proxy_users_n, len(similar)), random_state=42)

    # 각 proxy user로 점수 예측 후 평균
    all_scores = np.zeros(len(candidates), dtype=np.float32)

    for _, proxy in proxy_users.iterrows():
        pred_df = build_pred_df_for_user_movies(proxy, candidates, target_year, target_month)
        enc_df = safe_label_encode(pred_df, label_encoders, FEATURE_COLS)
        X = enc_df[FEATURE_COLS].values.astype(np.int64)
        scores = model.predict(X, batch_size=512, verbose=0).reshape(-1)
        all_scores += scores.astype(np.float32)

    all_scores /= len(proxy_users)

    top_indices = np.argsort(all_scores)[-top_k:][::-1]
    rec_rows = []
    for idx in top_indices:
        movie_data = candidates.iloc[idx]
        rec_rows.append({
            'movie_id': int(movie_data['movie_id']),
            'title': movie_data['title'],
            'year': int(movie_data['movie_year']),
            'decade': movie_data['movie_decade'],
            'genre1': movie_data['genre1'],
            'genre2': movie_data['genre2'] if pd.notna(movie_data['genre2']) else 'no',
            'genre3': movie_data['genre3'] if pd.notna(movie_data['genre3']) else 'no',
            'predicted_score': float(all_scores[idx]),
        })

    return pd.DataFrame(rec_rows)

def recommend_movies_for_existing_user(
    model, user_id, movies_df, users_df, ratings_df, label_encoders,
    target_year: int, target_month: int,
    top_k=10,
    enforce_year_filter: bool = False,
):
    # 타입 통일
    user_id = int(user_id)

    # 사용자가 이미 본 영화 (여기서 str 비교하면 망가집니다)
    seen_movies = set(ratings_df.loc[ratings_df['user_id'] == user_id, 'movie_id'].unique())

    user_row_df = users_df[users_df['user_id'] == user_id]
    if user_row_df.empty:
        st.warning("해당 사용자 ID를 users_df에서 찾지 못했습니다.")
        return None, None

    user_row = user_row_df.iloc[0]

    # 안 본 영화 후보
    unseen_movies = movies_df[~movies_df['movie_id'].isin(seen_movies)].copy()
    if unseen_movies.empty:
        st.warning("모든 영화를 이미 보셨습니다!")
        return None, None

    # (선택) 연도 정책 필터: 미래 영화 제외
    if enforce_year_filter and 'movie_year' in unseen_movies.columns:
        unseen_movies = unseen_movies[unseen_movies['movie_year'] <= int(target_year)]
        if unseen_movies.empty:
            st.warning("선택한 연도 이전에 볼 만한 후보 영화가 없습니다.")
            return None, None

    # 모델 입력 df 생성
    pred_df = build_pred_df_for_user_movies(user_row, unseen_movies, target_year, target_month)

    # 인코딩(transform only)
    enc_df = safe_label_encode(pred_df, label_encoders, FEATURE_COLS)

    # 예측
    X = enc_df[FEATURE_COLS].values.astype(np.int64)
    scores = model.predict(X, batch_size=512, verbose=0).reshape(-1)

    # Top-K
    top_indices = np.argsort(scores)[-top_k:][::-1]
    rec_rows = []
    for idx in top_indices:
        movie_data = unseen_movies.iloc[idx]
        rec_rows.append({
            'movie_id': int(movie_data['movie_id']),
            'title': movie_data['title'],
            'year': int(movie_data['movie_year']),
            'decade': movie_data['movie_decade'],
            'genre1': movie_data['genre1'],
            'genre2': movie_data['genre2'] if pd.notna(movie_data['genre2']) else 'no',
            'genre3': movie_data['genre3'] if pd.notna(movie_data['genre3']) else 'no',
            'predicted_score': float(scores[idx]),
        })
    recommendations_df = pd.DataFrame(rec_rows)

    # 시청 이력(최근 20개)
    user_r = ratings_df[ratings_df['user_id'] == user_id].copy()
    history = movies_df[movies_df['movie_id'].isin(user_r['movie_id'].values)].merge(
        user_r[['movie_id', 'rating', 'timestamp']], on='movie_id', how='left'
    )
    # timestamp가 문자열이면 정렬은 되지만, 진짜 시간 정렬하려면 datetime 변환 권장
    history = history.sort_values('timestamp', ascending=False)
    user_history_df = history[['title', 'movie_year', 'genre1', 'rating', 'timestamp']].head(20)

    return recommendations_df, user_history_df


# ===== 메인 앱 =====
def main():
    st.title("🎬 영화 추천 시스템")
    st.markdown("---")
    
    # 데이터 로딩
    try:
        with st.spinner("데이터를 불러오는 중..."):
            movies_df, users_df, ratings_df, label_encoders = load_data()
        st.success("✅ 데이터 로딩 완료!")
        movies_df, users_df, ratings_df = normalize_inputs(movies_df, users_df, ratings_df)

    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {e}")
        st.stop()
    
    # 사이드바: 모델 선택 및 설정
    st.sidebar.header("⚙️ 설정")
    
    model_type = st.sidebar.selectbox(
        "모델 선택",
        ["autoint", "autointmlp"],
        format_func=lambda x: "AutoInt (Attention Only)" if x == "autoint" else "AutoInt+MLP (Hybrid)"
    )
    
    top_k = st.sidebar.slider("추천 개수", min_value=5, max_value=20, value=10, step=1)
    
    # 모델 로딩
    try:
        with st.spinner(f"{model_type.upper()} 모델을 불러오는 중..."):
            model = load_model(model_type)
        st.sidebar.success(f"✅ {model_type.upper()} 모델 로딩 완료!")
    except Exception as e:
        st.sidebar.error(f"❌ 모델 로딩 실패: {e}")
        st.stop()
    
    # 메인 화면: 사용자 모드 선택
    st.header("👤 사용자 정보 입력")
    
    user_mode = st.radio(
        "사용자 유형을 선택하세요",
        ["🆕 새로운 사용자 (정보 직접 입력)", "👥 기존 사용자 (ID 선택)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # ===== 모드 1: 새로운 사용자 =====
    if user_mode == "🆕 새로운 사용자 (정보 직접 입력)":
        st.subheader("사용자 정보를 입력하세요")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("성별", ["M", "F"], format_func=lambda x: "남성" if x == "M" else "여성")
            age = st.selectbox(
                "나이대",
                [1, 18, 25, 35, 45, 50, 56],
                format_func=lambda x: {
                    1: "18세 미만",
                    18: "18-24세",
                    25: "25-34세",
                    35: "35-44세",
                    45: "45-49세",
                    50: "50-55세",
                    56: "56세 이상"
                }[x]
            )
        
        with col2:
            occupation = st.selectbox(
                "직업",
                list(range(21)),
                format_func=lambda x: {
                    0: "기타/미지정", 1: "학계/교육자", 2: "예술가", 3: "사무직/행정직",
                    4: "대학생/대학원생", 5: "고객 서비스", 6: "의사/보건의료",
                    7: "임원/관리직", 8: "농업/어업", 9: "주부", 10: "고등학생/중학생",
                    11: "변호사", 12: "프로그래머", 13: "은퇴", 14: "영업/마케팅",
                    15: "과학자", 16: "자영업", 17: "기술자/엔지니어",
                    18: "장인/제조", 19: "무직", 20: "작가"
                }[x]
            )
            zip_code = st.text_input("우편번호 (5자리)", value="00000", max_chars=5)
        
        st.info(f"""
        **입력하신 정보**
        - 성별: {'남성' if gender == 'M' else '여성'}
        - 나이대: {age}
        - 직업: {occupation}
        - 우편번호: {zip_code}
        """)
        
        # 통계 정보
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("전체 영화 수", f"{len(movies_df):,}개")
        with col_b:
            st.metric("추천 대상 영화", f"{len(movies_df):,}개")
        
        st.markdown("---")
        
        # 추천 버튼
        if st.button("🎯 추천 받기", type="primary", use_container_width=True):
            user_info = {
                'gender': gender,
                'age': age,
                'occupation': occupation,
                'zip': zip_code
            }
            
            with st.spinner("추천 결과를 생성하는 중..."):
                recommendations_df = recommend_movies_for_new_user(
                    model=model,
                    user_info=user_info,
                    movies_df=movies_df,
                    label_encoders=label_encoders,
                    top_k=top_k
                )
            
            # 결과 표시
            st.header("🎥 추천 결과")
            
            for idx, row in recommendations_df.iterrows():
                with st.container():
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.subheader(f"{idx+1}. {row['title']} ({row['year']})")
                        genres = [g for g in [row['genre1'], row['genre2'], row['genre3']] if g and g != 'no']
                        st.write(f"**장르**: {' | '.join(genres)}")
                    with col_b:
                        st.metric("예측 점수", row['predicted_score'])
                    st.markdown("---")
            
            # 다운로드 버튼
            csv = recommendations_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 추천 결과 다운로드 (CSV)",
                data=csv,
                file_name=f"recommendations_new_user.csv",
                mime="text/csv"
            )
    
    # ===== 모드 2: 기존 사용자 =====
    else:
        st.subheader("사용자 ID를 선택하세요")
        
        col1, col2 = st.columns(2)
        
        with col1:
            user_id_input = st.selectbox(
                "사용자 ID 선택",
                options=sorted(users_df['user_id'].unique()),
                index=5  # 기본값: user_id 6
            )
            
            # 선택된 사용자 정보 표시
            selected_user = users_df[users_df['user_id'] == user_id_input].iloc[0]
            st.info(f"""
            **선택된 사용자 정보**
            - 성별: {selected_user['gender']}
            - 나이: {selected_user['age']}
            - 직업 코드: {selected_user['occupation']}
            - 우편번호: {selected_user['zip']}
            """)
        
        with col2:
            st.metric("전체 영화 수", f"{len(movies_df):,}개")
            st.metric("전체 사용자 수", f"{len(users_df):,}명")
            user_rating_count = len(ratings_df[ratings_df['user_id'] == str(user_id_input)])
            st.metric("선택한 사용자의 평가 영화 수", f"{user_rating_count:,}개")
        
        st.markdown("---")
        
        # 추천 버튼
        if st.button("🎯 추천 받기", type="primary", use_container_width=True):
            with st.spinner("추천 결과를 생성하는 중..."):
                recommendations_df, user_history_df = recommend_movies_for_existing_user(
                    model=model,
                    user_id=user_id_input,
                    movies_df=movies_df,
                    users_df=users_df,
                    ratings_df=ratings_df,
                    label_encoders=label_encoders,
                    top_k=top_k
                )
            
            if recommendations_df is not None:
                # 결과 표시
                st.header("🎥 추천 결과")
                
                for idx, row in recommendations_df.iterrows():
                    with st.container():
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.subheader(f"{idx+1}. {row['title']} ({row['year']})")
                            genres = [g for g in [row['genre1'], row['genre2'], row['genre3']] if g and g != 'no']
                            st.write(f"**장르**: {' | '.join(genres)}")
                        with col_b:
                            st.metric("예측 점수", row['predicted_score'])
                        st.markdown("---")
                
                # 다운로드 버튼
                csv = recommendations_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 추천 결과 다운로드 (CSV)",
                    data=csv,
                    file_name=f"recommendations_user_{user_id_input}.csv",
                    mime="text/csv"
                )
                
                # 사용자 시청 이력
                if user_history_df is not None and len(user_history_df) > 0:
                    st.header("📺 사용자 시청 이력 (최근 20개)")
                    st.dataframe(
                        user_history_df,
                        use_container_width=True,
                        hide_index=True
                    )

if __name__ == "__main__":
    main()