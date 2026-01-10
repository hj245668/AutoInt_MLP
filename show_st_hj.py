import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib
from autoint import AutoIntModel, predict_model

# 페이지 설정
st.set_page_config(
    page_title="AutoInt 영화 추천",
    page_icon="🎬",
    layout="wide"
)

# 커스텀 CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1565C0;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(120deg, #E3F2FD, #BBDEFB);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .info-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1565C0;
    }
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1565C0;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    """데이터 및 모델 로드"""
    project_path = os.path.abspath(os.getcwd())
    data_dir_nm = 'data'
    movielens_dir_nm = 'ml-1m'
    model_dir_nm = 'model'
    data_path = f"{project_path}/{data_dir_nm}"
    model_path = f"{project_path}/{model_dir_nm}"
    
    field_dims = np.load(f'{data_path}/field_dims.npy')
    dropout = 0.4
    embed_dim = 16
    
    ratings_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/ratings_prepro.csv')
    movies_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/movies_prepro.csv')
    user_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/users_prepro.csv')
    
    model = AutoIntModel(
        field_dims, embed_dim, att_layer_num=3, att_head_num=2, att_res=True,
        l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, 
        dnn_dropout=dropout, init_std=0.0001
    )
    model(tf.constant([[0] * len(field_dims)], dtype=tf.int64))
    model.load_weights(f'{model_path}/autoInt_model_weights.weights.h5')
    
    label_encoders = joblib.load(f'{data_path}/label_encoders.pkl')
    
    return user_df, movies_df, ratings_df, model, label_encoders

def get_user_seen_movies(ratings_df):
    """사용자가 과거에 본 영화 목록"""
    return ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()

def get_user_non_seen_dict(movies_df, user_df, user_seen_movies):
    """사용자가 보지 않은 영화 딕셔너리"""
    unique_movies = movies_df['movie_id'].unique()
    unique_users = user_df['user_id'].unique()
    user_non_seen_dict = dict()
    
    for user in unique_users:
        user_seen_movie_list = user_seen_movies[user_seen_movies['user_id'] == user]['movie_id'].values[0]
        user_non_seen_movie_list = list(set(unique_movies) - set(user_seen_movie_list))
        user_non_seen_dict[user] = user_non_seen_movie_list
    
    return user_non_seen_dict

def get_user_info(user_id, users_df):
    """사용자 정보 조회"""
    return users_df[users_df['user_id'] == user_id]

def get_user_past_interactions(user_id, ratings_df, movies_df):
    """사용자 평점 4점 이상 영화"""
    return ratings_df[
        (ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)
    ].merge(movies_df, on='movie_id')

def get_recom(
    user_id: int,
    user_non_seen_dict: dict,
    user_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    r_year: int,
    r_month: int,
    model,
    label_encoders: dict,
    top_k: int = 10,
    enforce_year_filter: bool = False,
):
    """
    정상 추천 파이프라인:
    1) 후보 생성(안 본 영화)
    2) raw feature 구성
    3) (중요) 저장된 label_encoders로 transform만 수행 (fit 금지)
    4) AutoInt로 score 예측 후 score 기준 top-k
    5) movie_id 디코딩 + 랭킹/스코어 보존해서 결과 반환
    """

    # ---- 0) 후보 가져오기 ----
    user_non_seen_movie = user_non_seen_dict.get(user_id)
    if not user_non_seen_movie:
        return pd.DataFrame(columns=list(movies_df.columns) + ["score", "rank"])

    # ---- 1) raw feature 만들기 ----
    r_decade = f"{(r_year // 10) * 10}s"

    # 후보 영화 메타
    cand_movies = pd.DataFrame({"movie_id": user_non_seen_movie}).merge(
        movies_df, on="movie_id", how="left"
    )

    # 사용자 메타(한 행)
    user_row = user_df[user_df["user_id"] == user_id]
    if user_row.empty:
        return pd.DataFrame(columns=list(movies_df.columns) + ["score", "rank"])

    # 후보 개수만큼 사용자행 복제
    user_info = pd.concat([user_row] * len(cand_movies), ignore_index=True)
    user_info["rating_year"] = r_year
    user_info["rating_month"] = r_month
    user_info["rating_decade"] = r_decade

    # 합치기
    merge_data = pd.concat([cand_movies.reset_index(drop=True), user_info.reset_index(drop=True)], axis=1)

    # 모델이 기대하는 컬럼만(학습과 동일 순서 중요)
    feature_cols = [
        "user_id", "movie_id",
        "movie_decade", "movie_year",
        "rating_year", "rating_month", "rating_decade",
        "genre1", "genre2", "genre3",
        "gender", "age", "occupation", "zip"
    ]
    merge_data = merge_data[feature_cols].copy()

    # 결측 처리
    merge_data = merge_data.fillna("no")

    # ---- 2) encoding: transform ONLY (fit 금지) ----
    def safe_transform(col: str, series: pd.Series) -> np.ndarray:
        le = label_encoders.get(col)
        if le is None:
            raise KeyError(f"label_encoders에 '{col}'이 없습니다. 학습/추론 컬럼 구성이 다를 수 있어요.")

        # sklearn LabelEncoder는 unknown 처리 기능이 없어서 방어 로직 필요
        classes = set(le.classes_.tolist())
        if "no" in classes:
            series = series.map(lambda x: x if x in classes else "no")
        else:
            unknowns = set(series.unique()) - classes
            if unknowns:
                raise ValueError(
                    f"[{col}]에 학습에 없던 값이 들어왔습니다: {list(sorted(unknowns))[:10]} ... "
                    f"(해결: 학습 시 'no' 같은 토큰을 포함하거나, 전처리를 맞추세요)"
                )
        return le.transform(series)

    encoded_df = merge_data.copy()
    for col in feature_cols:
        encoded_df[col] = safe_transform(col, encoded_df[col])

    # ---- 3) scoring & ranking ----
    # predict_model은 (movie_encoded_id, score) top10을 반환 :contentReference[oaicite:2]{index=2}
    ranked = predict_model(model, encoded_df)

    if not ranked:
        return pd.DataFrame(columns=list(movies_df.columns) + ["score", "rank"])

    # get top_k 유지 (predict_model 내부 top=10이지만, 혹시를 대비)
    ranked = ranked[:top_k]

    movie_encoded_ids = [mid for (mid, score) in ranked]
    scores = [score for (mid, score) in ranked]

    # ---- 4) decode movie_id back to original ----
    movie_le = label_encoders["movie_id"]
    origin_movie_ids = movie_le.inverse_transform(np.array(movie_encoded_ids, dtype=int))

    # ---- 5) 결과 조립 (랭킹/스코어 보존) ----
    result = movies_df[movies_df["movie_id"].isin(origin_movie_ids)].copy()

    # 랭킹 순서대로 정렬 보장
    order_map = {int(mid): i for i, mid in enumerate(origin_movie_ids)}
    result["rank"] = result["movie_id"].map(order_map)
    result["score"] = result["rank"].map(lambda i: scores[int(i)] if pd.notna(i) else np.nan)
    result = result.sort_values("rank").reset_index(drop=True)

    # ---- 6) (선택) 정책 필터: 입력 연도 이전 영화만 보이게 ----
    if enforce_year_filter and "movie_year" in result.columns:
        result = result[result["movie_year"] <= r_year].reset_index(drop=True)

    return result


# 데이터 로드
try:
    users_df, movies_df, ratings_df, model, label_encoders = load_data()
    user_seen_movies = get_user_seen_movies(ratings_df)
    user_non_seen_dict = get_user_non_seen_dict(movies_df, users_df, user_seen_movies)
except Exception as e:
    st.error(f"⚠️ 데이터 로드 오류: {str(e)}")
    st.stop()

# 메인 UI
st.markdown('<div class="main-header">🎬 AutoInt 영화 추천 시스템</div>', unsafe_allow_html=True)

# 한 줄 입력
col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

with col1:
    user_id = st.number_input(
        "👤 사용자 ID", 
        min_value=int(users_df['user_id'].min()), 
        max_value=int(users_df['user_id'].max()), 
        value=int(users_df['user_id'].min())
    )

with col2:
    r_year = st.number_input(
        "📅 연도", 
        min_value=int(ratings_df['rating_year'].min()), 
        max_value=int(ratings_df['rating_year'].max()), 
        value=int(ratings_df['rating_year'].min())
    )

with col3:
    r_month = st.number_input(
        "📆 월", 
        min_value=int(ratings_df['rating_month'].min()), 
        max_value=int(ratings_df['rating_month'].max()), 
        value=int(ratings_df['rating_month'].min())
    )

with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    recommend_button = st.button("🎯 추천", use_container_width=True)

# 추천 결과
if recommend_button:
    with st.spinner('🔄 추천 생성 중...'):
        
        # 사용자 정보
        user_info = get_user_info(user_id, users_df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-container"><h3>👤</h3><p>성별</p><h4>{}</h4></div>'.format(
                user_info['gender'].values[0]), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-container"><h3>🎂</h3><p>나이</p><h4>{}</h4></div>'.format(
                user_info['age'].values[0]), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-container"><h3>💼</h3><p>직업</p><h4>{}</h4></div>'.format(
                user_info['occupation'].values[0]), unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-container"><h3>📍</h3><p>지역</p><h4>{}</h4></div>'.format(
                user_info['zip'].values[0]), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 2단 레이아웃
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown('<div class="info-section">', unsafe_allow_html=True)
            st.markdown("### 🎥 선호 영화 이력")
            user_interactions = get_user_past_interactions(user_id, ratings_df, movies_df)
            
            if len(user_interactions) > 0:
                st.caption(f"평점 4점 이상 • 총 {len(user_interactions)}개")
                display_df = user_interactions[['title', 'genres', 'rating']].head(10)
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=350)
            else:
                st.info("평점 4점 이상의 영화가 없습니다.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_right:
            st.markdown('<div class="info-section">', unsafe_allow_html=True)
            st.markdown("### ⭐ 추천 결과")
            recommendations = get_recom(
                user_id, user_non_seen_dict, users_df, movies_df, 
                r_year, r_month, model, label_encoders
            )
            
            st.caption(f"AutoInt 모델 • {len(recommendations)}개 추천")
            
            # 카드 형식으로 표시
            for idx, movie in recommendations.iterrows():
                st.markdown(f"""
                <div class="movie-card">
                    <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.3rem;">
                        🎬 {movie['title']}
                    </div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">
                        🎭 {movie['genres']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

else:
    # 초기 화면
    st.markdown('<div class="info-section">', unsafe_allow_html=True)
    st.markdown("### 📊 시스템 정보")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 사용자", f"{len(users_df):,}명")
    with col2:
        st.metric("🎬 영화", f"{len(movies_df):,}개")
    with col3:
        st.metric("⭐ 평점", f"{len(ratings_df):,}개")
    
    st.markdown("---")
    st.markdown("""
    **🔹 AutoInt 모델**
    - Attention 메커니즘으로 feature 간 상호작용 자동 학습
    - 해석 가능하고 효율적인 추천
    """)
    st.markdown('</div>', unsafe_allow_html=True)
