import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib
from autointmlp import AutoIntMLPModel, predict_model

# 페이지 설정
st.set_page_config(
    page_title="AutoInt+MLP 영화 추천",
    page_icon="🎬",
    layout="wide"
)

# 커스텀 CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #6A1B9A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(120deg, #F3E5F5, #E1BEE7);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .info-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #6A1B9A;
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
        background-color: #6A1B9A;
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
    
    model = AutoIntMLPModel(
        field_dims, embed_dim, att_layer_num=3, att_head_num=2, att_res=True, 
        dnn_hidden_units=(32, 32), dnn_activation='relu',
        l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, 
        dnn_dropout=dropout, init_std=0.0001
    )
    model(tf.constant([[0] * len(field_dims)], dtype=tf.int64))
    model.load_weights(f'{model_path}/autoIntMLP_model_weights.weights.h5')
    
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

def get_recom(user, user_non_seen_dict, user_df, movies_df, r_year, r_month, model, label_encoders):
    """영화 추천 생성"""
    user_non_seen_movie = user_non_seen_dict.get(user)
    user_id_list = [user for _ in range(len(user_non_seen_movie))]
    r_decade = str(r_year - (r_year % 10)) + 's'
    
    user_non_seen_movie_df = pd.merge(
        pd.DataFrame({'movie_id': user_non_seen_movie}), 
        movies_df, on='movie_id'
    )
    user_info = pd.merge(
        pd.DataFrame({'user_id': user_id_list}), 
        user_df, on='user_id'
    )
    user_info['rating_year'] = r_year
    user_info['rating_month'] = r_month
    user_info['rating_decade'] = r_decade
    
    merge_data = pd.concat([user_non_seen_movie_df, user_info], axis=1)
    merge_data.fillna('no', inplace=True)
    merge_data = merge_data[[
        'user_id', 'movie_id', 'movie_decade', 'movie_year', 
        'rating_year', 'rating_month', 'rating_decade', 
        'genre1', 'genre2', 'genre3', 'gender', 'age', 'occupation', 'zip'
    ]]
    
    for col, le in label_encoders.items():
        merge_data[col] = le.fit_transform(merge_data[col])
    
    recom_top = predict_model(model, merge_data)
    recom_top = [r[0] for r in recom_top]
    origin_m_id = label_encoders['movie_id'].inverse_transform(recom_top)
    
    result_movies = []
    for movie_id in origin_m_id:
        movie_info = movies_df[movies_df['movie_id'] == movie_id]
        if len(movie_info) > 0:
            result_movies.append(movie_info.iloc[0])
    
    return pd.DataFrame(result_movies)

# 데이터 로드
try:
    users_df, movies_df, ratings_df, model, label_encoders = load_data()
    user_seen_movies = get_user_seen_movies(ratings_df)
    user_non_seen_dict = get_user_non_seen_dict(movies_df, users_df, user_seen_movies)
except Exception as e:
    st.error(f"⚠️ 데이터 로드 오류: {str(e)}")
    st.stop()

# 메인 UI
st.markdown('<div class="main-header">🎬 AutoInt+MLP 영화 추천 시스템</div>', unsafe_allow_html=True)

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
            
            st.caption(f"AutoInt+MLP 모델 • {len(recommendations)}개 추천")
            
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
    **🔸 AutoInt+MLP 모델**
    - Attention 메커니즘 + Deep Neural Network
    - 복잡한 패턴 학습으로 고도화된 추천
    - DNN Hidden Units: (32, 32) + ReLU activation
    """)
    st.markdown('</div>', unsafe_allow_html=True)
