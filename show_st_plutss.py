import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib
from autoint import AutoIntModel, predict_model as predict_autoint
from autointmlp import AutoIntMLPModel, predict_model as predict_autointmlp

# 페이지 설정
st.set_page_config(
    page_title="영화 추천 시스템",
    page_icon="🎬",
    layout="wide"
)

# 커스텀 CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #262730;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #FF4B4B;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    """데이터 및 모델 로드"""
    project_path = os.path.abspath(os.getcwd())
    data_path = f"{project_path}/data"
    model_path = f"{project_path}/model"
    
    # 공통 데이터 로드
    field_dims = np.load(f'{data_path}/field_dims.npy')
    ratings_df = pd.read_csv(f'{data_path}/ml-1m/ratings_prepro.csv')
    movies_df = pd.read_csv(f'{data_path}/ml-1m/movies_prepro.csv')
    user_df = pd.read_csv(f'{data_path}/ml-1m/users_prepro.csv')
    label_encoders = joblib.load(f'{data_path}/label_encoders.pkl')
    
    # 모델 파라미터
    dropout = 0.4
    embed_dim = 16
    
    # AutoInt 모델
    model_autoint = AutoIntModel(
        field_dims, embed_dim, att_layer_num=3, att_head_num=2, att_res=True,
        l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, 
        dnn_dropout=dropout, init_std=0.0001
    )
    model_autoint(tf.constant([[0] * len(field_dims)], dtype=tf.int64))
    model_autoint.load_weights(f'{model_path}/autoInt_model_weights.weights.h5')
    
    # AutoIntMLP 모델
    model_autointmlp = AutoIntMLPModel(
        field_dims, embed_dim, att_layer_num=3, att_head_num=2, att_res=True,
        dnn_hidden_units=(32, 32), dnn_activation='relu',
        l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False,
        dnn_dropout=dropout, init_std=0.0001
    )
    model_autointmlp(tf.constant([[0] * len(field_dims)], dtype=tf.int64))
    model_autointmlp.load_weights(f'{model_path}/autoIntMLP_model_weights.weights.h5')
    
    return user_df, movies_df, ratings_df, model_autoint, model_autointmlp, label_encoders

def get_user_seen_movies(ratings_df):
    """사용자가 과거에 본 영화 목록"""
    return ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()

def get_user_non_seen_dict(movies_df, user_df, user_seen_movies):
    """사용자가 보지 않은 영화 딕셔너리"""
    unique_movies = movies_df['movie_id'].unique()
    unique_users = user_df['user_id'].unique()
    user_non_seen_dict = {}
    
    for user in unique_users:
        seen = user_seen_movies[user_seen_movies['user_id'] == user]['movie_id'].values[0]
        non_seen = list(set(unique_movies) - set(seen))
        user_non_seen_dict[user] = non_seen
    
    return user_non_seen_dict

def get_user_info(user_id, users_df):
    """사용자 정보 조회"""
    return users_df[users_df['user_id'] == user_id]

def get_user_past_interactions(user_id, ratings_df, movies_df):
    """사용자의 과거 선호 영화 (평점 4점 이상)"""
    merged = ratings_df[
        (ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)
    ].merge(movies_df, on='movie_id')
    
    merged['genres'] = merged[['genre1', 'genre2', 'genre3']].apply(
        lambda x: ', '.join([str(g) for g in x if pd.notna(g) and str(g) != '']), axis=1
    )
    
    return merged

def get_recommendations(user, user_non_seen_dict, user_df, movies_df, 
                       r_year, r_month, model, label_encoders, predict_fn):
    """영화 추천 생성"""
    user_non_seen_movie = user_non_seen_dict.get(user)
    user_id_list = [user] * len(user_non_seen_movie)
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
    
    recom_top = predict_fn(model, merge_data)
    recom_top = [r[0] for r in recom_top]
    origin_m_id = label_encoders['movie_id'].inverse_transform(recom_top)
    
    result = movies_df[movies_df['movie_id'].isin(origin_m_id)].copy()
    result['genres'] = result[['genre1', 'genre2', 'genre3']].apply(
        lambda x: ', '.join([str(g) for g in x if pd.notna(g) and str(g) != '']), axis=1
    )
    
    return result

# 데이터 로드
try:
    users_df, movies_df, ratings_df, model_autoint, model_autointmlp, label_encoders = load_data()
    user_seen_movies = get_user_seen_movies(ratings_df)
    user_non_seen_dict = get_user_non_seen_dict(movies_df, users_df, user_seen_movies)
except Exception as e:
    st.error(f"⚠️ 데이터 로드 중 오류 발생: {str(e)}")
    st.stop()

# 메인 UI
st.markdown('<h1 class="main-header">🎬 영화 추천 시스템</h1>', unsafe_allow_html=True)

# 사이드바 - 입력 파라미터
with st.sidebar:
    st.markdown("### ⚙️ 추천 설정")
    
    # 사용자 ID 직접 입력
    user_id = st.text_input(
        "👤 사용자 ID 입력",
        value="",
        placeholder="사용자 ID를 입력하세요"
    )
    
    # 입력값 검증 및 변환
    if user_id:
        try:
            user_id = int(user_id)
            if user_id not in users_df['user_id'].values:
                st.error(f"⚠️ 존재하지 않는 사용자 ID입니다. (범위: {int(users_df['user_id'].min())} ~ {int(users_df['user_id'].max())})")
                user_id = None
        except ValueError:
            st.error("⚠️ 숫자만 입력해주세요.")
            user_id = None
    else:
        user_id = None
    
    r_year = st.number_input(
        "📅 추천 타겟 연도",
        min_value=int(ratings_df['rating_year'].min()),
        max_value=int(ratings_df['rating_year'].max()),
        value=int(ratings_df['rating_year'].min())
    )
    
    r_month = st.number_input(
        "📆 추천 타겟 월",
        min_value=int(ratings_df['rating_month'].min()),
        max_value=int(ratings_df['rating_month'].max()),
        value=int(ratings_df['rating_month'].min())
    )
    
    model_choice = st.selectbox(
        "🤖 모델 선택",
        ["AutoInt", "AutoInt+MLP", "두 모델 비교"],
        help="AutoInt+MLP는 추가 Deep Neural Network 레이어를 포함합니다"
    )
    
    recommend_button = st.button("🎯 추천 결과 보기", type="primary", use_container_width=True)

# 메인 컨텐츠
if recommend_button and user_id is not None:
    with st.spinner('추천 결과를 생성하는 중...'):
                
                # 사용자 정보
                st.markdown('<h2 class="sub-header">📊 사용자 정보</h2>', unsafe_allow_html=True)
                user_info = get_user_info(user_id, users_df)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("성별", user_info['gender'].values[0])
                with col2:
                    st.metric("나이", user_info['age'].values[0])
                with col3:
                    st.metric("직업", user_info['occupation'].values[0])
                with col4:
                    st.metric("지역", user_info['zip'].values[0])
                
                # 과거 시청 이력
                st.markdown('<h2 class="sub-header">🎥 과거 선호 영화 (평점 4점 이상)</h2>', unsafe_allow_html=True)
                user_interactions = get_user_past_interactions(user_id, ratings_df, movies_df)
                
                if len(user_interactions) > 0:
                    st.dataframe(
                        user_interactions[['movie_id', 'title', 'genres', 'rating', 'timestamp']],
                        use_container_width=True,
                        hide_index=True
                    )
                    st.info(f"총 {len(user_interactions)}개의 영화를 선호했습니다.")
                else:
                    st.warning("평점 4점 이상의 영화가 없습니다.")
                
                # 추천 결과
                st.markdown('<h2 class="sub-header">⭐ 추천 결과</h2>', unsafe_allow_html=True)
                
                if model_choice == "두 모델 비교":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🔹 AutoInt 모델")
                        recommendations_1 = get_recommendations(
                            user_id, user_non_seen_dict, users_df, movies_df,
                            r_year, r_month, model_autoint, label_encoders, predict_autoint
                        )
                        st.dataframe(
                            recommendations_1[['movie_id', 'title', 'genres']],
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with col2:
                        st.markdown("#### 🔸 AutoInt+MLP 모델")
                        recommendations_2 = get_recommendations(
                            user_id, user_non_seen_dict, users_df, movies_df,
                            r_year, r_month, model_autointmlp, label_encoders, predict_autointmlp
                        )
                        st.dataframe(
                            recommendations_2[['movie_id', 'title', 'genres']],
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    model = model_autoint if model_choice == "AutoInt" else model_autointmlp
                    predict_fn = predict_autoint if model_choice == "AutoInt" else predict_autointmlp
                    
                    recommendations = get_recommendations(
                        user_id, user_non_seen_dict, users_df, movies_df,
                        r_year, r_month, model, label_encoders, predict_fn
                    )
                    
                    st.dataframe(
                        recommendations[['movie_id', 'title', 'genres']],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.success(f"✅ {model_choice} 모델로 {len(recommendations)}개의 영화를 추천했습니다!")

else:
    # 초기 화면
    if user_id is None:
        st.info("👈 왼쪽 사이드바에서 사용자 ID를 입력하고 '추천 결과 보기' 버튼을 눌러주세요.")
    
    # 데이터셋 통계
    st.markdown('<h2 class="sub-header">📈 데이터셋 통계</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("전체 사용자 수", f"{len(users_df):,}명")
    with col2:
        st.metric("전체 영화 수", f"{len(movies_df):,}개")
    with col3:
        st.metric("전체 평점 수", f"{len(ratings_df):,}개")