import streamlit as st
import pandas as pd
import pickle
import time

# Load Model
with open('pipe.pkl', 'rb') as file:
    pipe = pickle.load(file)

# Set Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="IPL Win Predictor", layout="wide")

# Custom CSS for Styling
st.markdown(
    """
    <style>
    body { background-color: #1E1E1E; color: white; font-family: 'Arial'; }
    .title { text-align: center; font-size: 36px; font-weight: bold; color: #FFD700; }
    .sidebar .sidebar-content { background-color: #121212; }
    .block-container { padding-top: 1rem; }
    .stButton>button { background-color: #FF5733; color: white; font-size: 20px; }
    .stButton>button:hover { background-color: #C70039; }
    .st-emotion-cache-13lnz3c { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
st.sidebar.title("🏏 IPL Win Predictor")
st.sidebar.image("https://www.iplt20.com/assets/images/IPL_LOGO_CORPORATE_2024.png", width=300)

# Dropdown Options
teams = sorted([
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
])
cities = sorted([
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
])

# Header
st.markdown("<h1 class='title'>🏆 IPL Win Predictor</h1>", unsafe_allow_html=True)

# Match Details Input
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('🏏 Select Batting Team', teams, key="batting")
with col2:
    bowling_team = st.selectbox('🎯 Select Bowling Team', teams, key="bowling")

# Error Message if Same Teams are Selected
if batting_team == bowling_team:
    st.error("⚠️ Batting team and Bowling team cannot be the same! Please select different teams.")
    st.stop()  # Stops execution until user selects different teams

selected_city = st.selectbox('🌍 Select Match City', cities)

# Target Runs Input
target = st.slider('🎯 Target Score', min_value=0, max_value=300, step=1)
score_max = max(target, 1)

# Current Match Statistics
col3, col4, col5 = st.columns(3)
with col3:
    score = st.slider('🏏 Current Score', min_value=0, max_value=score_max, step=1)
with col4:
    wickets = st.slider('❌ Wickets Lost', min_value=0, max_value=9, step=1)
with col5:
    overs = st.slider('⏳ Overs Completed', min_value=0, max_value=20, step=1)

# Predict Button
if st.button('⚡ Predict Winning Probability'):
    with st.spinner('🏏 Calculating Probabilities...'):
        time.sleep(2)

    # Calculate Remaining Stats
    runs_left = max(target - score, 0)
    balls_left = max(120 - (overs * 6), 1)
    remaining_wickets = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0

    # Prepare Input Data
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [remaining_wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    st.write("Input Data:", input_df)

    try:
        # Predict Win Probability
        result = pipe.predict_proba(input_df)
        batting_prob = round(result[0][1] * 100)
        bowling_prob = round(result[0][0] * 100)

        st.markdown("<h2 style='text-align: center; color: #FFD700;'>🏆 Winning Probability</h2>", unsafe_allow_html=True)
        st.success(f"{batting_team}: {batting_prob}%")
        st.error(f"{bowling_team}: {bowling_prob}%")

        st.progress(batting_prob / 100)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
