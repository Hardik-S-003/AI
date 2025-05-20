import streamlit as st
import pandas as pd
import pickle
import time
import plotly.graph_objects as go

# Load Model
with open('pipe.pkl', 'rb') as file:
    pipe = pickle.load(file)

# Load Player Stats (Mock example)
batsman_stats = {'Virat Kohli': {'strike_rate': 135}, 'Jos Buttler': {'strike_rate': 145}}
bowler_stats = {'Jasprit Bumrah': {'economy': 6.7}, 'Yuzvendra Chahal': {'economy': 7.2}}

# Set Page Configuration
st.set_page_config(page_title="IPL Win Predictor", layout="wide")

# Theme Toggle
theme = st.sidebar.radio("Choose Theme", ("Light", "Dark"))
if theme == "Dark":
    st.markdown("""
        <style>
        body { background-color: #1E1E1E; color: white; }
        .title { color: #FFD700; }
        </style>
    """, unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("\U0001F3CF IPL Win Predictor")
st.sidebar.image("https://www.iplt20.com/assets/images/IPL_LOGO_CORPORATE_2024.png", width=300)

# Dropdown Options
teams = sorted([
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
])
cities = sorted([
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Ahmedabad', 'Pune', 'Bengaluru'
])
players = sorted(set(list(batsman_stats.keys()) + list(bowler_stats.keys())))

# Header
st.markdown("<h1 class='title'>\U0001F3C6 IPL Win Predictor</h1>", unsafe_allow_html=True)

# Match Details Input
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('\U0001F3CF Select Batting Team', teams)
with col2:
    bowling_team = st.selectbox('\U0001F3AF Select Bowling Team', teams)

if batting_team == bowling_team:
    st.error("Batting team and Bowling team cannot be the same!")
    st.stop()

selected_city = st.selectbox('\U0001F30D Select Match City', cities)

# What-if Analysis Sliders
target = st.slider('\U0001F3AF Target Score', 0, 300, 150)
col3, col4, col5 = st.columns(3)
with col3:
    score = st.slider('\U0001F3CF Current Score', 0, target, 100)
with col4:
    wickets = st.slider('\u274C Wickets Lost', 0, 9, 3)
with col5:
    overs = st.slider('\u23F3 Overs Completed', 0, 20, 10)

# Player Selection
col6, col7 = st.columns(2)
with col6:
    batsman = st.selectbox("Current Batsman", players)
with col7:
    bowler = st.selectbox("Current Bowler", players)

# Video Panel
st.video("https://www.youtube.com/watch?v=7ELtVgGhxWk")

# Predict Button
if st.button('\u26A1 Predict Winning Probability'):
    with st.spinner('Calculating...'):
        time.sleep(2)

    runs_left = max(target - score, 0)
    balls_left = max(120 - (overs * 6), 1)
    remaining_wickets = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0

    batsman_sr = batsman_stats.get(batsman, {}).get('strike_rate', 120)
    bowler_econ = bowler_stats.get(bowler, {}).get('economy', 8.0)

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [remaining_wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr],
        'batsman_sr': [batsman_sr],
        'bowler_econ': [bowler_econ]
    })

    result = pipe.predict_proba(input_df)
    batting_prob = round(result[0][1] * 100)
    bowling_prob = round(result[0][0] * 100)

    st.markdown("<h2 style='text-align: center;'>\U0001F3C6 Winning Probability</h2>", unsafe_allow_html=True)
    st.success(f"{batting_team}: {batting_prob}%")
    st.error(f"{bowling_team}: {bowling_prob}%")
    st.progress(batting_prob / 100)

    # Match Progress Line Chart (Static Mock)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, overs+1)), y=[i*(batting_prob/overs) for i in range(1, overs+1)], mode='lines+markers', name='Batting Prob'))
    fig.update_layout(title='Match Progress Timeline', xaxis_title='Overs', yaxis_title='Win Probability (%)')
    st.plotly_chart(fig)
