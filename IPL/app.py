import streamlit as st
import pandas as pd
import pickle
import time
import plotly.graph_objects as go
import json
from datetime import datetime
import numpy as np

# Load Model
try:
    with open('pipe.pkl', 'rb') as file:
        pipe = pickle.load(file)
    # Validate model type
    if not hasattr(pipe, 'predict_proba'):
        raise ValueError("Model does not support probability predictions")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Load Player Stats
try:
    with open('player_stats.json', 'r') as file:
        player_stats = json.load(file)
except Exception as e:
    st.error(f"Error loading player stats: {str(e)}")
    st.stop()

# Initialize session state
if 'score_history' not in st.session_state:
    st.session_state.score_history = []
if 'prob_history' not in st.session_state:
    st.session_state.prob_history = []

# Set Page Configuration
st.set_page_config(page_title="IPL Win Predictor", layout="wide")

# Theme Toggle with Enhanced Styling
def set_theme(theme):
    if theme == "Dark":
        st.markdown("""
            <style>
            .stApp {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            .stSelectbox, .stSlider { 
                background-color: #2E2E2E;
                color: white;
            }
            .stSelectbox>div>div,
            .stSlider>div>div,
            .stTextInput>div>div,
            .stNumberInput>div>div {
                background-color: #2D2D2D;
                border: 1px solid #404040;
                border-radius: 4px;
                color: #FFFFFF;
                box-shadow: 0 1px 3px rgba(0,0,0,0.3);
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            .stProgress .st-progress-bar {
                background-color: #4CAF50;
            }
            .title {
                color: #FFD700;
                text-align: center;
                padding: 20px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .stMarkdown, .stInfo, .stSuccess, .stWarning, .stError {
                background-color: #2D2D2D;
                border-radius: 4px;
                padding: 8px;
                margin: 4px 0;
            }
            .streamlit-expanderHeader {
                color: #FFFFFF !important;
                background-color: #2D2D2D !important;
            }
            .stSlider>div>div>div>div {
                background-color: #4CAF50;
            }
            /* Info boxes in dark mode */
            .stInfo > div {
                background-color: #2C3E50 !important;
                color: #90CAF9 !important;
            }
            .stSuccess > div {
                background-color: #1B5E20 !important;
                color: #A5D6A7 !important;
            }
            .stWarning > div {
                background-color: #E65100 !important;
                color: #FFCC80 !important;
            }
            .stError > div {
                background-color: #B71C1C !important;
                color: #EF9A9A !important;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp {
                background-color: #FFFFFF;
                color: #2C3E50;
            }
            /* Darken logo in light mode */
            .stSidebar img {
                filter: brightness(0.7) contrast(1.2);
            }
            .stButton>button {
                background-color: #1E88E5;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #1976D2;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            /* Streamlit Components Styling */
            .stTextInput>div>div>input,
            .stNumberInput>div>div>input {
                background-color: #F8F9FA;
                border: 1px solid #E9ECEF;
                border-radius: 4px;
                color: #2C3E50;
            }
            .stSelectbox>div>div,
            .stSlider>div>div,
            .stTextInput>div>div,
            .stNumberInput>div>div {
                background-color: #E8F0FE;
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                color: #2C3E50;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .stMarkdown, .stInfo, .stSuccess, .stWarning, .stError {
                color: #2C3E50;
                background-color: #F8F9FA;
                border-radius: 4px;
                padding: 8px;
                margin: 4px 0;
            }
            .stProgress .st-progress-bar {
                background-color: #1E88E5;
                height: 6px;
                border-radius: 3px;
            }
            .stSidebar {
                background-color: #F8F9FA;
                border-right: 1px solid #E9ECEF;
            }
            .stSidebar .stMarkdown {
                color: #2C3E50;
            }
            /* Headers and Text */
            h1, h2, h3, h4, h5, h6, p {
                color: #2C3E50 !important;
            }
            .title {
                color: #1E88E5;
                text-align: center;
                padding: 20px;
                font-weight: 600;
                text-shadow: none;
                font-size: 2.5rem;
            }
            /* Info boxes */
            .stInfo > div {
                background-color: #E3F2FD;
                color: #1565C0;
                border: 1px solid #90CAF9;
            }
            .stSuccess > div {
                background-color: #E8F5E9;
                color: #2E7D32;
                border: 1px solid #A5D6A7;
            }
            .stWarning > div {
                background-color: #FFF3E0;
                color: #F57C00;
                border: 1px solid #FFCC80;
            }
            .stError > div {
                background-color: #FFEBEE;
                color: #C62828;
                border: 1px solid #EF9A9A;
            }
            /* Expander */
            .streamlit-expanderHeader {
                background-color: #F8F9FA;
                color: #2C3E50;
                border-radius: 4px;
            }
            /* Select Slider */
            .stSelectSlider>div>div {
                background-color: #F8F9FA;
                border: 1px solid #E9ECEF;
            }
            </style>
        """, unsafe_allow_html=True)

# Add theme selection in sidebar with styled radio buttons
st.markdown("""
    <style>
    /* Custom styling for radio buttons */
    .stRadio [role=radiogroup] {
        gap: 0;
        padding: 10px;
        border-radius: 8px;
        background-color: #f0f2f6;
    }
    
    .stRadio [role=radio] {
        width: 100%;
        padding: 10px 20px;
        margin: 0;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    
    .stRadio [role=radio][aria-checked="true"] {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Theme selector as toggle using checkbox
theme = "Dark" if st.sidebar.checkbox("🌙 Dark Mode", value=False) else "Light"
set_theme(theme)

# Sidebar Navigation
st.sidebar.title("\U0001F3CF IPL Win Predictor")
st.sidebar.image("https://www.iplt20.com/assets/images/IPL_LOGO_CORPORATE_2024.png", width=300)

# Dropdown Options and Team Mappings
teams = sorted([
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
])

# Team name to abbreviation mapping
team_abbr = {
    'Sunrisers Hyderabad': 'SRH',
    'Mumbai Indians': 'MI',
    'Royal Challengers Bangalore': 'RCB',
    'Kolkata Knight Riders': 'KKR',
    'Kings XI Punjab': 'PBKS',
    'Chennai Super Kings': 'CSK',
    'Rajasthan Royals': 'RR',
    'Delhi Capitals': 'DC'
}
cities = sorted([
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Ahmedabad', 'Pune', 'Bengaluru'
])
players = sorted(set(list(player_stats.keys())))

# Header
st.markdown("<h1 class='title'>\U0001F3C6 IPL Win Predictor</h1>", unsafe_allow_html=True)

# Match Details Input
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('\U0001F3CF Select Batting Team', teams)
    try:
        if batting_team:
            st.image(f"logo/{team_abbr[batting_team]}.png", width=100)
    except Exception as e:
        st.warning(f"Team logo not found for {batting_team}")
with col2:
    # Filter out batting team from bowling options
    bowling_teams = [team for team in teams if team != batting_team]
    bowling_team = st.selectbox('\U0001F3AF Select Bowling Team', bowling_teams)
    try:
        if bowling_team:
            st.image(f"logo/{team_abbr[bowling_team]}.png", width=100)
    except Exception as e:
        st.warning(f"Team logo not found for {bowling_team}")

if not batting_team or not bowling_team:
    st.error("Please select both teams!")
    st.stop()

selected_city = st.selectbox('\U0001F30D Select Match City', cities)
if not selected_city:
    st.error("Please select a match city!")
    st.stop()

# What-if Analysis Sliders
target = st.slider('\U0001F3AF Target Score', 0, 300, 150)
col3, col4, col5 = st.columns(3)
with col3:
    score = st.slider('\U0001F3CF Current Score', 0, target, 100)
with col4:
    wickets = st.slider('\u274C Wickets Lost', 0, 9, 3)
with col5:
    overs = st.slider('\u23F3 Overs Completed', 0, 20, 10)

# Advanced What-if Analysis
with st.expander("Advanced Match Factors"):
    col8, col9 = st.columns(2)
    with col8:
        pitch_condition = st.select_slider(
            'Pitch Condition',
            options=['Very Bowling Friendly', 'Bowling Friendly', 'Neutral', 'Batting Friendly', 'Very Batting Friendly'],
            value='Neutral'
        )
        dew_factor = st.slider('Dew Impact (0-10)', 0, 10, 5)
    with col9:
        weather_condition = st.selectbox(
            'Weather Condition',
            ['Clear', 'Partly Cloudy', 'Overcast', 'Light Rain', 'Humid']
        )
        pressure_factor = st.slider('Match Pressure (0-10)', 0, 10, 5)

# Predict Button
# Predict Button
if st.button('\u26A1 Predict Winning Probability'):
    # Validate match conditions
    if score >= target:
        st.error("Current score cannot be greater than or equal to target!")
        st.stop()
    
    if overs >= 20:
        st.error("Overs cannot exceed 20!")
        st.stop()
        
    if wickets >= 10:
        st.error("Wickets lost cannot exceed 10!")
        st.stop()

    with st.spinner('Calculating...'):
        time.sleep(2)

    runs_left = target - score
    balls_left = 120 - (overs * 6)
    remaining_wickets = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6 / balls_left) if balls_left > 0 else float('inf')  # Handle division by zero

    # Convert pitch condition to numerical factor
    pitch_factors = {
        'Very Bowling Friendly': 0.7,
        'Bowling Friendly': 0.85,
        'Neutral': 1.0,
        'Batting Friendly': 1.15,
        'Very Batting Friendly': 1.3
    }
    pitch_factor = pitch_factors.get(pitch_condition, 1.0)
    
    # Calculate weather impact
    weather_factors = {
        'Clear': 1.0,
        'Partly Cloudy': 0.95,
        'Overcast': 0.9,
        'Light Rain': 0.8,
        'Humid': 1.1
    }
    weather_factor = weather_factors.get(weather_condition, 1.0)
    
    # Normalize dew and pressure factors
    dew_adjustment = min(max(dew_factor / 10, 0), 1)
    pressure_adjustment = min(max(pressure_factor / 10, 0), 1)
    
    # Adjust player performance based on conditions
    # Create input DataFrame with all features including advanced factors
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [float(runs_left)],
        'balls_left': [float(balls_left)],
        'wickets': [float(remaining_wickets)],
        'total_runs_x': [float(target)],
        'crr': [float(crr)],
        'rrr': [float(rrr)],
        'pitch_factor': [float(pitch_factor)],
        'weather_factor': [float(weather_factor)],
        'dew_factor': [float(dew_adjustment)],
        'pressure_factor': [float(pressure_adjustment)]
    })

    # Validate input ranges
    range_validations = {
        'runs_left': (0, 300),
        'balls_left': (1, 120),
        'wickets': (0, 10),
        'total_runs_x': (0, 300),
        'crr': (0, 20),
        'rrr': (0, float('inf')),
        'pitch_factor': (0.7, 1.3),
        'weather_factor': (0.8, 1.1),
        'dew_factor': (0, 1),
        'pressure_factor': (0, 1)
    }

    for col, (min_val, max_val) in range_validations.items():
        if not (min_val <= input_df[col].iloc[0] <= max_val):
            st.error(f"Invalid value for {col}: {input_df[col].iloc[0]}")
            st.stop()

    # Get prediction with all factors included
    try:
        result = pipe.predict_proba(input_df)
        # Model now directly uses the advanced factors
        batting_prob = round(result[0][1] * 100)
        bowling_prob = round(result[0][0] * 100)
        
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        print("Debug - Input DataFrame:")
        print(input_df)
        st.stop()
        
    # Store history for timeline
    if len(st.session_state.score_history) == 0 or st.session_state.score_history[-1] != score:
        st.session_state.score_history.append(score)
        st.session_state.prob_history.append(batting_prob)

    # Determine predicted winner
    predicted_winner = batting_team if batting_prob > bowling_prob else bowling_team
    win_probability = max(batting_prob, bowling_prob)
    
    # Display prediction header
    st.markdown("<h2 style='text-align: center;'>\U0001F3C6 Match Prediction</h2>", unsafe_allow_html=True)
    
    # Display winner with dynamic styling
    winner_bg_color = "#E8F5E9" if theme == "Light" else "#1E3620"
    winner_text_color = "#2C3E50" if theme == "Light" else "#FFFFFF"
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background-color: {winner_bg_color}; 
         border-radius: 10px; margin: 10px 0; color: {winner_text_color}; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
        <h3 style='margin-bottom: 10px;'>Predicted Winner</h3>
        <div style='font-size: 2em; font-weight: bold; margin: 10px 0;'>{predicted_winner}</div>
        <div style='font-size: 1.2em;'>Win Confidence: {win_probability}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show detailed probabilities in collapsible section
    with st.expander("View Detailed Win Probabilities"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{batting_team}", f"{batting_prob}%")
        with col2:
            st.metric(f"{bowling_team}", f"{bowling_prob}%")
        st.progress(batting_prob / 100)

    # Add match situation analysis
    st.markdown("### Match Situation Analysis")
    
    # Required run rate analysis
    if rrr > 15:
        st.error(f"Required Run Rate ({rrr:.2f}) is very challenging!")
    elif rrr > 12:
        st.warning(f"Required Run Rate ({rrr:.2f}) is high but achievable")
    elif rrr > 9:
        st.info(f"Required Run Rate ({rrr:.2f}) is manageable")
    else:
        st.success(f"Required Run Rate ({rrr:.2f}) is comfortable")

    # Wickets analysis
    if remaining_wickets >= 7:
        st.success(f"Good batting depth with {remaining_wickets} wickets remaining")
    elif remaining_wickets >= 4:
        st.info(f"Need to be cautious with {remaining_wickets} wickets remaining")
    else:
        st.warning(f"Critical situation with only {remaining_wickets} wickets remaining")

    # Pressure situations
    if pressure_factor > 7:
        st.error("High pressure situation!")
        if batting_prob > 60:
            st.info("But the batting team is still in a strong position")
    elif pressure_factor > 5:
        st.warning("Moderate pressure situation")
    
    # Match turning point indicators
    if abs(batting_prob - 50) < 10:
        st.info("💫 This is a crucial phase of the game!")
    
    # Show timeline visualization
    if len(st.session_state.score_history) > 0:
        fig = go.Figure()
        
        # Win probability line
        fig.add_trace(go.Scatter(
            x=list(range(len(st.session_state.score_history))),
            y=st.session_state.prob_history,
            mode='lines+markers',
            name='Win Probability',
            line=dict(color='#4CAF50', width=3)
        ))
        
        # Score progression
        fig.add_trace(go.Scatter(
            x=list(range(len(st.session_state.score_history))),
            y=st.session_state.score_history,
            mode='lines+markers',
            name='Score Progression',
            yaxis='y2',
            line=dict(color='#1E88E5', width=3)
        ))
        
        # Layout updates
        fig.update_layout(
            title='Match Progress Timeline',
            xaxis_title='Updates',
            yaxis_title='Win Probability (%)',
            yaxis2=dict(
                title='Score',
                overlaying='y',
                side='right'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)
