import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load datasets
matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')

# Merge to get total runs for each match
total_runs_df = deliveries.groupby('match_id').sum()['total_runs'].reset_index()
total_runs_df.columns = ['id', 'total_runs']
df = matches.merge(total_runs_df, on='id')

# Filter necessary columns
df = df[['city', 'winner', 'team1', 'team2', 'toss_winner', 'toss_decision', 'result', 'dl_applied', 'win_by_runs', 'win_by_wickets', 'id', 'total_runs']]
df = df[df['result'] == 'normal']

# Select final matches and deliveries
final_df = df[['city', 'team1', 'team2', 'toss_winner', 'toss_decision', 'winner', 'total_runs']]

# Simplify team names
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals']

final_df = final_df[final_df['team1'].isin(teams) & final_df['team2'].isin(teams)]

# Rename columns and set up model input
deliveries_df = deliveries.merge(matches[['id', 'city', 'team1', 'team2', 'winner']], left_on='match_id', right_on='id')
deliveries_df = deliveries_df[deliveries_df['batting_team'].isin(teams) & deliveries_df['bowling_team'].isin(teams)]

# Add cumulative features
deliveries_df['current_score'] = deliveries_df.groupby('match_id')['total_runs'].cumsum()
deliveries_df['ball_number'] = deliveries_df.groupby('match_id').cumcount() + 1
deliveries_df['balls_left'] = 120 - deliveries_df['ball_number']
deliveries_df['wickets'] = deliveries_df.groupby('match_id')['player_dismissed'].transform(lambda x: x.notnull().cumsum())
deliveries_df['wickets_left'] = 10 - deliveries_df['wickets']

# Calculate crr and rrr
deliveries_df['crr'] = deliveries_df['current_score'] / (deliveries_df['ball_number'] / 6)
deliveries_df['rrr'] = (deliveries_df['total_runs_y'] - deliveries_df['current_score']) / (deliveries_df['balls_left'] / 6)

# Mock player impact features
np.random.seed(42)
deliveries_df['batsman_sr'] = np.random.normal(130, 10, size=len(deliveries_df))
deliveries_df['bowler_econ'] = np.random.normal(7, 1, size=len(deliveries_df))

# Determine target team
X = deliveries_df[['batting_team', 'bowling_team', 'city', 'total_runs_y', 'balls_left', 'wickets_left', 'crr', 'rrr', 'batsman_sr', 'bowler_econ']]
y = deliveries_df['batting_team'] == deliveries_df['winner']
y = y.astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Preprocessing pipeline
categorical_cols = ['batting_team', 'bowling_team', 'city']
numerical_cols = ['total_runs_y', 'balls_left', 'wickets_left', 'crr', 'rrr', 'batsman_sr', 'bowler_econ']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
], remainder='passthrough')

# Model pipeline
model = RandomForestClassifier(n_estimators=100, random_state=42)
pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
pipe.fit(X_train, y_train)

# Save the model
with open('pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print("Model training complete. Pipeline saved as pipe.pkl")
