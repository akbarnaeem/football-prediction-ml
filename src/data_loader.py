import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    try:
        data = pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        print("❌ Could not find 'all_games.csv'. Please make sure it's in the project directory.")
        exit()
    return data.dropna()

def prepare_team_stats(data):
    home_stats = data.groupby('HomeTeam').agg({
        'FTHG': 'mean',
        'HS': 'mean'
    }).rename(columns={'FTHG': 'AvgHomeGoals', 'HS': 'AvgHomeShots'})

    away_stats = data.groupby('AwayTeam').agg({
        'FTAG': 'mean',
        'AS': 'mean'
    }).rename(columns={'FTAG': 'AvgAwayGoals', 'AS': 'AvgAwayShots'})

    team_stats = pd.merge(home_stats, away_stats, left_index=True, right_index=True, how='outer')
    team_stats.fillna(0, inplace=True)

    return team_stats
