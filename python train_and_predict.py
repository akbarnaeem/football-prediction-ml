import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_predict(data_path):
    data = pd.read_csv(data_path)

    home_teams = data['HomeTeam'].unique()
    away_teams = data['AwayTeam'].unique()
    all_teams = sorted(set(home_teams) | set(away_teams))
    
    home_stats = data.groupby('HomeTeam').agg({
        'FTHG': 'mean',
        'HS': 'mean',
    }).rename(columns={
        'FTHG': 'AvgHomeGoals',
        'HS': 'AvgHomeShots'
    })

    away_stats = data.groupby('AwayTeam').agg({
        'FTAG': 'mean',
        'AS': 'mean',
    }).rename(columns={
        'FTAG': 'AvgAwayGoals',
        'AS': 'AvgAwayShots'
    })

    team_stats = pd.merge(home_stats, away_stats, left_index=True, right_index=True, how='outer')
    team_stats.fillna(0, inplace=True)

    data['HomeTeamStrength'] = data['HomeTeam'].map(team_stats['AvgHomeGoals'])
    data['AwayTeamDefense'] = data['AwayTeam'].map(team_stats['AvgAwayGoals'])
    data['HomeShots'] = data['HomeTeam'].map(team_stats['AvgHomeShots'])
    data['AwayShots'] = data['AwayTeam'].map(team_stats['AvgAwayShots'])
    data.dropna(subset=['HomeTeamStrength', 'AwayTeamDefense', 'HomeShots', 'AwayShots'], inplace=True)

    outcome_map = {'H': 0, 'D': 1, 'A': 2}
    data['Label'] = data['FTR'].map(outcome_map)

    X = data[['HomeTeamStrength', 'AwayTeamDefense', 'HomeShots', 'AwayShots']]
    y = data['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    home_team = "Arsenal"
    away_team = "Nice"

    new_match = pd.DataFrame({
        'HomeTeamStrength': [team_stats.loc[home_team, 'AvgHomeGoals']], 
        'AwayTeamDefense': [team_stats.loc[away_team, 'AvgAwayGoals']],   
        'HomeShots': [team_stats.loc[home_team, 'AvgHomeShots']],
        'AwayShots': [team_stats.loc[away_team, 'AvgAwayShots']],
    })

    prediction = model.predict(new_match)
    label = prediction[0]

    if label == 0:
        result = f"🏠 {home_team} wins (Home Win)"
    elif label == 1:
        result = "⚖ Draw"
    else:
        result = f"✈ {away_team} wins (Away Win)"

    print(f"\n📊 Prediction: {home_team} vs {away_team} → {result}")

data_path = 'all_games_data.csv'
train_and_predict(data_path)
