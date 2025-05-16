from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(data, team_stats, return_model=False):
    
    data = data.copy()
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

    return (model if return_model else (X_train, X_test, y_train, y_test))
