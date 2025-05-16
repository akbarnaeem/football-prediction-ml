from sklearn.metrics import accuracy_score, confusion_matrix

def make_predictions(model, X_test, y_test):
    print("\n🔍 Evaluating model...")
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"✅ Model Accuracy: {acc:.2f}")

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

def predict_user_match(model, team_stats):
    home_team = input("\nEnter Home Team: ").strip()
    away_team = input("Enter Away Team: ").strip()

    if home_team not in team_stats.index or away_team not in team_stats.index:
        print("❌ One or both team names are not in the available stats.")
        return

    new_match = {
        'HomeTeamStrength': team_stats.loc[home_team, 'AvgHomeGoals'],
        'AwayTeamDefense': team_stats.loc[away_team, 'AvgAwayGoals'],
        'HomeShots': team_stats.loc[home_team, 'AvgHomeShots'],
        'AwayShots': team_stats.loc[away_team, 'AvgAwayShots'],
    }

    input_df = pd.DataFrame([new_match])
    prediction = model.predict(input_df)[0]

    label_map = {0: f"🏠 {home_team} wins (Home Win)",
                 1: "⚖ Draw",
                 2: f"✈ {away_team} wins (Away Win)"}

    print(f"\n📊 Prediction: {home_team} vs {away_team} → {label_map[prediction]}")
