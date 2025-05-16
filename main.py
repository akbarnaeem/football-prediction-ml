from src.data_loader import load_data, prepare_team_stats
from src.model import train_model
from src.predict import make_predictions, predict_user_match

def main():
    print("⚽️ Football Match Outcome Predictor")

    data = load_data("all_games.csv")
    team_stats = prepare_team_stats(data)
    
    print("\n📋 Available Teams:")
    for team in sorted(team_stats.index):
        print("•", team)

    X_train, X_test, y_train, y_test = train_model(data, team_stats)
    
    model = train_model(data, team_stats, return_model=True)
    make_predictions(model, X_test, y_test)

    predict_user_match(model, team_stats)

if __name__ == "__main__":
    main()
