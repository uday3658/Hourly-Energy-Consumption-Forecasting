"""
Single entry point that runs the whole pipeline end to end:
ingestion -> preprocessing -> sequence building -> train RNN + LSTM
-> evaluate -> feature importance -> save all outputs.

Run with: python main.py
"""

import os

from keras.models import load_model

from src.config_reader import load_config
from src.data_loader import load_data, read_raw_data
from src.evaluate import compute_metrics, plot_predictions, save_scores
from src.feature_importance import permutation_importance, plot_feature_importance
from src.models import build_lstm_model, build_rnn_model
from src.preprocessing import run_preprocessing_pipeline


def main():
    config = load_config()

    ingestion_cfg = config["data_ingestion"]
    transform_cfg = config["data_transformation"]
    training_cfg = config["model_training"]
    eval_cfg = config["model_evaluation"]

    for cfg in (ingestion_cfg, transform_cfg, training_cfg, eval_cfg):
        os.makedirs(cfg["root_dir"], exist_ok=True)

    # ---- 1. Ingestion ----
    df = read_raw_data(ingestion_cfg["raw_data_path"])

    # ---- 2. Preprocessing / feature engineering ----
    df = run_preprocessing_pipeline(
        df,
        target_col=transform_cfg["target_column"],
        scaler_path=transform_cfg["scaler_path"],
    )
    df.to_csv(transform_cfg["transformed_data_path"])

    # ---- 3. Sequence building ----
    X_train, y_train, X_test, y_test = load_data(
        df,
        seq_len=transform_cfg["seq_len"],
        target_col=transform_cfg["target_column"],
        split_ratio=transform_cfg["split_ratio"],
    )

    n_features = X_train.shape[2]
    input_shape = (X_train.shape[1], n_features)
    params = training_cfg["params"]

    # ---- 4. Train models ----
    print("Training SimpleRNN...")
    rnn_model = build_rnn_model(input_shape, units=params["units"], dropout=params["dropout"])
    rnn_model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"])
    rnn_model.save(training_cfg["rnn_model_path"])

    print("Training LSTM...")
    lstm_model = build_lstm_model(input_shape, units=params["units"], dropout=params["dropout"])
    lstm_model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"])
    lstm_model.save(training_cfg["lstm_model_path"])

    # ---- 5. Evaluate ----
    rnn_predictions = rnn_model.predict(X_test)
    lstm_predictions = lstm_model.predict(X_test)

    scores = {
        "rnn": compute_metrics(y_test, rnn_predictions),
        "lstm": compute_metrics(y_test, lstm_predictions),
    }
    save_scores(scores, eval_cfg["scores_file_path"])
    print("Scores:", scores)

    plot_predictions(y_test, rnn_predictions, "Predictions made by SimpleRNN model",
                      save_path=eval_cfg["rnn_prediction_plot_path"])
    plot_predictions(y_test, lstm_predictions, "Predictions made by LSTM model",
                      save_path=eval_cfg["lstm_prediction_plot_path"])

    # ---- 6. Feature importance ----
    feature_names = df.columns.tolist()
    rnn_imp = permutation_importance(rnn_model, X_test, y_test, feature_names)
    lstm_imp = permutation_importance(lstm_model, X_test, y_test, feature_names)
    plot_feature_importance(rnn_imp, lstm_imp, save_path=eval_cfg["feature_importance_plot_path"])

    print("Pipeline complete. Outputs saved under 'outputs/' and models under 'models/'.")


if __name__ == "__main__":
    main()
