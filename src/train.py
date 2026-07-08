"""
Training pipeline: read config -> load data -> preprocess -> build sequences
-> train RNN + LSTM -> save models.

Run directly with: python -m src.train
"""

import os

from src.config_reader import load_config
from src.data_loader import load_data, read_raw_data
from src.models import build_lstm_model, build_rnn_model
from src.preprocessing import run_preprocessing_pipeline


def main():
    config = load_config()

    ingestion_cfg = config["data_ingestion"]
    transform_cfg = config["data_transformation"]
    training_cfg = config["model_training"]

    os.makedirs(transform_cfg["root_dir"], exist_ok=True)
    os.makedirs(training_cfg["root_dir"], exist_ok=True)

    # 1. Load raw data
    df = read_raw_data(ingestion_cfg["raw_data_path"])

    # 2. Feature engineering + normalization (scaler saved for later inverse-transform)
    df = run_preprocessing_pipeline(
        df,
        target_col=transform_cfg["target_column"],
        scaler_path=transform_cfg["scaler_path"],
    )
    df.to_csv(transform_cfg["transformed_data_path"])

    # 3. Build sliding-window sequences
    X_train, y_train, X_test, y_test = load_data(
        df,
        seq_len=transform_cfg["seq_len"],
        target_col=transform_cfg["target_column"],
        split_ratio=transform_cfg["split_ratio"],
    )
    print("X_train.shape =", X_train.shape)
    print("y_train.shape =", y_train.shape)
    print("X_test.shape  =", X_test.shape)
    print("y_test.shape  =", y_test.shape)

    n_features = X_train.shape[2]
    input_shape = (X_train.shape[1], n_features)

    params = training_cfg["params"]

    # 4. Train RNN
    print("\nTraining SimpleRNN model...")
    rnn_model = build_rnn_model(input_shape, units=params["units"], dropout=params["dropout"])
    rnn_model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"])
    rnn_model.save(training_cfg["rnn_model_path"])
    print(f"Saved RNN model to {training_cfg['rnn_model_path']}")

    # 5. Train LSTM
    print("\nTraining LSTM model...")
    lstm_model = build_lstm_model(input_shape, units=params["units"], dropout=params["dropout"])
    lstm_model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"])
    lstm_model.save(training_cfg["lstm_model_path"])
    print(f"Saved LSTM model to {training_cfg['lstm_model_path']}")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    main()
