"""
Model architectures, mirroring dm-practicle.ipynb:
- build_rnn_model(): 3-layer stacked SimpleRNN
- build_lstm_model(): 3-layer stacked LSTM
Both use the same shape (units, dropout) and end in a Dense(1) regression output.
"""

from keras.layers import Dense, Dropout, SimpleRNN, LSTM
from keras.models import Sequential


def build_rnn_model(input_shape, units: int = 40, dropout: float = 0.15) -> Sequential:
    """Build a 3-layer stacked SimpleRNN regression model."""
    model = Sequential()

    model.add(SimpleRNN(units, activation="tanh", return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))

    model.add(SimpleRNN(units, activation="tanh", return_sequences=True))
    model.add(Dropout(dropout))

    model.add(SimpleRNN(units, activation="tanh", return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="MSE")
    return model


def build_lstm_model(input_shape, units: int = 40, dropout: float = 0.15) -> Sequential:
    """Build a 3-layer stacked LSTM regression model."""
    model = Sequential()

    model.add(LSTM(units, activation="tanh", recurrent_activation="sigmoid",
                    return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))

    model.add(LSTM(units, activation="tanh", recurrent_activation="sigmoid", return_sequences=True))
    model.add(Dropout(dropout))

    model.add(LSTM(units, activation="tanh", recurrent_activation="sigmoid", return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="MSE")
    return model
