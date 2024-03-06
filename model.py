from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Attention, Input


def create_attention_layer(input_layer):
    attention = Attention(use_scale=True)([input_layer, input_layer])
    return attention


def create_lstm_model(input_shape, units=50):
    """
    Creates an LSTM model with an attention layer.

    Parameters:
    - input_shape (tuple): The shape of the input data.
    - units (int): The number of units in the LSTM layers.

    Returns:
    - A compiled Keras model.
    """
    input_layer = Input(shape=input_shape)
    lstm = LSTM(units=units, return_sequences=True)(input_layer)
    attention = create_attention_layer(lstm)
    lstm = LSTM(units=units, return_sequences=False)(attention)
    output_layer = Dense(units=1)(lstm)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def train_model(model, X_train, t_train, epochs=25, batch_size=32):
    """
    Trains the LSTM model.

    Parameters:
    - model: The LSTM model to train.
    - x_train: Features from the training data.
    - y_train: Target variable from the training data.
    - epochs (int): Number of epochs to train the model.
    - batch_size (int): Batch size for the training.

    Returns:
    - The history object containing training loss values and metrics values.
    """
    history = model.fit(X_train, t_train, epochs=epochs, batch_size=batch_size)
    return history


def make_prediction(model, X_test):
    """
    Uses the LSTM model to make predictions based on the input data.
    Postconditions: the output is normalized and must be inverse transformed for actual predictions.

    Parameters:
    - data: The dataset with the normalized Close column values.
    - model: The trained LSTM model.
    - x_test: The input data for making predictions. It should have the same shape as x_train.

    Returns:
    - predictions:
    """
    predictions = model.predict(X_test)
    return predictions

