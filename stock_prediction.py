import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# A hybrid model of LSTM and Random Forest
class Hybrid_Model:

    def __init__(self, data, filtered_data):

        # Create the training data for the model (using 95% of the dataset)
        training_data_len = int(np.ceil( len(data) * 0.95 ))

        # Scale the training data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Number of days used for prediction
        prediction_days = 60

        # Create the training data for the model (using 95% of the dataset)
        training_data_len = int(np.ceil( len(data) * 0.95 ))
        train_data = scaled_data[0:int(training_data_len), :]

        # Create the testing data
        test_data = scaled_data[training_data_len - prediction_days:, :]

        x_train, y_train, x_test = [], [], []
        y_test = data[training_data_len:, :]

        for i in range(prediction_days, len(train_data)):
            x_train.append(train_data[i-prediction_days:i, 0])
            y_train.append(train_data[i, 0])

        for i in range(prediction_days, len(test_data)):
            x_test.append(test_data[i-prediction_days:i, 0])

        # Convert x_train, y_train and x_test to numpy arrays
        x_train, y_train, x_test = np.array(x_train), np.array(y_train), np.array(x_test)

        # Reshape the x_train and x_test
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

        # LSTM model building
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(32))
        model.add(Dense(1))

        # LSTM model complilation with Adam optimizer
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # LSTM model training
        model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=0)

        # LSTM model predictions
        predictions_lstm = model.predict(x_test)
        predictions_lstm = scaler.inverse_transform(predictions_lstm)

        # Build and train Random Forest model
        regressor = RandomForestRegressor(n_estimators=300, random_state=42)
        regressor.fit(x_train.reshape(-1, prediction_days), y_train)

        # Random Forest model predictions
        predictions_rf = regressor.predict(x_test.reshape(-1, prediction_days))
        predictions_rf = scaler.inverse_transform(predictions_rf.reshape(-1, 1))

        # Combine predictions of both models (averaging)
        predictions = 0.5 * predictions_lstm + 0.5 * predictions_rf

        # Show the actual and predicted adjusted closing price (Adj Close)
        valid = pd.DataFrame()
        valid['Actual Adj Close']=filtered_data[training_data_len:]
        valid['Predicted Adj Close'] = predictions
        print(valid)

        # Calculate the root mean squared error (RMSE)
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        print("Root-mean-square error: {}".format(rmse))


def main():

    # Create a new dataframe with the 'Adj Close' column
    filtered_data = pd.read_csv('GOOG.csv').filter(['Adj Close'])

    # Convert the dataframe to a numpy array
    np_data = filtered_data.values

    Hybrid_Model(np_data, filtered_data)

if __name__ == "__main__":
    main()
