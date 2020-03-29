import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# Root Mean Square Error Function
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# Load the Training Data
Data = pd.read_csv(r"C:\Users\Utente\PycharmProjects\Housing Prices Competition\Data\train.csv")
# Keep the Output data in a variable
output = Data["SalePrice"]
# Drop the output column
Data = Data.drop(columns=["SalePrice"])

# List of the columns with NaN
NaN_Columns = Data.columns[Data.isna().any()].tolist()

# Maintain only the not NaN columns
Data = Data.drop(columns=NaN_Columns)
# Keep the names of the NoNaN columns
noNaNcolumns = Data.columns

# Find the name of the columns with dtype == object
Object_columns = Data.columns[Data.dtypes == np.object].tolist()
# Eliminate the column with dtype == object
Data = Data.drop(columns=Object_columns)

# Number of input features
input_data = len(Data.columns)

# Create the ANN
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(input_data, )))
model.add(Dense(50, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='relu'))
# Compile the model
model.compile(optimizer="rmsprop", loss=root_mean_squared_error, metrics =["accuracy"])

# Transform the data into numpy array
y = output.to_numpy()
X = Data.to_numpy()

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Fit the model
model.fit(X_train, y_train,
          batch_size=146,
          epochs=550,
          validation_split=0.1)

# Scale the test data
X_test = scaler.transform(X_test)

# Predict the data
y_pred = model.predict(X_test)

# Print the result
print("Result:")
print(mean_squared_error(y_test, y_pred))


