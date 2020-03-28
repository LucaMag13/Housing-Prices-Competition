import pandas as pd

# Read the training Data
Data = pd.read_csv(r"C:\Users\Utente\PycharmProjects\Housing Prices Competition\Data\train.csv")

# Print the columns containing NaN
print(Data.columns[Data.isna().any()].tolist())

# Number of Columns with NaN values
print(len(Data.columns[Data.isna().any()].tolist()))

# Print the Types of the columns without Nan
print(Data[Data.columns[Data.notna().any()]].dtypes)
