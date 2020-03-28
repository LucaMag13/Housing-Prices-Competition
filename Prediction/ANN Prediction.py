import pandas as pd
import numpy as np

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

