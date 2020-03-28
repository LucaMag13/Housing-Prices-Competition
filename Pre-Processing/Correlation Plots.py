import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the training Data
Data = pd.read_csv(r"C:\Users\Utente\PycharmProjects\Housing Prices Competition\Data\train.csv")

# List of the columns with NaN
NaN_Columns = Data.columns[Data.isna().any()].tolist()

# Maintain only the not NaN columns
NoNan = Data.drop(columns=NaN_Columns)
# Keep the names of the NoNaN columns
noNaNcolumns = NoNan.columns

# Loop over the name of the columns
for i in noNaNcolumns:
    # If the data are strings print the swarmplot
    if NoNan[i].dtype == np.object:
        sns.swarmplot(x=NoNan[i], y=NoNan['SalePrice'])
    # If the data are Float or Int print a regplot
    else:
        sns.regplot(x=NoNan[i], y=NoNan['SalePrice'])

    plt.show()




