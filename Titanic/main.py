import pandas as pd
import my_functions as mf
from sklearn.preprocessing import Imputer

# Load data
train_data = pd.read_csv('../data_titanic/train.csv')
test_data = pd.read_csv('../data_titanic/test.csv')
data = pd.concat([train_data, test_data])

# First checks on the data
'''
train_data.head()
train_data.info()
train_data.describe()
'''

cat_params = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
mf.df_value_counts(data, cat_params)
# train_data.hist(bins=50, figsize=(20, 15))

# ###### Correlations ###### #
# corr between Survived and Fare, anti-corr Survived and Pclass
corr_matrix = train_data.corr()


# ###### Data Cleaning ###### #
imputer = Imputer(strategy='median')
data_num = data.drop(cat_params, axis=1)
imputer.fit(data_num)
imputer.statistics_
data_num.columns

data_tr = pd.DataFrame(imputer.transform(data_num), columns)
