import pandas as pd
import my_functions as mf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from CategoricalEncoder import CategoricalEncoder

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

%matplotlib inline

# Load data
train_data = pd.read_csv('../data_titanic/train.csv')
test_data = pd.read_csv('../data_titanic/test.csv')
data = pd.concat([train_data, test_data])

# First checks on the data
"""
train_data.head()
train_data.info()
train_data.describe()
"""
# Separate data and labels
X_train = train_data.drop('Survived', axis=1)
y_train = train_data.loc[:, 'Survived']

# Categorical Parameters
cat_params = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
cat_params_toUse = ['Sex', 'Embarked']
num_params = list(X_train.drop(cat_params, axis=1).columns)
mf.df_value_counts(data, cat_params)
X_train.hist(bins=50, figsize=(20, 15))

# ###### Correlations ###### #
# corr between Survived and Fare, anti-corr Survived and Pclass
corr_matrix = train_data.corr()


# ###### Data Cleaning ###### #
# -----------------
# Combine Attributes ?
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_params)),
    ('imputer', Imputer(strategy='median')),
    # ('std_scaler', StandardScaler())  # remove mean and normalise to std # ??
])
data_num = num_pipeline.fit_transform(train_data)
data_num_tr = pd.DataFrame(data_num, columns=num_params)

cat_params_toUse

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(['Sex'])),
    ('label_binarizer', CategoricalEncoder(encoding="onehot-dense"))
])

# Join the two pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])

data_prepared = full_pipeline.fit_transform(X_train)
test_data.drop('PassengerId', axis=1)
test_data_prepared = full_pipeline.fit_transform(test_data)
data_prepared.shape
test_data_prepared.shape
# ######### Training and Evaluating on the Training Set #########
# Stochastic Grasient Descent
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(data_prepared, y_train)
test_idx = 20
sgd_clf.predict([data_prepared[test_idx]])
y_train.iloc[test_idx]

cross_val_score(sgd_clf, data_prepared, y_train, cv=3, scoring='accuracy')

# Random Forest
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(data_prepared, y_train)
y_probas_forest = cross_val_predict(forest_clf, data_prepared, y_train, cv=3,
                                    method='predict_proba')

cross_val_score(forest_clf, data_prepared, y_train, cv=3, scoring='accuracy')
y_pred = pd.DataFrame(forest_clf.predict(
    test_data_prepared), columns=['Survived'])
id_pred = test_data['PassengerId']
result_df = pd.concat([id_pred, y_pred], axis=1)

# Save output to file
result_df.to_csv("results/result.csv", index=False, float_format='%.0f')
