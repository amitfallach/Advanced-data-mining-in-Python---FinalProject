import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from madlan_data_prep import prepare_data
import joblib
from sklearn.compose import make_column_transformer
# _______________________________________________________________________
## test set

#excel_file = 'Dataset_for_test.xlsx'
#excel_file = 'https://github.com/amitfallach/Advanced-data-mining-in-Python---FinalProject/blob/main/Dataset_for_test.xlsx'
#df = pd.read_excel(excel_file)
# _______________________________________________________________________
## train set
#excel_file = 'https://github.com/amitfallach/Advanced-data-mining-in-Python---FinalProject/blob/main/output_all_students_Train_v10.csv'
excel_file = 'output_all_students_Train_v10.csv'
df = pd.read_csv(excel_file)
# _______________________________________________________________________

data = prepare_data(df)

x = data.drop("price", axis=1)
y = data.price.astype(float)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# _______________________________________________________________________

#We performed a test using VIF in order to get an indication of 
#the relevance of the columns. We performed the test in advance and put
# it in the current file because it is not possible to do the test without first splitting it into a training set and a test set.
# We left the code in the current file to illustrate the correct way of using the test (after splitting the sets).

null_col = X_train.columns[X_train.isnull().any()]
simp = SimpleImputer(strategy='most_frequent', add_indicator=False)
simp.fit(X_train[null_col])
X_train[null_col] = simp.transform(X_train[null_col])

numerical_features = ['Area',
 'number_in_street',
 'num_of_images',
 'hasElevator ',
 'hasParking ',
 'hasBars ',
 'hasStorage ',
 'hasAirCondition ',
 'hasBalcony ',
 'hasMamad ',
 'handicapFriendly ',
 'publishedDays ',
 'floor',
 'total_floors',
 'room_number']

# Calculate VIF for each numeric column
vif_data = pd.DataFrame()
vif_data["Variable"] = numerical_features
vif_data["VIF"] = [variance_inflation_factor(X_train[numerical_features].values.astype(np.float64), i)
                   for i in range(len(numerical_features))]

# show relevant numeric columns
filtered_variables = vif_data.loc[vif_data['VIF'] >= 5, 'Variable'].values
# _______________________________________________________________________

#From the current part of the code, we saved only the relevant columns according to VIF,
# we saved them only now (instead of in the prepare_data file) because we wanted to
# illustrate the use of VIF on all the numerical columns (according to the lecturers,
# the use of VIF is correct from the stage of splitting into a training set and a test set).

col_to_model  = ['Area', 'hasElevator ','hasParking ', 'hasBars ', 'hasStorage ',
                    'hasBalcony ', 'hasMamad ', 'handicapFriendly ', 'floor',
                    'City', 'type',  'city_area', 'condition ']

X_train = X_train[col_to_model]
X_test = X_test[col_to_model]
# _______________________________________________________________________

num_cols = [col for col in X_train.columns if X_train[col].dtypes!='O' and (X_train[col].dtypes!='object')]
cat_cols = [col for col in X_train.columns if (X_train[col].dtypes=='O' or (X_train[col].dtypes=='object'))]

# _______________________________________________________________________


numerical_pipeline = Pipeline([('numerical_imputation', SimpleImputer(strategy='most_frequent', add_indicator=False)),('scaling', StandardScaler())])

#categorical_pipeline = Pipeline([('categorical_imputation', SimpleImputer(strategy='most_frequent', add_indicator=False, fill_value='missing')),('one_hot_encoding', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
categorical_pipeline = Pipeline([
    ('categorical_imputation', SimpleImputer(strategy='most_frequent', add_indicator=False, fill_value='missing')),
    ('one_hot_encoding', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer without using ColumnTransformer
preprocessing_pipeline = make_column_transformer(
    (numerical_pipeline, num_cols),
    (categorical_pipeline, cat_cols),
    remainder='drop')


elastic_net_pipeline = Pipeline([('preprocessing', preprocessing_pipeline),('elastic_net', ElasticNet(alpha=0.8, l1_ratio=1, random_state=42))])

cross_val = KFold(n_splits=10)

scores = cross_val_score(elastic_net_pipeline, x, y, cv=cross_val , scoring='neg_mean_squared_error')

scores = cross_val_score(elastic_net_pipeline, x, y, cv=cross_val , scoring='neg_mean_squared_error')
print("Cross-validation scores:", scores)
print("Average MSE:", np.mean(-scores))
rmse_scores = np.sqrt(-scores)
print("Average RMSE:", np.mean(rmse_scores))

elastic_net_pipeline.fit(X_train, y_train)

# Predict on the test data
y_pred = elastic_net_pipeline.predict(X_test)
print(y_pred)
# _______________________________________________________________________
joblib.dump(elastic_net_pipeline, 'trained_model.pkl')