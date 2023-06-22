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
# _______________________________________________________________________
## test set

#excel_file = 'Dataset_for_test.xlsx'
excel_file = 'https://github.com/amitfallach/Advanced-data-mining-in-Python---FinalProject/blob/main/Dataset_for_test.xlsx'
df = pd.read_excel(excel_file)
# _______________________________________________________________________
## train set

# excel_file = 'output_all_students_Train_v10.csv'
# df = pd.read_csv(excel_file)
# _______________________________________________________________________

data = prepare_data(df)

x = data.drop("price", axis=1)
y = data.price.astype(float)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# _______________________________________________________________________

null_col = X_train.columns[X_train.isnull().any()]
simp = SimpleImputer(strategy='most_frequent', add_indicator=False)
simp.fit(X_train[null_col])
X_train[null_col] = simp.transform(X_train[null_col])

# _______________________________________________________________________

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


filtered_variables = vif_data.loc[vif_data['VIF'] >= 5, 'Variable'].values
X_train = X_train.drop(filtered_variables, axis=1)

# _______________________________________________________________________

num_cols = [col for col in X_train.columns if X_train[col].dtypes!='O' and (X_train[col].dtypes!='object')]
cat_cols = [col for col in X_train.columns if (X_train[col].dtypes=='O' or (X_train[col].dtypes=='object'))]

# _______________________________________________________________________


numerical_pipeline = Pipeline([('numerical_imputation', SimpleImputer(strategy='most_frequent', add_indicator=False)),('scaling', StandardScaler())])

categorical_pipeline = Pipeline([('categorical_imputation', SimpleImputer(strategy='most_frequent', add_indicator=False, fill_value='missing')),('one_hot_encoding', OneHotEncoder(sparse=False, handle_unknown='ignore'))])

column_transformer = ColumnTransformer([('numerical_preprocessing', numerical_pipeline, num_cols),('categorical_preprocessing', categorical_pipeline, cat_cols)], remainder='drop')

elastic_net_pipeline = Pipeline([('preprocessing', column_transformer),('elastic_net', ElasticNet(alpha=0.8, l1_ratio=1, random_state=42))])

cross_val = KFold(n_splits=10)

scores = cross_val_score(elastic_net_pipeline, x, y, cv=cross_val , scoring='neg_mean_squared_error')

scores = cross_val_score(elastic_net_pipeline, x, y, cv=cross_val , scoring='neg_mean_squared_error')
print("Cross-validation scores:", scores)
print("Average MSE:", np.mean(-scores))
rmse_scores = np.sqrt(-scores)
print("Average RMSE:", np.mean(rmse_scores))

# _______________________________________________________________________
#joblib.dump(elastic_net_pipeline, 'trained_model.pkl')