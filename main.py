import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import HashingEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Amazon.csv")
# print(data)
# print(data.info())
# print(data.describe())

data["Quantity_cat"]=pd.cut(data["Quantity"],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2 ,random_state=42)
for train_set , test_set in split.split(data,data["Quantity_cat"]):
    strat_train_set = data.loc[train_set].drop("Quantity_cat",axis=1)
    strat_test_set = data.loc[test_set].drop("Quantity_cat",axis=1)

# print(strat_train_set)

data = strat_train_set.copy()

data_features = data.drop("TotalAmount",axis=1)
data_labels = data["TotalAmount"].copy()

num_attbs =data_features.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_attbs =data_features.select_dtypes(include=["object"]).columns.tolist()

num_pipeline = Pipeline([
    ("scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("encoder",HashingEncoder(n_components=64)),
])

full_pipeline = ColumnTransformer([
    ("num",num_pipeline, num_attbs),
    ("cat",cat_pipeline, cat_attbs),
])

data_prepared =full_pipeline.fit_transform(data_features)
print(data_prepared.shape)

# model = RandomForestRegressor()
# model.fit(data_prepared,data_labels)
# model.predict(data_prepared)
# model_rmses =-cross_val_score(model , data_prepared , data_labels ,scoring="neg_root_mean_squared_error",cv=10)
# print(pd.Series(model_rmses).describe())

model_tree = DecisionTreeRegressor()
model_tree.fit(data_prepared,data_labels)
model_tree_rmses = -cross_val_score(model_tree , data_prepared , data_labels , scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(model_tree_rmses).describe())

model_liner = LinearRegression()
model_liner.fit(data_prepared,data_labels)
model_liner_rmses =-cross_val_score(model_tree , data_prepared , data_labels , scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(model_liner_rmses).describe())