import pandas as pd
import numpy as np
import seaborn as sns
import os
import pylab as pl
import datetime
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")

data = pd.read_csv("data.csv")

pd.set_option('display.max_columns', None)

data.head()

#Checking the null Values

data.isnull().sum()

#To check Duplicates
data[data.duplicated()]



num_vars = data.select_dtypes(include=['int','float']).columns.tolist()

num_cols = len(num_vars)
num_rows = (num_cols + 2 ) // 3
fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
axs = axs.flatten()

for i, var in enumerate (num_vars):
    sns.boxplot(x=data[var],ax=axs[i])
    axs[i].set_title(var)

if num_cols < len(axs):
  for i in range(num_cols , len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()


def zohaib (data,age):
 Q1 = data[age].quantile(0.25)
 Q3 = data[age].quantile(0.75)
 IQR = Q3 - Q1
 data= data.loc[~((data[age] < (Q1 - 1.5 * IQR)) | (data[age] > (Q3 + 1.5 * IQR))),]
 return data

data.boxplot(column=["Potential_Savings_Miscellaneous"])

data = zohaib(data,"Potential_Savings_Miscellaneous")


from sklearn import preprocessing
for col in data.select_dtypes(include=['object']).columns:
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(data[col].unique())
    data[col] = label_encoder.transform(data[col])
    print(f'{col} : {data[col].unique()}')

data.info()




from autoviz.AutoViz_Class import AutoViz_Class 
AV = AutoViz_Class()
import matplotlib.pyplot as plt
%matplotlib INLINE
filename = 'data.csv'
sep =","
dft = AV.AutoViz(
    filename  
)










X = data.drop("Disposable_Income", axis = 1)

y = data["Disposable_Income"]

X.head()

y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.neighbors import KNeighborsRegressor
classifier = KNeighborsRegressor(n_neighbors=7, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn import metrics

from sklearn.metrics import mean_absolute_percentage_error

mean_absolute_percentage_error(y_test,y_pred)*100


from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import mean_absolute_percentage_error

mean_absolute_percentage_error(y_test,y_pred)*100

y_pred=lr.predict(X_test)
y_pred

lr.score(X_test,y_test)



import statsmodels.api as sm
LR=sm.add_constant(X)
model=sm.OLS(y,LR).fit()
print(model.summary())



from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(random_state=123)
dt.fit(X_train,y_train)
y_pred_dt = dt.predict(X_test)
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred_dt)*100



from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000,random_state=125)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred_rf)*100



from sklearn.ensemble import GradientBoostingRegressor
n_est=300
sgbr = GradientBoostingRegressor(n_estimators=1000, random_state=0)
sgbr.fit(X_train, y_train);
y_pred_sgbr = sgbr.predict(X_test)
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred_sgbr)*100
print("Training data accuracy:", sgbr.score(X_train, y_train))
print("Testing data accuracy", sgbr.score(X_test, y_test))



from sklearn.ensemble import AdaBoostRegressor
n_est=1000
adr = AdaBoostRegressor(n_estimators=n_est, random_state=0)
adr.fit(X_train, y_train);

print("Training data accuracy:", adr.score(X_train, y_train))
print("Testing data accuracy", adr.score(X_test, y_test))




from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Define a function to calculate MAPE with handling for zero values in y_test
def mean_absolute_percentage_error(y_true, y_pred):
    # Prevent division by zero by filtering out zero values in y_true
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

# Initialize models dictionary
models = {
    'LinearRegression': LinearRegression(),
    'XGBRegressor': XGBRegressor(random_state=42),
    'CatBoostRegressor': CatBoostRegressor(verbose=False, random_state=42),
    'LGBMRegressor': LGBMRegressor(random_state=42, verbose=-1),
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
    'AdaBoostRegressor': AdaBoostRegressor(random_state=42),
    'ExtraTreesRegressor': ExtraTreesRegressor(random_state=42),
    'BaggingRegressor': BaggingRegressor(random_state=42),
    'SVR': SVR()
}

# Iterate through models and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # Print out results including MAPE
    print(f"{name}: MAE={mae:.2f}, MSE={mse:.2f}, R2={r2:.2f}, MAPE={mape:.2f}%")
