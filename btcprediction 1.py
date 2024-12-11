import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
# โหลดข้อมูล
data = pd.read_csv('bitcoin_dialy.csv')
print(data.head())

# ตรวจสอบmissing data
missing_data = data.isnull().sum()

# แปลงคอลัมน์ 'Start' และ 'End' ให้เป็น datetime
data['Start'] = pd.to_datetime(data['Start'])
data['End'] = pd.to_datetime(data['End'])

print("Missing data:\n", missing_data)
print("\nData types after conversion:\n", data.dtypes)

from sklearn.model_selection import train_test_split
# ตัวแปรอิสระ (features)
X = data[['Open', 'High', 'Low', 'Volume', 'Market Cap']]
# ตัวแปรตาม (target)
y = data['Close']
# แบ่งข้อมูลtrain/tast 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ขนาดของชุดข้อมูลที่แบ่งได้
print("Training set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# สร้างและ train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict test data 
y_pred_linear = linear_model.predict(X_test)

# ประเมินLinear Regression model
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# สร้างและ train Random Forest Regression model
random_forest_model = RandomForestRegressor(random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)

# ประเมิน Random Forest Regression model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# # สร้างและ train Gradient Boosting Regression model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# ประเมิน
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

model_evaluation_results = {
    "Linear Regression": {"MSE": mse_linear, "R2": r2_linear},
    "Random Forest Regression": {"MSE": mse_rf, "R2": r2_rf},
    "Gradient Boosting Regression": {"MSE": mse_gb, "R2": r2_gb}
}
model_evaluation_results

print("Linear Regression - MSE:", mse_linear, "\nR2 Score:", r2_linear)
print("Random Forest Regression - MSE:", mse_rf, "\nR2 Score:", r2_rf)
print("Gradient Boosting Regression - MSE:", mse_gb, "\nR2 Score:", r2_gb)


data['Start'] = pd.to_datetime(data['Start'])
data.set_index('Start', inplace=True)

data['Time_Index'] = (data.index - data.index.min()) / pd.Timedelta(days=1)
X = data[['Time_Index']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print('\n\n',y_pred_rf)


plt.figure(figsize=(10, 5))
plt.plot(data.index, y, label='Actual Price', color='lime')
plt.scatter(X_test.index, y_pred_rf, color='deeppink', label='Predicted Price', s=10)
plt.title('Comparison of True Prices and Predicted Prices by Dates (Random Forest)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

start_date = '2018-01-01'
end_date = '2020-12-31'

data_filtered = data.loc[start_date:end_date]
X_filtered = X.loc[start_date:end_date]
y_filtered = y.loc[start_date:end_date]

y_pred_filtered = rf_model.predict(X_filtered)
print('\n\n',y_pred_filtered)

plt.figure(figsize=(10, 5))
plt.plot(data_filtered.index, y_filtered, label='Actual Price', color='lime')
plt.scatter(data_filtered.index, y_pred_filtered, color='deeppink', label='Predicted Price', s=10)
plt.title('Comparison of True Prices and Predicted Prices (2018-2020)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
