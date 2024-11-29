import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load the datasets
item_categories = pd.read_csv(r'Kaggle DataSet/item_categories.csv')
items = pd.read_csv(r'Kaggle DataSet/items.csv')
sales_train = pd.read_csv(r'Kaggle DataSet/sales_train.csv')
shops = pd.read_csv(r'Kaggle DataSet/shops.csv')
test = pd.read_csv(r'Kaggle DataSet/test.csv')

# Preprocessing
# Handle missing values (if any)
sales_train = sales_train.fillna(0)
test = test.fillna(0)

# Convert dates to datetime format
sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')

# Extract year, month, and day from the date
sales_train['year'] = sales_train['date'].dt.year
sales_train['month'] = sales_train['date'].dt.month
sales_train['day'] = sales_train['date'].dt.day

# Aggregating sales data by shop_id, item_id, and month
monthly_sales = sales_train.groupby(['shop_id', 'item_id', 'year', 'month'])['item_cnt_day'].sum().reset_index()

# Merge item and shop info to the aggregated sales data
monthly_sales = pd.merge(monthly_sales, items[['item_id', 'item_category_id']], on='item_id', how='left')
monthly_sales = pd.merge(monthly_sales, shops[['shop_id', 'shop_name']], on='shop_id', how='left')

# Prepare features (X) and target (y)
X = monthly_sales[['shop_id', 'item_id', 'item_category_id', 'year', 'month']]
y = monthly_sales['item_cnt_day']

# Encoding categorical features with LabelEncoder
label_encoder = LabelEncoder()
X_encoded = X.copy()

# Apply LabelEncoder on each categorical column to avoid SettingWithCopyWarning
X_encoded['shop_id'] = label_encoder.fit_transform(X_encoded['shop_id'])
X_encoded['item_id'] = label_encoder.fit_transform(X_encoded['item_id'])
X_encoded['item_category_id'] = label_encoder.fit_transform(X_encoded['item_category_id'])


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# Train an XGBoost regressor
model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                         max_depth=5, alpha=10, n_estimators=100)

model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plotting the actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Plotting the distribution of actual sales values
plt.figure(figsize=(10, 6))
plt.hist(y_test, bins=30, alpha=0.7, color='blue', label='Actual Sales')
plt.hist(y_pred, bins=30, alpha=0.7, color='red', label='Predicted Sales')
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.title("Distribution of Actual vs Predicted Sales")
plt.legend()
plt.show()

# Feature importance from XGBoost
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='weight', max_num_features=10, height=0.5)
plt.title("Top 10 Feature Importance")
plt.show()





# Plotting the sales trends over time (by month)
monthly_sales_trends = sales_train.groupby(['year', 'month'])['item_cnt_day'].sum().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(monthly_sales_trends['month'].astype(str) + '-' + monthly_sales_trends['year'].astype(str),
         monthly_sales_trends['item_cnt_day'], marker='o', color='g')
plt.xticks(rotation=90)
plt.xlabel('Month-Year')
plt.ylabel('Total Sales')
plt.title('Sales Trends Over Time')
plt.tight_layout()
plt.show()

# Plotting sales by shop_id to see sales performance per shop
shop_sales = monthly_sales.groupby('shop_id')['item_cnt_day'].sum().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(shop_sales['shop_id'], shop_sales['item_cnt_day'], color='orange')
plt.xlabel('Shop ID')
plt.ylabel('Total Sales')
plt.title('Total Sales by Shop')
plt.show()

print("bitti")

