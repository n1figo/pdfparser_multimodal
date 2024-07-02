import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load the data
june_df = pd.read_excel("MAU_forecast_data_corrected.xlsx", sheet_name='June 2024')
july_df = pd.read_excel("MAU_forecast_data_corrected.xlsx", sheet_name='July 2024')

# Ensure no NaN values in the data
june_df.fillna(0, inplace=True)
july_df.fillna(0, inplace=True)

# Add additional features
june_df['Date'] = pd.to_datetime(june_df['Date'])
june_df['Day of the Week'] = june_df['Date'].dt.dayofweek

# Compute "전일비" for "Logged-in Customers" and "Customers Visiting by ID"
june_df['전일비_Logged-in Customers'] = june_df['Logged-in Customers'].diff().fillna(0)
june_df['전일비_Customers Visiting by ID'] = june_df['Customers Visiting by ID'].diff().fillna(0)
june_df['전일비_MAU'] = june_df['MAU'].diff().fillna(0)

# Updated features to include "전일비" columns
features = ["Logged-in Customers", "전일비_Logged-in Customers", "Day of the Week", "Customers Visiting by ID", "전일비_Customers Visiting by ID"]
X_train = june_df[features]
y_train = june_df["MAU"]

# Define the models
linear_model = LinearRegression()
random_forest_model = RandomForestRegressor()
svr_model = SVR()

# Train the models using the June data
linear_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)
svr_model.fit(X_train, y_train)

# Streamlit app
st.title("Real-Time MAU Prediction Service")

# Input values
logged_in_mau = st.number_input("Logged-in MAU", min_value=0)
unique_visitor_id_count = st.number_input("Unique Visitor ID Count", min_value=0)

# Compute today's MAU and its change from previous day
today_mau = logged_in_mau + unique_visitor_id_count
prev_day_data = june_df.iloc[-1]
prev_day_mau = prev_day_data['MAU']
prev_day_logged_in = prev_day_data['Logged-in Customers']
prev_day_visitor_id = prev_day_data['Customers Visiting by ID']

# Compute changes
change_logged_in = logged_in_mau - prev_day_logged_in
change_visitor_id = unique_visitor_id_count - prev_day_visitor_id

# Create input data for prediction
today = datetime.now()
day_of_week = today.weekday()
input_data = pd.DataFrame({
    "Logged-in Customers": [logged_in_mau],
    "전일비_Logged-in Customers": [change_logged_in],
    "Day of the Week": [day_of_week],
    "Customers Visiting by ID": [unique_visitor_id_count],
    "전일비_Customers Visiting by ID": [change_visitor_id]
})

# Make predictions
linear_pred = linear_model.predict(input_data)[0]
random_forest_pred = random_forest_model.predict(input_data)[0]
svr_pred = svr_model.predict(input_data)[0]

# Display predictions
st.subheader("Predicted End of Day MAU")
st.write(f"Linear Regression: {linear_pred:.2f}")
st.write(f"Random Forest: {random_forest_pred:.2f}")
st.write(f"SVR: {svr_pred:.2f}")

# Visualization
st.subheader("Incremental Learning: Predictions for End of June MAU")
fig, ax = plt.subplots()
days = list(range(1, len(june_df)))
incremental_predictions = {
    "Linear Regression": [],
    "Random Forest": [],
    "SVR": []
}

for i in range(1, len(june_df)):
    X_train_incremental = june_df[features].iloc[:i]
    y_train_incremental = june_df["MAU"].iloc[:i]
    
    linear_model.fit(X_train_incremental, y_train_incremental)
    random_forest_model.fit(X_train_incremental, y_train_incremental)
    svr_model.fit(X_train_incremental, y_train_incremental)
    
    X_test_incremental = X_train.iloc[-1].values.reshape(1, -1)
    
    incremental_predictions["Linear Regression"].append(linear_model.predict(X_test_incremental)[0])
    incremental_predictions["Random Forest"].append(random_forest_model.predict(X_test_incremental)[0])
    incremental_predictions["SVR"].append(svr_model.predict(X_test_incremental)[0])

ax.plot(days, incremental_predictions["Linear Regression"], label='Linear Regression')
ax.plot(days, incremental_predictions["Random Forest"], label='Random Forest')
ax.plot(days, incremental_predictions["SVR"], label='SVR')
ax.axhline(y=june_df["MAU"].iloc[-1], color='r', linestyle='--', label='Actual MAU (End of June)')
ax.set_xlabel('Number of Days Used for Training')
ax.set_ylabel('Predicted End of June MAU')
ax.set_title('Incremental Learning: Predictions for End of June MAU')
ax.legend()
st.pyplot(fig)

# Alert if MAU is below target
target_mau = st.number_input("Target MAU", min_value=0)
if today_mau < target_mau:
    st.warning("MAU is below the target! Consider running events to boost engagement.")
