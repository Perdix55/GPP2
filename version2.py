import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Function to fetch gas prices from the website
def fetch_gas_prices(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    data = []
    table = soup.find('table')  
if table is None:  
    print("Table not found in the HTML.")  
    return pd.DataFrame()  


    for row in table.find_all('tr')[1:]:  # Skip header
        cols = row.find_all('td')
        try:
            date = cols[0].text.strip()
            price = float(cols[1].text.strip().replace('$', ''))
            data.append({'date': date, 'price': price})
        except (IndexError, ValueError):
            continue  # Skip rows with missing or invalid data

    return pd.DataFrame(data)

# Function to plot historical trends
def plot_trends(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df['price'], marker='o', label='Gas Price')
    plt.title('Gas Price Trends')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid()
    plt.legend()
    st.pyplot(plt)  # Use Streamlit's pyplot for visualization

# Function to forecast future prices
def forecast_prices(df):
    df['date_numeric'] = (df['date'] - df['date'].min()).dt.days
    model = LinearRegression()
    X = df['date_numeric'].values.reshape(-1, 1)
    y = df['price'].values
    model.fit(X, y)

    future_dates = np.array([df['date_numeric'].max() + i for i in range(1, 8)]).reshape(-1, 1)
    predictions = model.predict(future_dates)
    return predictions

# Streamlit App
st.title("Gas Price Forecasting Application")
st.write("View historical gas prices and predictions for future price trends.")

# URL for gas price data
url = "https://ycharts.com/indicators/us_gas_price"

# Fetch and display data
gas_prices = fetch_gas_prices(url)

if gas_prices.empty:
    st.write("No data available. Please check the source or try again later.")
else:
    st.subheader("Historical Gas Prices")
    st.write(gas_prices.head())  # Display first few rows of data
    plot_trends(gas_prices)

    st.subheader("Predicted Gas Prices for the Next Week")
    predictions = forecast_prices(gas_prices)
    for i, pred in enumerate(predictions, start=1):
        st.write(f"Day {i}: ${pred:.2f}")
