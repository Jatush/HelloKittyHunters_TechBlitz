import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
df = pd.read_csv('AAPL_Data.csv', parse_dates=["Date"])

# Calculate Daily Returns
df['Daily_Return'] = df['Close'].pct_change()

# Calculate Moving Averages
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()

# Volatility - Standard deviation of daily returns
volatility = df['Daily_Return'].std()

# Average Volume
avg_volume = df['Volume'].mean()

# Price Trend - % Change from first to last close
price_change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100

# Key Insights Dictionary
key_insights = {
    "Start Date": str(df['Date'].iloc[0].date()),
    "End Date": str(df['Date'].iloc[-1].date()),
    "Price Change (%)": round(price_change_pct, 2),
    "Volatility (%)": round(volatility * 100, 2),
    "Average Volume": int(avg_volume),
    "Highest Closing Price": round(df['Close'].max(), 2),
    "Lowest Closing Price": round(df['Close'].min(), 2)
}

# Actionable Recommendations
if price_change_pct > 2:
    recommendation = "Bullish trend: Consider buying or holding the stock."
elif price_change_pct < -2:
    recommendation = "Bearish trend: Consider risk mitigation or short-term exit."
else:
    recommendation = "Neutral trend: Hold and monitor market conditions."

key_insights["Investment Recommendation"] = recommendation

# Risk Assessment
risk_assessment = "Moderate Risk" if volatility < 0.02 else "High Risk"
key_insights["Risk Assessment"] = risk_assessment

# Plot Closing Price and Moving Averages
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.plot(df['Date'], df['MA5'], label='5-Day MA', linestyle='--')
plt.plot(df['Date'], df['MA10'], label='10-Day MA', linestyle='--')
plt.title('Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('closing_price_moving_averages.png')
plt.close()

# Plot Daily Returns
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Daily_Return'] * 100, color='purple')
plt.title('Daily Return (%) Over Time')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig('daily_returns.png')
plt.close()

# Store insights as string for AI inputs
ai_inputs = f"""
Key Insights:
- Price Change Over Period: {key_insights['Price Change (%)']}%
- Volatility: {key_insights['Volatility (%)']}%
- Average Volume: {key_insights['Average Volume']}
- Highest Close: {key_insights['Highest Closing Price']}
- Lowest Close: {key_insights['Lowest Closing Price']}

Risk Assessment: {risk_assessment}

Investment Recommendation: {recommendation}

Inputs to AI:
- Data Points: Close prices, Daily returns, Moving averages
- Market Conditions: Moderate volatility detected
- Strategic Goal: Long-term growth
- Time Frame: Next 12 months
"""