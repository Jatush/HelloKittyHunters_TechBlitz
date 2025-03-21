import os
from dotenv import load_dotenv
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
class ConversationalResponse(BaseModel):
    """Classify the company into Tickets"""
    symbol: str = Field(description="Classify the company into Tickets")

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = "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"
user_input=input("Enter the Company:")
from langchain_core.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate([
    ("system", '''You are an assistant that returns the company ticket based on the company.
                    Apple Inc. (AAPL)
                    Microsoft Corporation (MSFT)
                    Amazon.com, Inc. (AMZN)
                    Alphabet Inc. (Class A) (GOOGL)
                    Meta Platforms, Inc. (META)
                    NVIDIA Corporation (NVDA)
                    Tesla, Inc. (TSLA)
                    Berkshire Hathaway Inc. (Class B) (BRK.B)
                    JPMorgan Chase & Co. (JPM)
                    Johnson & Johnson (JNJ)
     
    '''),
    ("user", "Provide ticket for company {company}")
])
llm_groq=ChatGroq(model='llama-3.3-70b-versatile')
structured_llm = llm_groq.with_structured_output(ConversationalResponse)
chain = prompt_template | structured_llm
response=chain.invoke({"company":user_input})
symbol=response.symbol
print(symbol)
# structured_llm.invoke("")
# Initialize the LLM
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.9
)

# Pydantic Models for Structured Outputs
class FinancialData(BaseModel):
    filename: str

class ForecastData(BaseModel):
    forecast: dict

class VisualizationOutput(BaseModel):
    message: str

# Tool: Fetch financial data and save to CSV
@tool
def fetch_and_save_financial_data(ticker_symbol: str) -> FinancialData:
    """
    Fetches historical financial data for the given ticker symbol and saves it to a CSV file.
    Returns:
        FinancialData: Object containing the filename where data is saved.
    """
    ticker = yf.Ticker(ticker_symbol)
    historical_data = ticker.history(period='1y')
    filename = f'{ticker_symbol}_Data.csv'
    historical_data.to_csv(filename)
    return FinancialData(filename=filename)

# Tool: Generate forecast from CSV data
@tool
def generate_forecast_from_csv(financial_data: FinancialData) -> ForecastData:
    """
    Generates a simple 12-month financial forecast from data stored in a CSV file.

    Args:
        financial_data (FinancialData): Object containing the filename with historical data.

    Returns:
        ForecastData: Forecasted data.
    """
    df = pd.read_csv(financial_data.filename, parse_dates=['Date'], index_col='Date')
    df['Monthly Return'] = df['Close'].pct_change().resample('M').mean()
    last_price = df['Close'].iloc[-1]
    forecasted_prices = [last_price * (1 + df['Monthly Return'].mean())**i for i in range(1, 13)]
    forecast_dates = pd.date_range(start=df.index[-1], periods=13, freq='M')[1:]
    forecast_df = pd.DataFrame({'Forecasted Price': forecasted_prices}, index=forecast_dates)
    return ForecastData(forecast=forecast_df.to_dict())

# Tool: Create visualizations from CSV data and forecast
@tool
def create_visualizations_from_csv(financial_data: FinancialData, forecast_data: ForecastData) -> VisualizationOutput:
    """
    Creates visualizations for historical and forecasted financial data.

    Args:
        financial_data (FinancialData): Object containing the filename with historical data.
        forecast_data (ForecastData): Forecasted data.

    Returns:
        VisualizationOutput: Confirmation message.
    """
    # historical_df = pd.read_csv(financial_data.filename, parse_dates=['Date'], index_col='Date')
    # forecast_df = pd.DataFrame.from_dict(forecast_data.forecast)
    # forecast_df.index = pd.to_datetime(forecast_df.index)

    # plt.figure(figsize=(12, 6))
    # plt.plot(historical_df.index, historical_df['Close'], label='Historical Closing Price')
    # plt.plot(forecast_df.index, forecast_df['Forecasted Price'], label='Forecasted Price', linestyle='--')
    # plt.title('Historical and Forecasted Stock Prices')
    # plt.xlabel('Date')
    # plt.ylabel('Price (USD)')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('financial_forecast.png')
    # plt.close()
    return VisualizationOutput(message="Visualizations created and saved as financial_forecast.png")

# Define Agents
data_retrieval_agent = Agent(
    role="Data Retrieval Specialist",
    goal="Fetch financial data for analysis.",
    backstory="An experienced data engineer specializing in retrieving accurate and up-to-date financial market data.",
    tools=[fetch_and_save_financial_data],
    llm=llm
)

financial_forecasting_agent = Agent(
    role="Financial Analyst",
    goal="Generate 12-month financial forecasts.",
    backstory="A skilled financial analyst with a knack for predicting market trends using historical data.",
    tools=[generate_forecast_from_csv],
    llm=llm
)

visualization_agent = Agent(
    role="Visualization Expert",
    goal="Create visual representations of financial data.",
    backstory="A creative data scientist with expertise in crafting clear and insightful financial visualizations.",
    tools=[create_visualizations_from_csv],
    llm=llm
)

# Define Tasks
data_retrieval_task = Task(
    description="Retrieve and save financial data for ticker symbol.",
    agent=data_retrieval_agent,
    inputs={'ticker_symbol': symbol},
    expected_output="FinancialData object containing the filename with historical market data.",
    output_pydantic=FinancialData
)


crew = Crew(
    agents=[
        data_retrieval_agent,
    ],
    tasks=[
        data_retrieval_task,
    ],
    process=Process.sequential,  # Execute tasks in order
    verbose=True
)
def run():
    crew.kickoff()

run()

import matplotlib.pyplot as plt
from io import StringIO
df = pd.read_csv(f'{symbol}_Data.csv', parse_dates=["Date"])

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
plt.savefig(f'{symbol}_closing_price_moving_averages.png')
plt.close()

# Plot Daily Returns
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Daily_Return'] * 100, color='purple')
plt.title('Daily Return (%) Over Time')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{symbol}_daily_returns.png')
plt.close()

# Store insights as string for AI inputs
ai_inputs = f"""
Key Insights for Company {user_input}:
- Price Change Over Period: {key_insights['Price Change (%)']}%
- Volatility: {key_insights['Volatility (%)']}%
- Average Volume: {key_insights['Average Volume']}
- Highest Close: {key_insights['Highest Closing Price']}
- Lowest Close: {key_insights['Lowest Closing Price']}

Assess Risk based on volatility and reccomandation whether Bullish trend: Consider buying or holding the stock.,Bearish trend: Consider risk mitigation or short-term exit."Neutral trend: Hold and monitor market conditions.Neutral trend: Hold and monitor market conditions.
"""
prompt_template_2 = ChatPromptTemplate([
    ("system", ai_inputs),
    ("user", "Provide forecasting for the company: {company}")
])

chain_2=prompt_template_2 | llm_groq
response=chain_2.invoke({'company':user_input})
print(response.content)