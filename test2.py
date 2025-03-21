import os
from dotenv import load_dotenv
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from pydantic import BaseModel, Field
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = "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"

# Streamlit UI
st.title("AI-powered Stock Forecasting")
user_input = st.text_input("Enter the Company Name (e.g., Apple Inc.):")

if user_input:
    def run_all():
        class ConversationalResponse(BaseModel):
            """Classify the company into Tickets"""
            symbol: str = Field(description="Classify the company into Tickets")

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
                        Johnson & Johnson (JNJ)'''),
            ("user", "Provide ticket for company {company}")
        ])

        llm_groq = ChatGroq(model='llama-3.3-70b-versatile')
        structured_llm = llm_groq.with_structured_output(ConversationalResponse)
        chain = prompt_template | structured_llm
        response = chain.invoke({"company": user_input})
        symbol = response.symbol

        st.write(f"*Ticker Symbol Detected:* {symbol}")

        llm = LLM(model="groq/llama-3.3-70b-versatile", temperature=0.9)

        class FinancialData(BaseModel):
            filename: str

        @tool
        def fetch_and_save_financial_data(ticker_symbol: str) -> FinancialData:
            """
            Fetches historical financial data for the given ticker symbol and saves it to a CSV file.
            Returns:
                FinancialData: Object containing the filename where data is saved.
            """
            ticker = yf.Ticker(symbol)
            historical_data = ticker.history(period='1y')
            filename = f'{symbol}_Data.csv'
            historical_data.to_csv(filename)
            return FinancialData(filename=filename)

        data_retrieval_agent = Agent(
            role="Data Retrieval Specialist",
            goal="Fetch financial data for analysis.",
            backstory="An experienced data engineer specializing in retrieving accurate and up-to-date financial market data.",
            tools=[fetch_and_save_financial_data],
            llm=llm
        )

        data_retrieval_task = Task(
            description="Retrieve and save financial data for ticker symbol.",
            agent=data_retrieval_agent,
            inputs={'ticker_symbol': symbol},
            expected_output="FinancialData object containing the filename with historical market data.",
            output_pydantic=FinancialData
        )

        crew = Crew(
            agents=[data_retrieval_agent],
            tasks=[data_retrieval_task],
            process=Process.sequential,
            verbose=False
        )

        crew.kickoff()

        df = pd.read_csv(f'{symbol}_Data.csv', parse_dates=["Date"])
        df['Daily_Return'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        volatility = df['Daily_Return'].std()
        avg_volume = df['Volume'].mean()
        price_change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100

        key_insights = {
            "Start Date": str(df['Date'].iloc[0].date()),
            "End Date": str(df['Date'].iloc[-1].date()),
            "Price Change (%)": round(price_change_pct, 2),
            "Volatility (%)": round(volatility * 100, 2),
            "Average Volume": int(avg_volume),
            "Highest Closing Price": round(df['Close'].max(), 2),
            "Lowest Closing Price": round(df['Close'].min(), 2)
        }

        # Plot 1: Closing Price with Moving Averages
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
        closing_plot_path = f'{symbol}_closing_price_moving_averages.png'
        plt.savefig(closing_plot_path)
        plt.close()

        # Plot 2: Daily Returns
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Daily_Return'] * 100, color='purple')
        plt.title('Daily Return (%) Over Time')
        plt.xlabel('Date')
        plt.ylabel('Daily Return (%)')
        plt.grid(True)
        plt.tight_layout()
        daily_returns_path = f'{symbol}_daily_returns.png'
        plt.savefig(daily_returns_path)
        plt.close()

        ai_inputs = f"""
        Key Insights for Company {user_input}:
        - Price Change Over Period: {key_insights['Price Change (%)']}%
        - Volatility: {key_insights['Volatility (%)']}%
        - Average Volume: {key_insights['Average Volume']}
        - Highest Close: {key_insights['Highest Closing Price']}
        - Lowest Close: {key_insights['Lowest Closing Price']}

        Assess Risk based on volatility and recommendation whether Bullish trend: Consider buying or holding the stock., Bearish trend: Consider risk mitigation or short-term exit. Neutral trend: Hold and monitor market conditions.
        """

        prompt_template_2 = ChatPromptTemplate([
            ("system", ai_inputs),
            ("user", "Provide forecasting for the company: {company}")
        ])
        chain_2 = prompt_template_2 | llm_groq
        response_2 = chain_2.invoke({'company': user_input})

        st.subheader("AI Forecasting and Recommendation")
        st.write(response_2.content)

        st.subheader("Visual Analysis")
        st.image(closing_plot_path, caption="Stock Price with Moving Averages")
        st.image(daily_returns_path, caption="Daily Returns Over Time")

        st.subheader("Download Data")
        with open(f'{symbol}_Data.csv', 'rb') as file:
            st.download_button(
                label="Download CSV",
                data=file,
                file_name=f'{symbol}_Data.csv',
                mime='text/csv'
            )

    run_all()