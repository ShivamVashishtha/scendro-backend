# app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import yfinance as yf
import math
from datetime import datetime
import pytz
import os
import time
import re
from openai import OpenAI
from dotenv import load_dotenv

# --- Load environment ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY", "7GBK6ASVSW2UEON4")
GNEWS_KEY = os.getenv("GNEWS_API_KEY", "c3908fd05a6295e9175cbb3b4cdf48b8")

app = FastAPI()

# ✅ Allow frontend (React) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class RecommendationRequest(BaseModel):
    capital: float = Field(..., gt=0)
    duration_days: int = Field(..., gt=0)
    growth_target: float = Field(..., gt=0, lt=1000)
    risk_tolerance: float = Field(..., ge=0.0, le=1.0)
    tickers: List[str]

class ChatRequest(BaseModel):
    question: str

# --- Helper functions ---
sector_weights = {"Technology": 1.1, "Healthcare": 1.05, "Energy": 1.0, "Financial Services": 0.95, "Consumer Cyclical": 0.9}

def safe_float(val, default):
    try:
        return float(val) if val not in [None, 'None', ''] else default
    except:
        return default

def fetch_stock_data(ticker):
    ticker_obj = yf.Ticker(ticker)
    history = ticker_obj.history(period="7d", interval="1h", prepost=True)
    if history.empty:
        raise ValueError(f"No data found for {ticker}")
    return history["Close"]

def fetch_fundamentals(ticker):
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_KEY}"
    data = requests.get(url).json()
    pe = safe_float(data.get("PERatio"), 15.0)
    rev_growth = safe_float(data.get("QuarterlyRevenueGrowthYOY"), 0.1)
    sector = data.get("Sector", "Technology") or "Technology"
    return pe, rev_growth, sector

def fetch_news(ticker):
    url = f"https://gnews.io/api/v4/search?q={ticker}&token={GNEWS_KEY}&lang=en&max=3"
    try:
        data = requests.get(url).json()
        return [article['title'] for article in data.get("articles", [])]
    except:
        return []

def create_features(ticker):
    prices = fetch_stock_data(ticker)
    pe, rev_growth, sector = fetch_fundamentals(ticker)
    returns = prices.pct_change().dropna()
    volatility = returns.std()
    avg_return = returns.mean()
    rsi = (returns[returns > 0].mean() / abs(returns[returns < 0].mean())) * 100 if returns[returns < 0].mean() != 0 else 50
    ema_ratio = prices.ewm(span=20).mean().iloc[-1] / prices.iloc[-1] if prices.iloc[-1] != 0 else 1
    drawdown = ((prices.cummax() - prices) / prices.cummax()).max()
    sharpe_like = avg_return / volatility if volatility != 0 else 0
    sector_weight = sector_weights.get(sector, 1.0)
    return [rsi, volatility, avg_return, ema_ratio, drawdown, pe, rev_growth, sector_weight, sharpe_like], {
        "rsi": rsi, "volatility": volatility, "avg_return": avg_return, "pe": pe, "growth": rev_growth, "sector": sector,
        "latest_price": prices.iloc[-1] if not prices.empty else None
    }

# --- AI model training on startup ---
model = None
scaler = None

@app.on_event("startup")
def train_model():
    global model, scaler
    tickers = ["AAPL", "TSLA", "GOOG", "MSFT", "NVDA", "AMZN", "META", "INTC"]
    X = [create_features(t)[0] for t in tickers]
    y = [12, 14, 10, 9, 18, 13, 11, 7]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6)
    model.fit(X_scaled, y)

# --- API routes for Paper Trading and Options Trading ---
@app.get("/api/options/dates")
def get_option_dates(symbol: str):
    ticker = yf.Ticker(symbol)
    try:
        dates = ticker.options
        return {"expirations": dates}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/options/chain")
def get_options_chain(symbol: str, expiry: str):
    try:
        ticker = yf.Ticker(symbol)
        chain = ticker.option_chain(expiry)

        def sanitize(option):
            return {
                key: (None if isinstance(value, float) and math.isnan(value) else value)
                for key, value in option.items()
            }

        def format_option(opt):
            return {
                "contractSymbol": opt.contractSymbol,
                "strike": opt.strike,
                "lastPrice": opt.lastPrice,
                "bid": opt.bid,
                "ask": opt.ask,
                "volume": opt.volume,
                "openInterest": opt.openInterest,
                "impliedVolatility": opt.impliedVolatility,
            }

        calls = [sanitize(format_option(row)) for _, row in chain.calls.iterrows()]
        puts = [sanitize(format_option(row)) for _, row in chain.puts.iterrows()]

        return {"calls": calls, "puts": puts}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/price")
def get_price(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="1d", interval="1m", prepost=True)
        if history.empty:
            raise ValueError("No intraday data available")
        latest_price = history["Close"].iloc[-1]
        return {"price": round(latest_price, 2)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/market-status")
def market_status():
    try:
        eastern = pytz.timezone("US/Eastern")
        now = datetime.now(eastern)
        is_open = (
            now.weekday() < 5 and
            (now.hour > 9 or (now.hour == 9 and now.minute >= 30)) and
            (now.hour < 16)
        )
        return {"isMarketOpen": is_open}
    except Exception as e:
        return {"error": str(e)}

# --- AI Advisor API routes ---
@app.post("/recommend")
def recommend_portfolio(request: RecommendationRequest):
    results = []
    diagnostics = []
    for ticker in request.tickers:
        time.sleep(0.2)
        features, meta = create_features(ticker)
        X_scaled = scaler.transform([features])
        predicted = float(model.predict(X_scaled)[0])
        volatility = meta['volatility']
        confidence = max(1 - volatility, 0.01)
        score = max((predicted * confidence + meta['avg_return'] * 100 - request.growth_target), 0) / 100
        invest = max(request.capital * score * request.risk_tolerance, 0)

        if invest >= 0:
            results.append({
                "ticker": ticker,
                "predicted_growth": round(predicted, 2),
                "score": invest,
                "justification": f"{ticker} offers {predicted:.2f}% forecasted growth with healthy metrics: PE={meta['pe']}, Sector={meta['sector']}, Volatility={volatility:.3f}"
            })
        diagnostics.append(f"{ticker}: model={predicted:.2f}, PE={meta['pe']}, Growth={meta['growth']:.2f}, Sector={meta['sector']}")

    total_score = sum(r['score'] for r in results)
    if total_score > 0:
        for r in results:
            weight = r['score'] / total_score
            r['suggested_investment'] = round(request.capital * weight, 2)
            r['portfolio_pct'] = round(weight * 100, 2)
            del r['score']
    else:
        return {"recommendations": [], "message": "No strong picks.", "diagnostics": diagnostics}

    return {"recommendations": results, "progress": 100, "diagnostics": diagnostics}

@app.post("/chat")
def ai_chat(request: ChatRequest):
    try:
        # Find ticker from question
        match = re.search(r'\b[A-Z]{2,5}\b', request.question.upper())
        ticker = match.group(0) if match else "TSLA"

        # Fetch history (7d, 1h interval)
        ticker_obj = yf.Ticker(ticker)
        history = ticker_obj.history(period="7d", interval="1h", prepost=True)
        if history.empty or history["Close"].isnull().all():
            history = ticker_obj.history(period="1d", interval="1m", prepost=True)

        if history.empty:
            raise ValueError(f"No data found for {ticker}")

        # Latest price
        latest_price = history["Close"].iloc[-1]
        latest_timestamp = history.index[-1].strftime("%Y-%m-%d %H:%M %p EST")

        # Real RSI calculation
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]

        real_rsi = calculate_rsi(history["Close"])

        # Volatility
        returns = history["Close"].pct_change().dropna()
        volatility = returns.std()

        # Fundamentals
        pe, rev_growth, sector = fetch_fundamentals(ticker)
        headlines = fetch_news(ticker)

        # Compose indicators
        indicators = (
            f"## {ticker} Snapshot\n\n"
            f"- Price: ${latest_price:.2f} (as of {latest_timestamp})\n"
            f"- RSI: {real_rsi:.2f}\n"
            f"- Volatility: {volatility:.4f}\n"
            f"- Earnings Growth: {rev_growth:.2f}\n"
            f"- PE Ratio: {pe:.2f}\n"
            f"- Sector: {sector}\n"
            f"- News Headlines:\n" + "\n".join([f"  - {h}" for h in headlines])
        )

        # Chat Prompt
        messages = [
            {"role": "system", "content": (
                "You are InciteAI, a sophisticated stock strategist. You analyze any trading-related question. "
                "Respond in markdown format with these sections:\n\n"
                "## Market Outlook\n"
                "## Indicator Breakdown\n"
                "## Sentiment Summary\n"
                "## Recommended Action: *Buy*/*Sell*/*Hold*/*Wait* (only if applicable)\n"
                "## Confidence Score: (0–100%) (only if applicable)"
            )},
            {"role": "user", "content": f"{request.question}\n\n{indicators}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=800
        )

        content = response.choices[0].message.content.strip()

        # Try parsing Recommended Action and Confidence Score
        action_match = re.search(r"\*\*Recommended Action\*\*:\s?\*?([A-Za-z]+)\*?", content)
        confidence_match = re.search(r"\*\*Confidence Score\*\*:\s?(\d+)%?", content)

        action = action_match.group(1).capitalize() if action_match else "Unknown"
        confidence = confidence_match.group(1) if confidence_match else "?"

        return {
            "reasoning": content,
            "recommended_action": action,
            "confidence_score": confidence
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

