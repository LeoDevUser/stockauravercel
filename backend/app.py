from analysis import analyze_stock
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import trie
import json
import os

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:3000",
    # Vercel domains - update these after deployment
    "https://stock-aura-vercel.vercel.app",
    "https://stockaura.vercel.app",
]

# Also allow any *.vercel.app subdomain via regex
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/analyze')
async def analyze(
    ticker: str,
    period: str = '5y',
    window_days: int = 5,
    account_size: int = 10000,
    risk_per_trade: float = 0.02
):
    result = analyze_stock(
        ticker,
        period,
        window_days,
        account_size=account_size,
        risk_per_trade=risk_per_trade
    )
    if not result.get('title') and ticker.upper() in data_map:
        result['title'] = data_map[ticker.upper()]['title']
    return result


# Build trie for searching tickers
with open('tickers.json', 'r') as fp:
    ticker_data = json.load(fp)
t = []
data_map = {}
for key, value in ticker_data.items():
    ticker = value['ticker']
    t.append(ticker)
    data_map[ticker] = value

tickers = trie.Trie(t, data_map)


@app.get('/search')
async def search(q: str, limit: int = 10):
    results = tickers.complete(q.upper())
    if len(results) >= limit:
        return results[:limit]
    return results


@app.get('/top')
async def top():
    with open('top_stocks.json', 'r') as fp:
        data = json.load(fp)
    return data


@app.get("/")
def root():
    return {"message": "StockAura API is running"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
