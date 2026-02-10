# StockAura

A statistical stock analysis platform that uses time-series metrics to evaluate trading opportunities based on market regimes, momentum patterns, and predictability scores.

## Features

- **Statistical Analysis**: Hurst exponent, ADF test, Ljung-Box test, momentum correlation
- **Market Regime Detection**: Trending vs mean-reverting behavior classification
- **Trading Signals**: Buy/Short/Wait recommendations based on statistical validation
- **Position Sizing**: Risk-based position calculator with stop-loss levels
- **Liquidity Analysis**: Volume-based feasibility checks and slippage estimation
- **Top 50 Screener**: Pre-ranked trading opportunities across major tickers
- **Interactive Charts**: Candlestick visualization with adjustable parameters

## Tech Stack

**Frontend**: React + TypeScript + Vite  
**Backend**: FastAPI  
**Charts**: Lightweight Charts  
**Styling**: CSS

## Quick Start

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py  # or uvicorn main:app
```

## Project Structure

```
frontend/
├── src/
│   ├── components/      # Reusable UI components
│   │   ├── Chart.tsx           # Candlestick chart
│   │   ├── SearchBar.tsx       # Ticker search with autocomplete
│   │   ├── TradingVerdict.tsx  # Signal analysis display
│   │   └── Tooltip.tsx         # Info tooltips
│   ├── pages/
│   │   ├── LandingPage.tsx     # Home/search page
│   │   ├── ResultsPage.tsx     # Individual stock analysis
│   │   └── Top.tsx             # Top 50 opportunities
│   └── styles/          # Component-specific CSS
backend/
├── api/                 # Analysis endpoints
└── data/                # Market data & calculations
```

## Key Metrics Explained

| Metric | What It Means |
|--------|---------------|
| **Hurst Exponent** | <0.45 = mean-reverting, >0.55 = trending, ~0.5 = random walk |
| **Predictability Score** | 0-4 scale based on passing statistical tests (ADF, Ljung-Box, etc.) |
| **Regime Stability** | Out-of-sample validation — pattern consistency over time |
| **Edge/Friction Ratio** | Expected return vs trading costs (need >3x to be tradeable) |

## API Endpoints

- `GET /api/search?q={ticker}` - Search for stocks
- `GET /api/analyze?ticker={ticker}&period=5y&window_days=5` - Full analysis
- `GET /api/top` - Top 50 trading opportunities (cached)

## Configuration

Users can adjust:
- **Account Size**: Total portfolio value
- **Risk Tolerance**: Max loss per trade (% of account)
- **Transaction Costs**: Commission percentage

## Disclaimer

**This is a research tool, not financial advice.** Historical patterns don't guarantee future results. Always paper trade first and never risk more than you can afford to lose.
