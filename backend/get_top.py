"""
StockAura — Top Market Cap Scanner

Analyzes the top stocks by market cap for trading opportunities.
Uses 30 Hurst shuffles (vs 50 for individual lookups) for faster batch processing.

Signal tiers:
  🟢 HIGH CONVICTION  — 3-5/5 predictability, all gates passed
  🟡 SPECULATIVE      — 2/5 predictability, stability + edge passed (half position)
  🔴 SHORT/WAIT       — Valid signal but bearish or waiting
  ⚪ DO NOT TRADE     — Failed validation
"""

import json
import re
import sys
from typing import Dict, List
import time
from datetime import datetime
import random

# Add backend to path
sys.path.insert(0, './backend')
from analysis import analyze_stock as run_analysis

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
TICKERS_FILE = "tickers.json"
TOP_N_STOCKS = 1200                 # Top 1200 by market cap (pre-filter)
REQUEST_DELAY = 1.0                 # Seconds between API requests
MAX_RETRIES = 3
RETRY_DELAY = 5
DEFAULT_ACCOUNT_SIZE = 10000
DEFAULT_RISK_PER_TRADE = 0.02
BATCH_SHUFFLES = 30                 # Fewer shuffles for speed (50 for individual lookups)

last_request_time = 0


def should_skip(ticker: str, title: str) -> bool:
    """
    Filter out tickers that shouldn't be in the stock screener:
    - ETFs, trusts, index funds
    - OTC foreign ordinaries (thin US volume, often duplicates of primary listings)
    """
    t = ticker.upper()
    title_up = title.upper()

    # Skip ETFs / trusts / index funds
    # Use word-boundary-aware matching to avoid false positives
    # (e.g. "NETFLIX" contains "ETF", "NORTHERN TRUST CORP" contains "TRUST")

    # These match as whole words only
    etf_word_patterns = [
        r'\bETF\b',           # "ETF" as standalone word (not inside NETFLIX)
        r'\bSPDR\b',
        r'\bISHARES\b',
        r'\bVANGUARD\b',
        r'\bPROSHARES\b',
        r'\bDIREXION\b',
        r'\bWISDOMTREE\b',
        r'\bGRAYSCALE\b',
    ]
    if any(re.search(pat, title_up) for pat in etf_word_patterns):
        return True

    # "TRUST" only if it looks like a fund/ETF trust, not a REIT or bank
    # Fund trusts: "XXX TRUST" at end, or "XXX TRUST," — but NOT "TRUST CORP", "TRUST INC"
    if ' TRUST' in title_up:
        # Keep if it's a company (REIT, bank, etc.)
        company_suffixes = ['CORP', 'INC', 'CO.', 'CO,', 'LTD', 'GROUP', 'BANCORP']
        is_company = any(suf in title_up for suf in company_suffixes)
        # Keep REITs (they're stocks, not ETFs)
        is_reit = any(kw in title_up for kw in ['REALTY', 'PROPERTY', 'INDUSTRIAL', 'ESSEX'])
        if not is_company and not is_reit:
            return True

    # Specific fund names
    if 'INVESCO QQQ' in title_up:
        return True

    # Skip likely OTC foreign ordinaries / unsponsored ADRs
    # Pattern: 5+ character tickers ending in F or Y (e.g. RTNTF, DTEGY, HTHIY)
    # This catches thin-volume OTC listings while preserving normal tickers
    # like F (Ford), COF (Capital One), SONY (4 chars), INFY (4 chars)
    if len(t) >= 5 and t[-1] in ('F', 'Y'):
        return True

    # Skip tickers with dots (foreign exchanges like .L, .TO, .DE)
    if '.' in t:
        return True

    return False


def load_tickers(filepath: str, limit: int = None) -> List[Dict[str, str]]:
    """Load tickers from JSON file, filtering out ETFs and OTC foreign ordinaries."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    tickers = []
    skipped = 0

    for key, value in data.items():
        ticker = value['ticker']
        title = value['title']

        if should_skip(ticker, title):
            skipped += 1
            continue

        tickers.append({
            'ticker': ticker,
            'title': title
        })
        if limit and len(tickers) >= limit:
            break

    print(f"📋 Filtered out {skipped} tickers (ETFs, trusts, OTC/ADR, foreign exchanges)")
    print(f"✅ {len(tickers)} stocks passed filter")

    return tickers


def rate_limited_sleep():
    """Enforce rate limiting"""
    global last_request_time

    current_time = time.time()
    time_since_last = current_time - last_request_time

    if time_since_last < REQUEST_DELAY:
        sleep_time = REQUEST_DELAY - time_since_last + random.uniform(0, 0.2)
        time.sleep(sleep_time)

    last_request_time = time.time()


def analyze_stock_with_retry(ticker: str, title: str, retry_count: int = 0) -> Dict:
    """Analyze with retry logic and batch-optimized shuffle count"""
    try:
        rate_limited_sleep()

        data = run_analysis(
            ticker=ticker,
            period='5y',
            window_days=5,
            account_size=DEFAULT_ACCOUNT_SIZE,
            risk_per_trade=DEFAULT_RISK_PER_TRADE,
            n_shuffles=BATCH_SHUFFLES
        )

        if data.get('error'):
            error_msg = str(data.get('error', '')).lower()
            if ('unauthorized' in error_msg or '401' in error_msg or 'crumb' in error_msg) and retry_count < MAX_RETRIES:
                print(f"   ⏳ Rate limited on {ticker}, waiting {RETRY_DELAY}s... (retry {retry_count + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
                return analyze_stock_with_retry(ticker, title, retry_count + 1)
            return None

        score = calculate_score(data)

        return {
            'ticker': ticker,
            'title': title,
            'score': score,
            'final_signal': data.get('final_signal'),
            'predictability_score': data.get('predictability_score', 0),
            'regime_stability': data.get('regime_stability'),
            'momentum_corr': data.get('momentum_corr'),
            'momentum_corr_oos': data.get('momentum_corr_oos'),
            'expected_edge_pct': data.get('expected_edge_pct'),
            'total_friction_pct': data.get('total_friction_pct'),
            'current': data.get('current'),
            'currency': data.get('currency'),
            'trend_direction': data.get('trend_direction'),
            'sharpe': data.get('sharpe'),
            'volatility': data.get('volatility'),
            'suggested_shares': data.get('suggested_shares'),
            'z_ema': data.get('z_ema'),
            'hurst': data.get('hurst'),
            'hurst_significant': data.get('hurst_significant'),
            'stop_loss_price': data.get('stop_loss_price'),
            'lb_pvalue': data.get('lb_pvalue'),
            'adf_pvalue': data.get('adf_pvalue'),
            'vp_ratio': data.get('vp_ratio'),
            'vp_confirming': data.get('vp_confirming'),
            'trade_quality': data.get('trade_quality'),
            'quality_label': data.get('quality_label'),
        }

    except Exception as e:
        error_msg = str(e).lower()
        if ('unauthorized' in error_msg or '401' in error_msg or 'crumb' in error_msg) and retry_count < MAX_RETRIES:
            print(f"   ⏳ Rate limited on {ticker}, waiting {RETRY_DELAY}s... (retry {retry_count + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
            return analyze_stock_with_retry(ticker, title, retry_count + 1)
        return None


def calculate_score(data: Dict) -> float:
    """
    Calculate composite score for ranking.

    Score components:
      Predictability:   0-40 pts  (10 per test passed)
      Regime stability: 0-20 pts  (existence-based: 0/0.5/1.0 × 20)
      Edge vs friction: 0-20 pts  (bonus for edge > 3× friction)
      Signal quality:   -50 to 20  (signal-dependent)
      Liquidity bonus:  0-10 pts
      Volatility:       -5 to 0    (penalty for >50%)
    """
    score = 0.0

    # Predictability (0-40)
    score += data.get('predictability_score', 0) * 10

    # Regime stability (0-20) — now discrete: 0.0, 0.5, 1.0
    if data.get('regime_stability') is not None:
        score += data.get('regime_stability') * 20

    # Edge vs friction (0-20)
    edge = data.get('expected_edge_pct', 0)
    friction = data.get('total_friction_pct', 0)
    if friction > 0:
        ratio = edge / friction
        if ratio > 3:
            score += min(20, (ratio - 3) * 4)

    # Signal quality (0-20) — includes speculative signals
    signal_scores = {
        # High conviction
        'BUY_UPTREND': 20, 'BUY_PULLBACK': 20,
        'SHORT_DOWNTREND': 18, 'BUY_MOMENTUM': 15, 'SHORT_MOMENTUM': 15,
        'SHORT_BOUNCES_ONLY': 12,
        'WAIT_PULLBACK': 8, 'WAIT_SHORT_BOUNCE': 8,
        'WAIT_OR_SHORT_BOUNCE': 5, 'WAIT_FOR_REVERSAL': 5, 'WAIT_FOR_TREND': 3,
        # Speculative (same base scores but lower due to 2/5 predictability)
        'SPEC_BUY_UPTREND': 15, 'SPEC_BUY_PULLBACK': 15,
        'SPEC_SHORT_DOWNTREND': 13, 'SPEC_BUY_MOMENTUM': 10, 'SPEC_SHORT_MOMENTUM': 10,
        'SPEC_SHORT_BOUNCES_ONLY': 8,
        'SPEC_WAIT_OR_SHORT_BOUNCE': 3, 'SPEC_WAIT_FOR_REVERSAL': 3,
        # Rejected
        'NO_CLEAR_SIGNAL': 0, 'DO_NOT_TRADE': -50
    }
    score += signal_scores.get(data.get('final_signal', ''), 0)

    # Volatility penalty
    if data.get('volatility') and data.get('volatility') > 50:
        score -= 5

    return score


def get_signal_icon(signal: str) -> str:
    """Map signal to display icon"""
    if not signal:
        return '⚪'

    if signal.startswith('SPEC_'):
        # Speculative tier — orange
        if 'BUY' in signal:
            return '🟡'
        elif 'SHORT' in signal:
            return '🟠'
        else:
            return '🟡'
    elif 'BUY' in signal:
        return '🟢'
    elif 'SHORT' in signal:
        return '🔴'
    elif 'WAIT' in signal:
        return '🟡'
    else:
        return '⚪'


def get_signal_category(signal: str) -> str:
    """Categorize signal for counting"""
    if not signal:
        return 'none'
    if signal.startswith('SPEC_'):
        return 'speculative'
    if 'BUY' in signal:
        return 'buy'
    if 'SHORT' in signal:
        return 'short'
    if 'WAIT' in signal:
        return 'wait'
    return 'none'


def analyze_batch(tickers: List[Dict]) -> List[Dict]:
    """Analyze stocks sequentially with rate limiting"""
    results = []

    est_minutes = len(tickers) * 1.1 / 60  # ~1.1s per stock with overhead

    print(f"\n{'='*80}")
    print(f"Analyzing {len(tickers)} stocks (filtered from top {TOP_N_STOCKS} by market cap)")
    print(f"Rate limit: {REQUEST_DELAY}s per request | Hurst shuffles: {BATCH_SHUFFLES}")
    print(f"Estimated time: ~{est_minutes:.0f}-{est_minutes*1.5:.0f} minutes")
    print(f"{'='*80}\n")

    start_time = time.time()
    counts = {'buy': 0, 'short': 0, 'wait': 0, 'speculative': 0}

    for i, ticker_info in enumerate(tickers, 1):
        result = analyze_stock_with_retry(ticker_info['ticker'], ticker_info['title'])

        if result:
            results.append(result)

            signal = result.get('final_signal', '')
            icon = get_signal_icon(signal)
            cat = get_signal_category(signal)
            if cat in counts:
                counts[cat] += 1

            # Calculate ETA
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = len(tickers) - i
            eta_minutes = (remaining / rate / 60) if rate > 0 else 0

            # Truncate signal for display
            display_signal = signal[:22] if signal else 'N/A'

            print(f"[{i:3d}/{len(tickers)}] {icon} {ticker_info['ticker']:6s} "
                  f"Score: {result['score']:6.1f} | {display_signal:22s} "
                  f"| ETA: {eta_minutes:4.1f}m")
        else:
            print(f"[{i:3d}/{len(tickers)}] ✗ {ticker_info['ticker']:6s} FAILED")

        # Progress update every 50 stocks
        if i % 50 == 0:
            elapsed_mins = (time.time() - start_time) / 60
            print(f"\n📊 Progress: {i}/{len(tickers)} | "
                  f"Tradeable: {len(results)} | "
                  f"🟢 {counts['buy']} | 🔴 {counts['short']} | "
                  f"🟡 {counts['wait']} | 🟠 {counts['speculative']} spec | "
                  f"Time: {elapsed_mins:.1f}m\n")

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


def save_results(results: List[Dict], filename: str = "top_stocks.json"):
    """Save to JSON"""
    tradeable = [r for r in results if r.get('final_signal') not in ('DO_NOT_TRADE', 'NO_CLEAR_SIGNAL', None)]

    output = {
        'timestamp': datetime.now().isoformat(),
        'total_analyzed': len(results),
        'total_tradeable': len(tradeable),
        'config': {
            'n_shuffles': BATCH_SHUFFLES,
            'top_n': TOP_N_STOCKS,
            'account_size': DEFAULT_ACCOUNT_SIZE,
            'risk_per_trade': DEFAULT_RISK_PER_TRADE,
        },
        'stocks': results[:min(len(results),150)]  # Top 150 for review
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Saved top 150 to {filename}")
    return filename


def print_summary(results: List[Dict]):
    """Print categorized results"""

    # Separate by tier
    high_conviction = [r for r in results if r.get('final_signal') and
                       not r['final_signal'].startswith('SPEC_') and
                       r['final_signal'] not in ('DO_NOT_TRADE', 'NO_CLEAR_SIGNAL')]
    speculative = [r for r in results if r.get('final_signal', '').startswith('SPEC_')]

    # ── HIGH CONVICTION ──────────────────────────────────────────────────
    if high_conviction:
        print(f"\n{'='*100}")
        print(f"🏆 HIGH CONVICTION SIGNALS ({len(high_conviction)} stocks) — 3-5/5 predictability")
        print(f"{'='*100}")
        print(f"{'#':<4} {'Ticker':<8} {'Score':<7} {'Signal':<24} {'Edge%':<8} {'Pred':<5} {'Stab':<6} {'Qual':<6} {'Trend':<6}")
        print(f"{'-'*100}")

        for i, s in enumerate(high_conviction[:30], 1):
            signal = s['final_signal'][:22] if s['final_signal'] else 'N/A'
            edge = f"{s.get('expected_edge_pct', 0):.1f}%" if s.get('expected_edge_pct') else 'N/A'
            stab = f"{s.get('regime_stability', 0)*100:.0f}%" if s.get('regime_stability') is not None else 'N/A'
            qual = f"{s.get('trade_quality', 0):.1f}" if s.get('trade_quality') is not None else 'N/A'

            if 'BUY' in signal:
                sig = f"\033[92m{signal:24s}\033[0m"
            elif 'SHORT' in signal:
                sig = f"\033[91m{signal:24s}\033[0m"
            elif 'WAIT' in signal:
                sig = f"\033[93m{signal:24s}\033[0m"
            else:
                sig = f"{signal:24s}"

            print(f"{i:<4} {s['ticker']:<8} {s['score']:<7.1f} {sig} {edge:<8} "
                  f"{s['predictability_score']}/5   {stab:<6} {qual:<6} {s.get('trend_direction', 'N/A'):<6}")
    else:
        print(f"\n⚪ No high-conviction signals found.")

    # ── SPECULATIVE ──────────────────────────────────────────────────────
    if speculative:
        print(f"\n{'='*100}")
        print(f"⚠  SPECULATIVE SIGNALS ({len(speculative)} stocks) — 2/5 predictability, half position")
        print(f"{'='*100}")
        print(f"{'#':<4} {'Ticker':<8} {'Score':<7} {'Signal':<24} {'Edge%':<8} {'Pred':<5} {'Stab':<6} {'Qual':<6} {'Trend':<6}")
        print(f"{'-'*100}")

        for i, s in enumerate(speculative[:30], 1):
            signal = s['final_signal'][:22] if s['final_signal'] else 'N/A'
            edge = f"{s.get('expected_edge_pct', 0):.1f}%" if s.get('expected_edge_pct') else 'N/A'
            stab = f"{s.get('regime_stability', 0)*100:.0f}%" if s.get('regime_stability') is not None else 'N/A'
            qual = f"{s.get('trade_quality', 0):.1f}" if s.get('trade_quality') is not None else 'N/A'

            sig = f"\033[93m{signal:24s}\033[0m"  # Orange/yellow

            print(f"{i:<4} {s['ticker']:<8} {s['score']:<7.1f} {sig} {edge:<8} "
                  f"{s['predictability_score']}/5   {stab:<6} {qual:<6} {s.get('trend_direction', 'N/A'):<6}")
    else:
        print(f"\n⚪ No speculative signals found.")

    # ── OVERALL STATS ────────────────────────────────────────────────────
    all_signals = [r.get('final_signal', '') for r in results]
    buy = sum(1 for s in all_signals if 'BUY' in s and not s.startswith('SPEC_'))
    short = sum(1 for s in all_signals if 'SHORT' in s and not s.startswith('SPEC_'))
    wait = sum(1 for s in all_signals if 'WAIT' in s and not s.startswith('SPEC_'))
    spec_buy = sum(1 for s in all_signals if s.startswith('SPEC_') and 'BUY' in s)
    spec_short = sum(1 for s in all_signals if s.startswith('SPEC_') and 'SHORT' in s)
    spec_wait = sum(1 for s in all_signals if s.startswith('SPEC_') and 'WAIT' in s)
    dnt = sum(1 for s in all_signals if s in ('DO_NOT_TRADE', 'NO_CLEAR_SIGNAL', ''))

    print(f"\n{'='*60}")
    print(f"📈 Signal Distribution ({len(results)} analyzed)")
    print(f"{'='*60}")
    print(f"   HIGH CONVICTION:")
    print(f"      🟢 BUY:   {buy:3d} stocks")
    print(f"      🔴 SHORT: {short:3d} stocks")
    print(f"      🟡 WAIT:  {wait:3d} stocks")
    print(f"   SPECULATIVE (half position):")
    print(f"      🟢 BUY:   {spec_buy:3d} stocks")
    print(f"      🔴 SHORT: {spec_short:3d} stocks")
    print(f"      🟡 WAIT:  {spec_wait:3d} stocks")
    print(f"   ⚪ DO NOT TRADE: {dnt:3d} stocks")
    print(f"{'='*60}\n")


def main():
    print(f"\n{'='*100}")
    print("STOCKAURA — TOP MARKET CAP ANALYZER")
    print(f"Scanning top {TOP_N_STOCKS} by market cap (filtering ETFs, OTC/ADR)")
    print(f"Hurst shuffles: {BATCH_SHUFFLES} (batch mode) | Min predictability: 2/5")
    print(f"{'='*100}\n")

    print("📂 Loading tickers...")
    tickers = load_tickers(TICKERS_FILE, limit=TOP_N_STOCKS)

    est_minutes = len(tickers) * 1.1 / 60
    input(f"\n⏸  Press ENTER to start analysis (~{est_minutes:.0f}-{est_minutes*1.5:.0f} minutes)...")

    start = time.time()
    results = analyze_batch(tickers)
    elapsed = (time.time() - start) / 60

    tradeable = [r for r in results if r.get('final_signal') not in ('DO_NOT_TRADE', 'NO_CLEAR_SIGNAL', None)]

    print(f"\n⏱  Total time: {elapsed:.1f} minutes")
    print(f"📊 Analyzed: {len(tickers)} stocks")
    print(f"✅ Tradeable: {len(tradeable)} opportunities ({len([r for r in tradeable if not r['final_signal'].startswith('SPEC_')])} high conviction, {len([r for r in tradeable if r['final_signal'].startswith('SPEC_')])} speculative)")

    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()
