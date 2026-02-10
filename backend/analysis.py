import yfinance as yf
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def format_number(num):
    num = float(num)
    if abs(num) >= 1e12:
        return f'{num/1e12:.2f}T'
    elif abs(num) >= 1e9:
        return f'{num/1e9:.2f}B'
    elif abs(num) >= 1e6:
        return f'{num/1e6:.2f}M'
    elif abs(num) >= 1e3:
        return f'{num/1e3:.2f}K'
    else:
        return f'{num}'


def dfa_hurst(series, min_box=10, max_box=None, num_scales=20):
    """
    Detrended Fluctuation Analysis (DFA) to estimate Hurst exponent.
    
    More robust than R/S analysis:
    - Less sensitive to short-term correlations and trends
    - Better performance on finite-length series
    - More stable across different parameter choices
    
    Returns: H, scales, fluctuations, poly
    - H: Hurst exponent (slope of log-log plot)
    - scales: box sizes used
    - fluctuations: DFA fluctuation at each scale
    - poly: polynomial fit coefficients
    """
    N = len(series)
    if max_box is None:
        max_box = N // 4  # Use at most 1/4 of series length
    
    if max_box <= min_box or N < min_box * 4:
        return np.nan, None, None, None
    
    # Step 1: Integrate the mean-centered series (cumulative sum of deviations)
    y = np.cumsum(series - np.mean(series))
    
    # Step 2: Generate logarithmically spaced box sizes
    scales = np.unique(
        np.logspace(np.log10(min_box), np.log10(max_box), num=num_scales).astype(int)
    )
    # Filter out scales that are too large
    scales = scales[scales <= N // 2]
    
    if len(scales) < 4:
        return np.nan, None, None, None
    
    fluctuations = []
    
    for box_size in scales:
        # Number of non-overlapping boxes
        n_boxes = N // box_size
        if n_boxes < 2:
            continue
        
        rms_values = []
        
        # Forward pass
        for i in range(n_boxes):
            start = i * box_size
            end = start + box_size
            segment = y[start:end]
            
            # Fit linear trend to segment
            x = np.arange(box_size)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            
            # Calculate RMS of detrended segment
            rms = np.sqrt(np.mean((segment - trend) ** 2))
            rms_values.append(rms)
        
        # Backward pass (use remaining data from the end)
        for i in range(n_boxes):
            start = N - (i + 1) * box_size
            end = start + box_size
            if start < 0:
                break
            segment = y[start:end]
            
            x = np.arange(box_size)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            
            rms = np.sqrt(np.mean((segment - trend) ** 2))
            rms_values.append(rms)
        
        if len(rms_values) > 0:
            # Overall fluctuation for this scale
            F = np.sqrt(np.mean(np.array(rms_values) ** 2))
            if F > 0:
                fluctuations.append(F)
            else:
                fluctuations.append(np.nan)
        else:
            fluctuations.append(np.nan)
    
    # Trim scales to match valid fluctuations
    scales = scales[:len(fluctuations)]
    fluctuations = np.array(fluctuations)
    
    # Remove NaN entries
    valid = ~np.isnan(fluctuations) & (fluctuations > 0)
    scales = scales[valid]
    fluctuations = fluctuations[valid]
    
    if len(scales) < 4:
        return np.nan, None, None, None
    
    # Step 3: Log-log fit to get Hurst exponent
    log_scales = np.log(scales.astype(float))
    log_fluct = np.log(fluctuations)
    
    poly = np.polyfit(log_scales, log_fluct, 1)
    H = poly[0]
    
    return H, scales, fluctuations, poly


def hurst_with_baseline(series, n_shuffles=50, **kwargs):
    """
    Compute DFA Hurst exponent with shuffled baseline comparison.
    
    Shuffling destroys temporal structure while preserving distribution.
    If the real Hurst is not significantly different from shuffled Hurst,
    the detected regime (trending/mean-reverting) is likely noise.
    
    Returns: H, H_shuffled_mean, H_shuffled_std, is_significant, scales, fluctuations, poly
    - H: real Hurst exponent
    - H_shuffled_mean: mean Hurst from shuffled series (should be ~0.5)
    - H_shuffled_std: std of shuffled Hurst values
    - is_significant: True if |H - 0.5| is statistically distinguishable from random
    """
    # Real Hurst
    H, scales, fluctuations, poly = dfa_hurst(series, **kwargs)
    
    if np.isnan(H):
        return H, np.nan, np.nan, False, None, None, None
    
    # Shuffled baseline
    shuffled_hursts = []
    rng = np.random.default_rng(42)  # Reproducible
    
    for _ in range(n_shuffles):
        shuffled = rng.permutation(series)
        H_shuf, _, _, _ = dfa_hurst(shuffled, **kwargs)
        if not np.isnan(H_shuf):
            shuffled_hursts.append(H_shuf)
    
    if len(shuffled_hursts) < 10:
        # Not enough valid shuffles — can't assess significance
        return H, np.nan, np.nan, False, scales, fluctuations, poly
    
    H_shuf_mean = np.mean(shuffled_hursts)
    H_shuf_std = np.std(shuffled_hursts)
    
    # Significant if real H is >1.5 standard deviations from shuffled mean
    if H_shuf_std > 0:
        z_score = abs(H - H_shuf_mean) / H_shuf_std
        is_significant = z_score > 1.5
    else:
        is_significant = False
    
    return H, H_shuf_mean, H_shuf_std, is_significant, scales, fluctuations, poly


def multi_day_momentum_corr(daily_returns, block_days=3):
    """
    Calculate momentum correlation using NON-OVERLAPPING multi-day blocks.
    
    Measures: "Does the direction of this 3-day period predict the next 3-day period?"
    
    Returns: (correlation, n_pairs) or (None, 0)
    """
    n = len(daily_returns)
    n_blocks = n // block_days
    
    if n_blocks < 20:  # Need at least 10 pairs
        return None, 0
    
    # Build non-overlapping block returns
    blocks = []
    for i in range(n_blocks):
        start = i * block_days
        end = start + block_days
        block_return = np.prod(1 + daily_returns[start:end]) - 1
        blocks.append(block_return)
    
    blocks = np.array(blocks)
    
    # Consecutive block pairs: does block[i] predict block[i+1]?
    x = blocks[:-1]
    y = blocks[1:]
    
    if np.std(x) == 0 or np.std(y) == 0:
        return None, 0
    
    corr = np.corrcoef(x, y)[0, 1]
    
    if np.isnan(corr):
        return None, 0
    
    return float(corr), len(x)


def non_overlapping_mean_reversion(returns, window_days):
    """
    Mean reversion analysis using non-overlapping windows.
    
    After a large up/down block, what happens in the NEXT block?
    
    Returns: (mean_rev_up, mean_rev_down) or (None, None)
    """
    n = len(returns)
    n_blocks = n // window_days
    
    if n_blocks < 6:
        return None, None
    
    block_returns = []
    for i in range(n_blocks):
        start = i * window_days
        end = start + window_days
        block = returns[start:end]
        block_ret = np.prod(1 + block) - 1
        block_returns.append(block_ret)
    
    block_returns = np.array(block_returns)
    
    if len(block_returns) < 6:
        return None, None
    
    q75 = np.percentile(block_returns, 75)
    q25 = np.percentile(block_returns, 25)
    
    mean_rev_up = None
    mean_rev_down = None
    
    # After large up blocks
    up_next = []
    for i in range(len(block_returns) - 1):
        if block_returns[i] > q75:
            up_next.append(block_returns[i + 1])
    
    if len(up_next) > 0:
        mean_rev_up = float(np.mean(up_next))
    
    # After large down blocks
    down_next = []
    for i in range(len(block_returns) - 1):
        if block_returns[i] < q25:
            down_next.append(block_returns[i + 1])
    
    if len(down_next) > 0:
        mean_rev_down = float(np.mean(down_next))
    
    return mean_rev_up, mean_rev_down


def volume_price_confirmation(df, lookback=60):
    """
    Volume-Price Confirmation Test (5th predictability test)
    """
    try:
        recent = df.tail(lookback).copy()
        
        if len(recent) < 20:
            return None
        
        recent['daily_return'] = recent['Close'].pct_change()
        recent = recent.dropna(subset=['daily_return', 'Volume'])
        
        up_days = recent[recent['daily_return'] > 0]
        down_days = recent[recent['daily_return'] < 0]
        
        if len(up_days) < 5 or len(down_days) < 5:
            return None
        
        avg_vol_up = float(up_days['Volume'].mean())
        avg_vol_down = float(down_days['Volume'].mean())
        
        if avg_vol_down == 0:
            return None
        
        vp_ratio = avg_vol_up / avg_vol_down
        
        # Determine recent trend from last 63 trading days (3 months)
        if len(df) >= 63:
            price_now = float(df['Close'].iloc[-1])
            price_3m = float(df['Close'].iloc[-63])
            trend_3m = (price_now - price_3m) / price_3m
        else:
            trend_3m = 0
        
        if trend_3m > 0.03:
            trend_for_vp = 'UP'
            vp_confirming = vp_ratio > 1.10
        elif trend_3m < -0.03:
            trend_for_vp = 'DOWN'
            vp_confirming = vp_ratio < 0.90
        else:
            trend_for_vp = 'NEUTRAL'
            vp_confirming = False
        
        return {
            'vp_ratio': round(vp_ratio, 3),
            'vp_confirming': vp_confirming,
            'avg_vol_up': avg_vol_up,
            'avg_vol_down': avg_vol_down,
            'trend_for_vp': trend_for_vp,
        }
    except Exception:
        return None


def calculate_trade_quality(res):
    """
    Trade Quality Score (0-10) — How good is the current setup?
    
    Components (each 0-2 points):
    1. Multi-timeframe trend alignment
    2. Entry timing via Z-EMA
    3. Risk-adjusted returns (Sharpe quality)
    4. Volatility appropriateness
    5. Volume-price confirmation
    """
    components = {}
    total = 0.0
    
    # ── 1. MULTI-TIMEFRAME ALIGNMENT (0-2) ──────────────────────────────
    returns = []
    for key in ['recent_return_1m', 'recent_return_3m', 'recent_return_6m', 'recent_return_1y']:
        val = res.get(key)
        if val is not None:
            returns.append(val)
    
    if len(returns) >= 3:
        positive = sum(1 for r in returns if r > 0.02)
        negative = sum(1 for r in returns if r < -0.02)
        total_periods = len(returns)
        
        alignment = max(positive, negative) / total_periods
        
        if alignment >= 1.0:
            components['trend_alignment'] = 2.0
        elif alignment >= 0.75:
            components['trend_alignment'] = 1.5
        elif alignment >= 0.5:
            components['trend_alignment'] = 0.8
        else:
            components['trend_alignment'] = 0.3
    else:
        components['trend_alignment'] = 0.5
    
    total += components['trend_alignment']
    
    # ── 2. ENTRY TIMING VIA Z-EMA (0-2) ─────────────────────────────────
    z_ema = res.get('z_ema')
    trend = res.get('trend_direction')
    
    if z_ema is not None:
        abs_z = abs(z_ema)
        
        if trend == 'UP':
            if z_ema < -0.5:
                components['entry_timing'] = 2.0
            elif z_ema < 0.5:
                components['entry_timing'] = 1.5
            elif z_ema < 1.0:
                components['entry_timing'] = 1.0
            elif z_ema < 1.5:
                components['entry_timing'] = 0.5
            else:
                components['entry_timing'] = 0.0
        elif trend == 'DOWN':
            if z_ema > 0.5:
                components['entry_timing'] = 2.0
            elif z_ema > -0.5:
                components['entry_timing'] = 1.5
            elif z_ema > -1.0:
                components['entry_timing'] = 1.0
            elif z_ema > -1.5:
                components['entry_timing'] = 0.5
            else:
                components['entry_timing'] = 0.0
        else:
            if abs_z < 0.5:
                components['entry_timing'] = 1.5
            elif abs_z < 1.0:
                components['entry_timing'] = 1.0
            else:
                components['entry_timing'] = 0.3
    else:
        components['entry_timing'] = 0.5
    
    total += components['entry_timing']
    
    # ── 3. RISK-ADJUSTED RETURNS / SHARPE (0-2) ─────────────────────────
    sharpe = res.get('sharpe')
    
    if sharpe is not None:
        if sharpe > 1.5:
            components['sharpe_quality'] = 2.0
        elif sharpe > 1.0:
            components['sharpe_quality'] = 1.5
        elif sharpe > 0.5:
            components['sharpe_quality'] = 1.0
        elif sharpe > 0:
            components['sharpe_quality'] = 0.5
        elif sharpe > -0.5:
            components['sharpe_quality'] = 0.2
        else:
            components['sharpe_quality'] = 0.0
    else:
        components['sharpe_quality'] = 0.5
    
    total += components['sharpe_quality']
    
    # ── 4. VOLATILITY APPROPRIATENESS (0-2) ──────────────────────────────
    vol = res.get('volatility')
    
    if vol is not None:
        if 20 <= vol <= 35:
            components['volatility_fit'] = 2.0
        elif 15 <= vol <= 45:
            components['volatility_fit'] = 1.5
        elif 10 <= vol <= 55:
            components['volatility_fit'] = 0.8
        else:
            components['volatility_fit'] = 0.3
    else:
        components['volatility_fit'] = 0.5
    
    total += components['volatility_fit']
    
    # ── 5. VOLUME-PRICE CONFIRMATION (0-2) ───────────────────────────────
    vp = res.get('volume_price_data')
    
    if vp is not None and vp.get('vp_confirming') is not None:
        if vp['vp_confirming']:
            ratio = vp['vp_ratio']
            if vp['trend_for_vp'] == 'UP':
                if ratio > 1.3:
                    components['volume_confirmation'] = 2.0
                elif ratio > 1.15:
                    components['volume_confirmation'] = 1.5
                else:
                    components['volume_confirmation'] = 1.0
            elif vp['trend_for_vp'] == 'DOWN':
                if ratio < 0.7:
                    components['volume_confirmation'] = 2.0
                elif ratio < 0.85:
                    components['volume_confirmation'] = 1.5
                else:
                    components['volume_confirmation'] = 1.0
            else:
                components['volume_confirmation'] = 0.5
        else:
            components['volume_confirmation'] = 0.2
    else:
        components['volume_confirmation'] = 0.5
    
    total += components['volume_confirmation']
    
    # ── FINAL SCORE ──────────────────────────────────────────────────────
    total = round(min(10.0, total), 1)
    
    if total >= 8.0:
        label = 'Excellent'
    elif total >= 6.0:
        label = 'Good'
    elif total >= 4.0:
        label = 'Fair'
    else:
        label = 'Poor'
    
    return {
        'trade_quality': total,
        'quality_components': components,
        'quality_label': label,
    }


def calculate_amihud_illiquidity(df):
    """
    Calculate Amihud Illiquidity ratio: |Return| / (Volume * Price)
    """
    try:
        df_temp = df.copy()
        df_temp['abs_return'] = np.abs(df_temp['Return'])
        df_temp['volume_times_price'] = df_temp['Volume'] * df_temp['Close']
        df_temp['illiquidity'] = df_temp['abs_return'] / (df_temp['volume_times_price'] + 1e-10)
        
        amihud = df_temp['illiquidity'].tail(30).mean()
        return float(amihud)
    except Exception:
        return None


def calculate_dynamic_slippage(df):
    """
    Estimate slippage based on volatility and daily price range.
    """
    try:
        df_temp = df.copy()
        df_temp['daily_range_pct'] = (df_temp['High'] - df_temp['Low']) / df_temp['Close']
        avg_daily_range = df_temp['daily_range_pct'].tail(30).mean()
        
        estimated_slippage = avg_daily_range * 0.05
        return float(estimated_slippage)
    except Exception:
        return 0.0005


def get_liquidity_score(amihud_illiquidity, position_size_vs_volume):
    """
    Determine liquidity quality based on Amihud ratio and position size
    """
    if amihud_illiquidity is None or position_size_vs_volume is None:
        return 'UNKNOWN'
    
    if amihud_illiquidity < 0.001 and position_size_vs_volume < 0.005:
        return 'HIGH'
    
    if amihud_illiquidity < 0.01 and position_size_vs_volume < 0.02:
        return 'MEDIUM'
    
    return 'LOW'


def get_liquidity_warning(liquidity_score, position_size_vs_volume, amihud_illiquidity):
    """
    Generate warning message if liquidity is concerning.
    These are ADVISORY warnings — they do NOT block trading signals.
    """
    warnings = []
    
    if position_size_vs_volume > 0.10:
        warnings.append(f"Position is {position_size_vs_volume*100:.1f}% of daily volume — expect significant market impact and slippage")
    elif position_size_vs_volume > 0.05:
        warnings.append(f"Position is {position_size_vs_volume*100:.2f}% of daily volume — may cause noticeable slippage")
    elif position_size_vs_volume > 0.02:
        warnings.append(f"Position is {position_size_vs_volume*100:.2f}% of daily volume — minor slippage possible")
    
    if amihud_illiquidity and amihud_illiquidity > 0.01:
        warnings.append("Stock is illiquid — high price impact on large orders")
    
    return ' | '.join(warnings) if warnings else None


exchange_to_currency = {'T': 'JPY', 'NYB': '', 'CO': 'DKK', 'L': 'GBP or GBX', 'DE': 'EUR', 'PA': 'EUR', 'TO': 'CAD', 'V': 'CAD'}

def analyze_stock(ticker, period="5y", window_days=5, account_size=10000, risk_per_trade=0.02, n_shuffles=50):
    try:
        df = yf.download([ticker], period=period, progress=False)
    except Exception:
        return {"error": "Connection error with data provider", "ticker": ticker}

    if df.empty:
        return {"error": "No data found, symbol may be delisted", "ticker": ticker}

    if len(df) < 252:
        return {"error": "Insufficient historical data", "ticker": ticker}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    yfticker = yf.Ticker(ticker)
    tmp = ticker.split('.')
    currency = 'USD'
    if len(tmp) == 2:
        currency = exchange_to_currency[tmp[1]]
    title = yfticker.info.get('longName')
    current = yfticker.info.get('currentPrice')
    cap = format_number(yfticker.info.get('marketCap'))
    if current is None or current == 0:
        current = round(float(df['Close'].iloc[-1]),2)
    
    # ═══════════════════════════════════════════════════════════════════════
    # FIX #1: REMOVED the early return that blocked the entire results page
    # when account_size < share price. Now we always run the full analysis
    # and handle affordability in the position sizing section instead.
    # ═══════════════════════════════════════════════════════════════════════
    
    OHLC = df.reset_index()[['Date','Open', 'High', 'Close', 'Low']]
    OHLC['Date'] = pd.to_datetime(OHLC['Date']).dt.strftime('%Y-%m-%d')

    df['Return'] = df['Close'].pct_change()

    returns = df['Return'].dropna().values

    res = {
        'ticker': ticker,
        'window_days': window_days,
        'period': period,
        'hurst': None,
        'hurst_oos': None,
        'hurst_significant': None,
        'hurst_shuffled_mean': None,
        'momentum_corr': None,
        'momentum_corr_oos': None,
        'lb_pvalue': None,
        'adf_pvalue': None,
        'mean_rev_up': None,
        'mean_rev_down': None,
        'mean_rev_up_oos': None,
        'mean_rev_down_oos': None,
        'sharpe': None,
        'volatility': None,
        'Return': None,
        'predictability_score': 0,
        'zscore': None,
        'z_ema': None,
        'regime_stability': None,
        'recent_return_1y': None,
        'recent_return_6m': None,
        'recent_return_3m': None,
        'recent_return_1m': None,
        'trend_direction': None,
        # Volume-Price Confirmation
        'volume_price_data': None,
        'vp_ratio': None,
        'vp_confirming': None,
        # Trade Quality Score
        'trade_quality': None,
        'quality_components': None,
        'quality_label': None,
        # Position Sizing Fields
        'suggested_shares': None,
        'stop_loss_price': None,
        'position_risk_amount': None,
        'position_size_warning': None,
        'volatility_category': None,
        'final_signal': None,
        # Liquidity & Friction Analysis
        'avg_daily_volume': None,
        'amihud_illiquidity': None,
        'liquidity_score': None,
        'position_size_vs_volume': None,
        'estimated_slippage_pct': None,
        'total_friction_pct': None,
        'expected_edge_pct': None,
        'is_liquid_enough': None,
        'liquidity_failed': False,
        'calculated_shares': None,
        'liquidity_warning': None,
        'title': title,
        'current': current,
        'cap': cap,
        'currency': currency,
        'OHLC': OHLC.to_dict('records'),
        'data_points': len(df),
        'transaction_cost': 0.001,  # 0.1% per trade
        'slippage': 0.0005,  # 0.05%
        'risk_per_trade': risk_per_trade,
        'account_size_input': account_size,
    }

    # Split data: 70% train, 30% test for out-of-sample validation
    split_idx = int(len(df) * 0.7)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    returns_train = df_train['Return'].dropna().values
    returns_test = df_test['Return'].dropna().values

    # CALCULATE RECENT RETURNS
    current = df['Close'].iloc[-1]
    
    if len(df) >= 252:
        price_1y_ago = df['Close'].iloc[-252]
        recent_return_1y = (current - price_1y_ago) / price_1y_ago
        res['recent_return_1y'] = float(recent_return_1y)
    
    if len(df) >= 126:
        price_6m_ago = df['Close'].iloc[-126]
        recent_return_6m = (current - price_6m_ago) / price_6m_ago
        res['recent_return_6m'] = float(recent_return_6m)
    
    if len(df) >= 63:
        price_3m_ago = df['Close'].iloc[-63]
        recent_return_3m = (current - price_3m_ago) / price_3m_ago
        res['recent_return_3m'] = float(recent_return_3m)
    
    if len(df) >= 21:
        price_1m_ago = df['Close'].iloc[-21]
        recent_return_1m = (current - price_1m_ago) / price_1m_ago
        res['recent_return_1m'] = float(recent_return_1m)
    
    # Determine trend direction
    if res['recent_return_1y'] is not None:
        if res['recent_return_1y'] > 0.05:
            res['trend_direction'] = 'UP'
        elif res['recent_return_1y'] < -0.05:
            res['trend_direction'] = 'DOWN'
        else:
            res['trend_direction'] = 'NEUTRAL'

    # Z-SCORE (Simple 20-day MA)
    if len(df) >= 20:
        roll_m = df['Close'].rolling(window=20).mean()
        roll_s = df['Close'].rolling(window=20).std()
        curr_std = roll_s.iloc[-1]
        if curr_std > 0:
            z = (df['Close'].iloc[-1] - roll_m.iloc[-1]) / curr_std
            res['zscore'] = float(z)

    # Z-EMA (Exponential 20-day MA)
    if len(df) >= 20:
        ema_m = df['Close'].ewm(span=20).mean()
        ema_s = df['Close'].ewm(span=20).std()
        ema_std = ema_s.iloc[-1]
        if ema_std > 0:
            z_ema = (df['Close'].iloc[-1] - ema_m.iloc[-1]) / ema_std
            res['z_ema'] = float(z_ema)

    # Risk & Performance Metrics
    if len(returns) > 2:
        std_dev = returns.std()
        avg_ret = returns.mean()
        res['volatility'] = float(std_dev * np.sqrt(252) * 100)
        res['Return'] = float(avg_ret * 252 * 100)
        if std_dev > 0:
            res['sharpe'] = float(avg_ret / std_dev * np.sqrt(252))

    # VOLATILITY CATEGORY
    if res['volatility'] is not None:
        vol = res['volatility']
        if vol < 15:
            res['volatility_category'] = 'VERY_LOW'
        elif vol < 25:
            res['volatility_category'] = 'LOW'
        elif vol < 35:
            res['volatility_category'] = 'MODERATE'
        elif vol < 50:
            res['volatility_category'] = 'HIGH'
        else:
            res['volatility_category'] = 'VERY_HIGH'

    # ═══════════════════════════════════════════════════════════════════════
    # POSITION SIZING — New model: Trade Size + Risk Tolerance
    #
    # Parameters (renamed for clarity):
    #   account_size  → "Trade Size": capital allocated to THIS trade
    #   risk_per_trade → "Risk Tolerance": % drawdown that triggers stop loss
    #
    # How it works:
    #   Shares = trade_size / share_price  (you spend what you allocated)
    #   Stop loss (long)  = entry * (1 - risk_tolerance)
    #   Stop loss (short) = entry * (1 + risk_tolerance)
    #   Risk amount = trade_size * risk_tolerance
    #
    # At 100% risk: no stop loss for longs (accept total loss)
    # At 2% risk: tight stop, small position risk
    # ═══════════════════════════════════════════════════════════════════════
    if current and current > 0:
        trade_size = account_size  # Renamed for clarity in this block
        
        # How many whole shares can we buy with our trade size?
        exact_shares = trade_size / current
        whole_shares = int(exact_shares)
        
        # Store for liquidity analysis
        res['calculated_shares'] = exact_shares
        
        if whole_shares >= 1:
            res['suggested_shares'] = whole_shares
            
            position_value = whole_shares * current
            
            # Stop loss based on risk tolerance (user-specified %)
            stop_loss_dist = current * risk_per_trade
            
            if risk_per_trade >= 1.0:
                # 100% risk = no stop loss for longs (accept total loss)
                res['stop_loss_price_long'] = None
                res['stop_loss_price_short'] = None
                res['stop_loss_price'] = None
            else:
                res['stop_loss_price_long'] = float(current - stop_loss_dist)
                res['stop_loss_price_short'] = float(current + stop_loss_dist)
                res['stop_loss_price'] = float(current - stop_loss_dist)  # Default to long
            
            # Risk amount = what you lose if stop is hit
            risk_amount = position_value * risk_per_trade
            res['position_risk_amount'] = float(risk_amount)
            
            notes = []
            
            # Note about rounding down
            leftover = trade_size - position_value
            if leftover > 0 and exact_shares >= 1.5:
                notes.append(
                    f"Buying {whole_shares} shares at ${current:,.2f} = ${position_value:,.2f} "
                    f"(${leftover:,.2f} unused from ${trade_size:,.2f} trade size)."
                )
            
            if risk_per_trade >= 1.0:
                notes.append(
                    f"No stop loss — 100% risk tolerance means you accept total loss of ${position_value:,.2f}."
                )
            elif risk_per_trade >= 0.5:
                notes.append(
                    f"Wide stop loss at {risk_per_trade*100:.1f}% — "
                    f"you'd lose ${risk_amount:,.2f} if stop is hit."
                )
            
            if notes:
                res['position_size_note'] = ' '.join(notes)
                
        else:
            # Can't afford even 1 share
            res['suggested_shares'] = None
            res['stop_loss_price'] = None
            res['stop_loss_price_long'] = None
            res['stop_loss_price_short'] = None
            res['position_risk_amount'] = None
            
            res['position_size_note'] = (
                f"Cannot afford 1 share at ${current:,.2f} with ${trade_size:,.2f} trade size. "
                f"Minimum trade size: ${current:,.2f}. "
                f"The full analysis is shown below."
            )

    # ═══════════════════════════════════════════════════════════════════════
    # PREDICTABILITY SCORE (5 tests, each worth 1 point)
    # ═══════════════════════════════════════════════════════════════════════

    # Ljung-Box Test (informational only — NOT scored)
    if len(returns_train) > 10:
        lb_test = acorr_ljungbox(returns_train, lags=[10], return_df=True)
        res['lb_pvalue'] = float(lb_test.iloc[0, 1])

    # ADF Test (informational only — NOT scored)
    if len(df['Close'].dropna()) > 20:
        try:
            adf_result = adfuller(df['Close'].dropna())
            res['adf_pvalue'] = float(adf_result[1])
        except Exception: 
            pass

    # HURST EXPONENT (DFA with shuffled baseline)
    if len(returns_train) > 100:
        try:
            H, H_shuf_mean, H_shuf_std, is_sig, _, _, _ = hurst_with_baseline(
                returns_train, n_shuffles=n_shuffles
            )
            if not np.isnan(H):
                res['hurst'] = float(H)
                res['hurst_significant'] = bool(is_sig)
                if not np.isnan(H_shuf_mean):
                    res['hurst_shuffled_mean'] = float(H_shuf_mean)
                
                if is_sig and (H > 0.55 or H < 0.45):
                    res['predictability_score'] += 1
        except Exception:
            pass
    
    # Hurst out-of-sample
    oos_shuffles = max(10, n_shuffles // 3)
    if len(returns_test) > 100:
        try:
            H_oos, _, _, _, _, _, _ = hurst_with_baseline(returns_test, n_shuffles=oos_shuffles)
            if not np.isnan(H_oos):
                res['hurst_oos'] = float(H_oos)
        except Exception:
            pass

    # MOMENTUM CORRELATION
    if len(returns_train) > 30:
        m_corr, n_pairs = multi_day_momentum_corr(returns_train, block_days=3)
        if m_corr is not None:
            res['momentum_corr'] = float(m_corr)
            if abs(m_corr) > 0.08:
                res['predictability_score'] += 1

        mean_rev_up, mean_rev_down = non_overlapping_mean_reversion(returns_train, window_days)
        res['mean_rev_up'] = mean_rev_up
        res['mean_rev_down'] = mean_rev_down

        if res['mean_rev_up'] is not None and res['mean_rev_down'] is not None:
            if abs(res['mean_rev_up']) > 0.003 and abs(res['mean_rev_down']) > 0.003:
                res['predictability_score'] += 1

    # OUT-OF-SAMPLE TESTING
    if len(returns_test) > 30:
        m_corr_oos, _ = multi_day_momentum_corr(returns_test, block_days=3)
        if m_corr_oos is not None:
            res['momentum_corr_oos'] = float(m_corr_oos)

        mean_rev_up_oos, mean_rev_down_oos = non_overlapping_mean_reversion(returns_test, window_days)
        res['mean_rev_up_oos'] = mean_rev_up_oos
        res['mean_rev_down_oos'] = mean_rev_down_oos

    # REGIME STABILITY CHECK
    MOMENTUM_MIN_THRESHOLD = 0.05
    
    if res.get('momentum_corr') is not None and res.get('momentum_corr_oos') is not None:
        corr_in = res['momentum_corr']
        corr_oos = res['momentum_corr_oos']
        
        if abs(corr_in) > MOMENTUM_MIN_THRESHOLD:
            same_sign = (corr_in > 0 and corr_oos > 0) or (corr_in < 0 and corr_oos < 0)
            
            if same_sign and abs(corr_oos) >= MOMENTUM_MIN_THRESHOLD:
                res['regime_stability'] = 1.0
            elif same_sign:
                res['regime_stability'] = 0.5
            else:
                res['regime_stability'] = 0.0
        else:
            res['regime_stability'] = 0.0
    
    if (res.get('hurst') is not None and res.get('hurst_oos') is not None 
        and res.get('hurst_significant')):
        hurst_in = res['hurst']
        hurst_oos = res['hurst_oos']
        
        in_trending = hurst_in > 0.55
        in_reverting = hurst_in < 0.45
        oos_trending = hurst_oos > 0.55
        oos_reverting = hurst_oos < 0.45
        
        hurst_agrees = (in_trending and oos_trending) or (in_reverting and oos_reverting)
        
        if not hurst_agrees and res.get('regime_stability', 0) > 0.5:
            res['regime_stability'] = 0.5

    # PREDICTABILITY TEST 4: Regime Stability OOS
    if res.get('regime_stability') is not None and res['regime_stability'] >= 0.5:
        res['predictability_score'] += 1

    # PREDICTABILITY TEST 5: Volume-Price Confirmation
    vp_data = volume_price_confirmation(df, lookback=60)
    if vp_data is not None:
        res['volume_price_data'] = vp_data
        res['vp_ratio'] = vp_data['vp_ratio']
        res['vp_confirming'] = vp_data['vp_confirming']
        
        if vp_data['vp_confirming']:
            res['predictability_score'] += 1

    # ═══════════════════════════════════════════════════════════════════════
    # LIQUIDITY ANALYSIS — FIX #3: Liquidity is now ADVISORY, not a gate.
    #
    # Old bug: position_size_vs_volume >= 2% set liquidity_failed=True,
    # which forced generate_trading_signal() to return DO_NOT_TRADE.
    # This was wrong because:
    #   - The user can adjust account size/risk to fix liquidity
    #   - A valid pattern shouldn't be hidden because of position sizing
    #   - The 2% threshold was arbitrary for the "too large" case
    #
    # New logic:
    # - Liquidity warnings are always computed and displayed
    # - They NEVER override the trading signal
    # - The signal reflects pattern quality only
    # - Edge vs friction still matters (keeps the 3x friction gate)
    # ═══════════════════════════════════════════════════════════════════════

    # Calculate 30-day average volume
    avg_vol_30 = df['Volume'].tail(30).mean()
    res['avg_daily_volume'] = float(avg_vol_30)
    
    # Amihud Illiquidity Ratio
    amihud = calculate_amihud_illiquidity(df)
    res['amihud_illiquidity'] = amihud
    
    # Position size as % of daily volume
    if res.get('calculated_shares') and avg_vol_30 > 0:
        position_size_vs_vol = res['calculated_shares'] / avg_vol_30
        res['position_size_vs_volume'] = float(position_size_vs_vol)
    elif res['suggested_shares'] and avg_vol_30 > 0:
        position_size_vs_vol = res['suggested_shares'] / avg_vol_30
        res['position_size_vs_volume'] = float(position_size_vs_vol)
    
    # Dynamic Slippage Estimate
    dynamic_slippage = calculate_dynamic_slippage(df)
    res['estimated_slippage_pct'] = dynamic_slippage * 100
    
    # Total Friction (Slippage + Transaction Cost) round trip
    total_friction = (dynamic_slippage + res['transaction_cost']) * 2
    res['total_friction_pct'] = float(total_friction * 100)
    
    # Calculate Expected Edge
    if res['momentum_corr'] is not None and res['volatility'] is not None:
        expected_edge = abs(res['momentum_corr']) * res['volatility']
        res['expected_edge_pct'] = float(expected_edge)
    
    # Liquidity assessment — ADVISORY ONLY, does not block signals
    if (res.get('expected_edge_pct') is not None and
        res.get('total_friction_pct') is not None):
        
        edge_too_small = res['expected_edge_pct'] <= (res['total_friction_pct'] * 3)
        
        if edge_too_small:
            # Edge vs friction is still a hard gate — this is about pattern quality,
            # not position sizing. If your edge can't cover costs, don't trade.
            res['is_liquid_enough'] = False
            res['liquidity_failed'] = False  # Pattern issue, not liquidity
        else:
            res['is_liquid_enough'] = True
            res['liquidity_failed'] = False
    else:
        res['is_liquid_enough'] = True
        res['liquidity_failed'] = False
    
    # Liquidity Quality Score (informational)
    res['liquidity_score'] = get_liquidity_score(amihud, res['position_size_vs_volume'])
    
    # Liquidity Warning (advisory — shown to user but doesn't block signal)
    if res['position_size_vs_volume'] and res['position_size_vs_volume'] > 0.02:
        res['liquidity_warning'] = get_liquidity_warning(
            res['liquidity_score'],
            res['position_size_vs_volume'] or 0,
            amihud
        )
    elif res['liquidity_score'] == 'LOW':
        res['liquidity_warning'] = get_liquidity_warning(
            res['liquidity_score'],
            res['position_size_vs_volume'] or 0,
            amihud
        )

    # GENERATE FINAL TRADING SIGNAL
    res['final_signal'] = generate_trading_signal(res)
    
    # Update stop loss based on final signal (LONG vs SHORT)
    # Skip if risk_per_trade >= 1.0 (no stop loss)
    if (res['final_signal'] and res.get('stop_loss_price_long') is not None 
        and res.get('stop_loss_price_short') is not None):
        short_signals = [
            'SHORT_DOWNTREND', 'SHORT_BOUNCES_ONLY', 'SHORT_MOMENTUM',
            'SPEC_SHORT_DOWNTREND', 'SPEC_SHORT_BOUNCES_ONLY', 'SPEC_SHORT_MOMENTUM',
            'WAIT_OR_SHORT_BOUNCE', 'SPEC_WAIT_OR_SHORT_BOUNCE'
        ]
        
        if res['final_signal'] in short_signals:
            res['stop_loss_price'] = res['stop_loss_price_short']
        else:
            res['stop_loss_price'] = res['stop_loss_price_long']
    
    # SPECULATIVE TIER: Halve position size for 2/5 predictability signals
    if res['final_signal'] and res['final_signal'].startswith('SPEC_'):
        if res.get('suggested_shares') is not None and res['suggested_shares'] > 1:
            full_shares = res['suggested_shares']
            half_shares = max(1, full_shares // 2)
            res['suggested_shares'] = half_shares
            res['speculative_full_shares'] = full_shares
            
            # Recalculate risk amount for halved position
            half_position_value = half_shares * (res.get('current') or 0)
            res['position_risk_amount'] = float(half_position_value * risk_per_trade)
            
            res['position_size_note'] = (
                f"⚠ SPECULATIVE: Position halved from {full_shares} to {half_shares} shares "
                f"(${half_position_value:,.2f}). Only {res.get('predictability_score', 0)}/5 statistical tests passed — "
                f"reduced position size limits downside from weaker conviction."
            )
    
    # TRADE QUALITY SCORE
    if res['final_signal'] and res['final_signal'] not in ('DO_NOT_TRADE', 'NO_CLEAR_SIGNAL'):
        quality = calculate_trade_quality(res)
        res['trade_quality'] = quality['trade_quality']
        res['quality_components'] = quality['quality_components']
        res['quality_label'] = quality['quality_label']
    
    # Zero out edge if pattern failed statistical validation
    if res['final_signal'] in ['DO_NOT_TRADE', 'NO_CLEAR_SIGNAL']:
        res['expected_edge_pct'] = 0.0
        
        validation_failures = []
        
        if res.get('predictability_score', 0) < 2:
            validation_failures.append(f"Predictability score {res['predictability_score']}/5 (need ≥2)")
        
        if res.get('regime_stability') is not None and res.get('regime_stability') < 0.5:
            if res.get('regime_stability') == 0.0:
                validation_failures.append("Regime stability 0% — momentum direction REVERSED out-of-sample")
            else:
                validation_failures.append(f"Regime stability {res['regime_stability']*100:.0f}% (need ≥50%)")
        
        if res.get('momentum_corr') is not None and abs(res['momentum_corr']) <= 0.08:
            validation_failures.append(f"Weak momentum (|r|={abs(res['momentum_corr']):.3f}, need >0.08)")
        
        if res.get('hurst_significant') is False:
            validation_failures.append("Hurst exponent not distinguishable from random (failed baseline test)")
        
        if res.get('vp_confirming') is False or res.get('vp_confirming') is None:
            vp_ratio = res.get('vp_ratio')
            if vp_ratio is not None:
                validation_failures.append(f"Volume doesn't confirm trend (up/down vol ratio: {vp_ratio:.2f})")
            else:
                validation_failures.append("Volume-price confirmation unavailable")
        
        if validation_failures:
            failure_msg = "; ".join(validation_failures)
            pattern_warning = f"Pattern failed statistical validation: {failure_msg}. No exploitable edge exists."
        else:
            pattern_warning = "Pattern failed statistical validation — no exploitable edge exists"

        existing_warning = res.get('liquidity_warning')
        if existing_warning:
            res['liquidity_warning'] = f"{existing_warning} | {pattern_warning}"
        else:
            res['liquidity_warning'] = pattern_warning
    
    return res


def generate_trading_signal(res):
    """
    Generate final trading signal based on all metrics.
    
    PREDICTABILITY SCORE: 0-5:
      1. Momentum correlation  2. Hurst/DFA  3. Mean reversion
      4. Regime stability OOS  5. Volume-price confirmation
    
    TWO TIERS:
    - HIGH CONVICTION (predictability ≥3/5): Full position
    - SPECULATIVE (predictability 2/5): Reduced position
    
    Hard rejections:
    - Regime stability 0% (sign flip) → DO_NOT_TRADE
    - Edge < 3x friction → DO_NOT_TRADE
    - Predictability < 2 → DO_NOT_TRADE
    
    NOTE: Liquidity (position size vs volume) is ADVISORY only.
    It does NOT block the signal. The user can adjust account/risk.
    """
    
    # HARD GATE: Minimum predictability
    pred_score = res.get('predictability_score', 0)
    if pred_score < 2:
        return 'DO_NOT_TRADE'
    
    # HARD GATE: Regime Stability (no sign flips)
    if res.get('regime_stability') is not None and res.get('regime_stability') < 0.5:
        return 'DO_NOT_TRADE'
    
    # HARD GATE: Edge vs Friction (pattern quality, not position sizing)
    if not res.get('is_liquid_enough', False):
        return 'DO_NOT_TRADE'
    
    # Determine tier
    is_speculative = pred_score < 3
    
    # Momentum check
    momentum = res.get('momentum_corr')
    if momentum is None or abs(momentum) <= 0.08:
        return 'NO_CLEAR_SIGNAL'
    
    # Trend direction
    trend = res.get('trend_direction')
    if trend not in ['UP', 'DOWN']:
        if abs(momentum) > 0.15:
            return 'WAIT_FOR_TREND'
        else:
            return 'DO_NOT_TRADE'
    
    hurst = res.get('hurst')
    z_ema = res.get('z_ema')
    hurst_significant = res.get('hurst_significant', False)
    
    # ─── GENERATE SIGNAL ────────────────────────────────────────────────
    signal = None
    
    if trend == 'UP':
        if momentum > 0.08:
            if hurst_significant and hurst is not None and hurst > 0.55:
                if z_ema is not None:
                    if z_ema > 1.0:
                        signal = 'WAIT_PULLBACK'
                    elif z_ema > -0.5:
                        signal = 'BUY_UPTREND'
                    else:
                        signal = 'BUY_PULLBACK'
                else:
                    signal = 'BUY_UPTREND'
            else:
                if z_ema is not None and z_ema > 1.0:
                    signal = 'WAIT_PULLBACK'
                else:
                    signal = 'BUY_MOMENTUM'
        else:
            signal = 'WAIT_OR_SHORT_BOUNCE'
    
    elif trend == 'DOWN':
        if momentum > 0.08:
            if hurst_significant and hurst is not None and hurst > 0.55:
                if z_ema is not None:
                    if z_ema < -1.0:
                        signal = 'WAIT_SHORT_BOUNCE'
                    elif z_ema < 0.5:
                        signal = 'SHORT_DOWNTREND'
                    else:
                        signal = 'SHORT_BOUNCES_ONLY'
                else:
                    signal = 'SHORT_DOWNTREND'
            else:
                if z_ema is not None and z_ema < -1.0:
                    signal = 'WAIT_SHORT_BOUNCE'
                else:
                    signal = 'SHORT_MOMENTUM'
        else:
            signal = 'WAIT_FOR_REVERSAL'
    
    else:
        if abs(momentum) > 0.15:
            signal = 'WAIT_FOR_TREND'
        else:
            signal = 'DO_NOT_TRADE'
    
    # ─── APPLY SPECULATIVE PREFIX ───────────────────────────────────────
    if is_speculative and signal is not None:
        actionable_signals = [
            'BUY_UPTREND', 'BUY_PULLBACK', 'BUY_MOMENTUM',
            'SHORT_DOWNTREND', 'SHORT_BOUNCES_ONLY', 'SHORT_MOMENTUM',
            'WAIT_OR_SHORT_BOUNCE', 'WAIT_FOR_REVERSAL'
        ]
        if signal in actionable_signals:
            signal = 'SPEC_' + signal
    
    return signal or 'DO_NOT_TRADE'
