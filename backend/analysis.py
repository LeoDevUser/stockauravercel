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


def dfa_hurst(series, min_box=10, num_scales=20):
    """
    Detrended Fluctuation Analysis (DFA) to estimate Hurst exponent.
    """
    N = len(series)
    max_box = N // 4
    
    if max_box <= min_box or N < min_box * 4:
        return np.nan, None, None, None
    
    #Integrate the mean-centered series (cumulative sum of deviations)
    y = np.cumsum(series - np.mean(series))
    
    #Generate logarithmically spaced box sizes
    scales = np.unique(
        np.logspace(np.log10(min_box), np.log10(max_box), num=num_scales).astype(int)
    )
    
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
        
        # Backward pass (to avoid wasting data and have more values for the fluctuation est)
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
    """
    # Real Hurst
    H, scales, fluctuations, poly = dfa_hurst(series, **kwargs)
    
    if np.isnan(H):
        return H, np.nan, np.nan, False, None, None, None
    
    # Shuffled baseline
    shuffled_hursts = []
    rng = np.random.default_rng(42)
    
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
    is_significant = False
    if H_shuf_std > 0:
        z_score = abs(H - H_shuf_mean) / H_shuf_std
        is_significant = z_score > 1.5
    
    return H, H_shuf_mean, H_shuf_std, is_significant, scales, fluctuations, poly


def multi_day_momentum_corr(daily_returns, block_days=3):
    """
    Calculate momentum correlation using non-overlapping multi-day blocks.
    
    Measures: "Does the direction of this 3-day period predict the next 3-day period?"
    
    Returns: correlation or None
    """
    n = len(daily_returns)
    n_blocks = n // block_days
    
    if n_blocks < 20:
        return None
    
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
        return None
    
    corr = np.corrcoef(x, y)[0, 1]
    
    if np.isnan(corr):
        return None
    
    return float(corr)


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
    Volume-Price Confirmation Test
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
    
    #1. MULTI-TIMEFRAME TREND ALIGNMENT (0-2)
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
        components['trend_alignment'] = 0.0
    
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
        components['entry_timing'] = 0.0
    
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
        components['sharpe_quality'] = 0.0
    
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
        components['volatility_fit'] = 0.0
    
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
        components['volume_confirmation'] = 0.0
    
    total += components['volume_confirmation']
    
    # ── FINAL SCORE ──────────────────────────────────────────────────────
    total = round(total, 1)
    
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
        'volatility_category': None,
        'final_signal': None,
        # Liquidity & Friction Analysis
        'avg_daily_volume': None,
        'position_size_vs_volume': None,
        'total_friction_pct': None,
        'expected_edge_pct': None,
        'title': title,
        'current': current,
        'cap': cap,
        'currency': currency,
        'OHLC': OHLC.to_dict('records'),
        'data_points': len(df),
        'transaction_cost': 0.001,  # 0.1% per trade
        'risk_per_trade': risk_per_trade,
        'account_size_input': account_size,
    }

    # Split data: 70% train, 30% test for out-of-sample validation
    split_idx = int(len(df) * 0.7)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    returns_train = df_train['Return'].dropna().values
    returns_test = df_test['Return'].dropna().values

    #Compute recent returns
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
    
    # Determine trend direction over last 6 months
    if res['recent_return_6m'] is not None:
        if res['recent_return_6m'] > 0.04:
            res['trend_direction'] = 'UP'
        elif res['recent_return_6m'] < -0.04:
            res['trend_direction'] = 'DOWN'
        else:
            res['trend_direction'] = 'NEUTRAL'

    # Z-SCORE
    if len(df) >= 20:
        roll_m = df['Close'].rolling(window=20).mean()
        roll_s = df['Close'].rolling(window=20).std()
        curr_std = roll_s.iloc[-1]
        if curr_std > 0:
            z = (df['Close'].iloc[-1] - roll_m.iloc[-1]) / curr_std
            res['zscore'] = float(z)

    # Z-EMA
    if len(df) >= 20:
        ema_m = df['Close'].ewm(span=20, adjust=False, min_periods=20).mean()
        ema_s = df['Close'].ewm(span=20, adjust=False, min_periods=20).std()
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

    # Dertermine volatility category
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

    if current and current > 0:

        # How many whole shares can we buy with our trade size?
        exact_shares = account_size / current
        whole_shares = int(exact_shares)
        
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
            leftover = account_size - position_value
            if leftover > 0 and exact_shares >= 1.5:
                notes.append(
                    f"Buying {whole_shares} shares at ${current:,.2f} = ${position_value:,.2f} "
                    f"(${leftover:,.2f} unused from ${account_size:,.2f} trade size)."
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
                f"Cannot afford 1 share at ${current:,.2f} with ${account_size:,.2f} trade size. "
                f"Minimum trade size: ${current:,.2f}. "
                f"The full analysis is shown below."
            )

    # Ljung-Box Test (informational only NOT scored)
    if len(returns_train) > 10:
        lb_test = acorr_ljungbox(returns_train, lags=[10], return_df=True)
        res['lb_pvalue'] = float(lb_test.iloc[0, 1])

    # ADF Test (informational only NOT scored)
    if len(df['Close'].dropna()) > 20:
        try:
            adf_result = adfuller(df['Close'].dropna())
            res['adf_pvalue'] = float(adf_result[1])
        except Exception: 
            pass

    #Predictability Score (5 tests, each worth 1 point)

    # test 1 HURST EXPONENT (DFA with shuffled baseline)
    if len(returns_train) > 100:
        try:
            H, H_shuf_mean, _, is_sig, _, _, _ = hurst_with_baseline(
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

    # test 2 MOMENTUM CORRELATION
    if len(returns_train) > 30:
        m_corr = multi_day_momentum_corr(returns_train, block_days=3)
        if m_corr is not None:
            res['momentum_corr'] = float(m_corr)
            if abs(m_corr) > 0.08:
                res['predictability_score'] += 1

        mean_rev_up, mean_rev_down = non_overlapping_mean_reversion(returns_train, window_days)
        res['mean_rev_up'] = mean_rev_up
        res['mean_rev_down'] = mean_rev_down

        # test 3 MEAN REVERSION ATFTER EXTREMES
        if res['mean_rev_up'] is not None and res['mean_rev_down'] is not None:
            if abs(res['mean_rev_up']) > 0.01 and abs(res['mean_rev_down']) > 0.01:
                res['predictability_score'] += 1

    # momentum out-of-sample
    if len(returns_test) > 30:
        m_corr_oos = multi_day_momentum_corr(returns_test, block_days=3)
        if m_corr_oos is not None:
            res['momentum_corr_oos'] = float(m_corr_oos)

        mean_rev_up_oos, mean_rev_down_oos = non_overlapping_mean_reversion(returns_test, window_days)
        res['mean_rev_up_oos'] = mean_rev_up_oos
        res['mean_rev_down_oos'] = mean_rev_down_oos

    # test 4 REGIME STABILITY CHECK
    MOMENTUM_MIN_THRESHOLD = 0.08
    
    res['regime_stability'] = 0.0
    if res.get('momentum_corr') is not None and res.get('momentum_corr_oos') is not None:
        corr_in = res['momentum_corr']
        corr_oos = res['momentum_corr_oos']
        
        if abs(corr_in) > MOMENTUM_MIN_THRESHOLD:
            same_sign = (corr_in > 0 and corr_oos > 0) or (corr_in < 0 and corr_oos < 0)
            
            if same_sign and abs(corr_oos) >= MOMENTUM_MIN_THRESHOLD:
                res['regime_stability'] = 1.0
            elif same_sign:
                res['regime_stability'] = 0.5
    
    if res['regime_stability'] >= 0.5:
        res['predictability_score'] += 1

    # test 5: Volume-Price Confirmation
    vp_data = volume_price_confirmation(df, lookback=60)
    if vp_data is not None:
        res['volume_price_data'] = vp_data
        res['vp_ratio'] = vp_data['vp_ratio']
        res['vp_confirming'] = vp_data['vp_confirming']
        
        if vp_data['vp_confirming']:
            res['predictability_score'] += 1

    # Calculate 30-day average volume
    avg_vol_30 = df['Volume'].tail(30).mean()
    res['avg_daily_volume'] = float(avg_vol_30)
    
    SLIPPAGE_ESTIMATE = 0.0005  # 0.05% standard slippage estimate
    total_friction = (SLIPPAGE_ESTIMATE + res['transaction_cost']) * 2
    res['total_friction_pct'] = float(total_friction * 100)

    # Calculate Expected Edge
    if res['momentum_corr'] is not None and res['volatility'] is not None:
        expected_edge = abs(res['momentum_corr']) * res['volatility']
        res['expected_edge_pct'] = float(expected_edge)
    
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
    - Regime stability < 0.5 → DO_NOT_TRADE
    - Edge < 3x friction → DO_NOT_TRADE
    - Predictability < 2 → DO_NOT_TRADE
    """
    
    # HARD GATE: Minimum predictability
    pred_score = res.get('predictability_score', 0)
    if pred_score < 2:
        return 'DO_NOT_TRADE'
    
    # HARD GATE: Regime Stability (no sign flips)
    #TODO this might be to strict, will test
    if res.get('regime_stability',0.0) < 0.5:
        return 'DO_NOT_TRADE'
    
    # Directly check edge vs friction ratio
    edge_ratio = res.get('expected_edge_pct') / res['total_friction_pct']
    if edge_ratio <= 3:  # Need 3x edge to cover costs
        return 'DO_NOT_TRADE'
    
    # Determine tier
    is_speculative = pred_score < 3
    
    # Momentum check
    momentum = res.get('momentum_corr',0.0) 
    if abs(momentum) <= 0.08:
        return 'NO_CLEAR_SIGNAL'
    
    # Trend direction
    trend = res.get('trend_direction')
    if trend not in ['UP', 'DOWN']:
        return 'WAIT_FOR_TREND'
    
    hurst = res.get('hurst')
    z_ema = res.get('z_ema')
    hurst_significant = res.get('hurst_significant', False)
    
    #Generate Signal
    signal = None
    
    if trend == 'UP':
        if momentum > 0:
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
        if momentum < 0:
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
    
    #Apply Speculative Prefix
    if is_speculative and signal is not None:
        actionable_signals = [
            'BUY_UPTREND', 'BUY_PULLBACK', 'BUY_MOMENTUM',
            'SHORT_DOWNTREND', 'SHORT_BOUNCES_ONLY', 'SHORT_MOMENTUM',
            'WAIT_OR_SHORT_BOUNCE', 'WAIT_FOR_REVERSAL'
        ]
        if signal in actionable_signals:
            signal = 'SPEC_' + signal
    
    return signal or 'DO_NOT_TRADE'

if __name__ == "__main__":
    print(analyze_stock('PLTR'))
