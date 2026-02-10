import { useSearchParams, useNavigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import Chart from '../components/Chart'
import { TradingVerdict } from '../components/TradingVerdict'
import '../styles/ResultsPage.css'
import SearchBar from '../components/SearchBar'
import logo from '../assets/logo-dark.png'
import home from '../assets/home.png'
import Tooltip from '../components/Tooltip'
import { tooltipContent } from '../utils/TooltipContent'
import { apiUrl } from '../utils/api'

export interface AnalysisResult {
  ticker: string
  window_days: number
  period: string
  hurst: number | null
  hurst_oos: number | null
  hurst_significant: boolean | null
  hurst_shuffled_mean: number | null
  momentum_corr: number | null
  momentum_corr_oos: number | null
  lb_pvalue: number | null
  adf_pvalue: number | null
  mean_rev_up: number | null
  mean_rev_down: number | null
  mean_rev_up_oos: number | null
  mean_rev_down_oos: number | null
  sharpe: number | null
  volatility: number | null
  Return: number | null
  predictability_score: number
  zscore: number | null
  z_ema: number | null
  volatility_category: string | null
  final_signal: string | null
  regime_stability: number | null
  data_points: number
  transaction_cost: number
  slippage: number
  trend_direction: 'UP' | 'DOWN' | 'NEUTRAL' | null
  recent_return_1y: number | null
  recent_return_6m: number | null
  recent_return_3m: number | null
  recent_return_1m: number | null
  title: string
  current: number
  cap: string | null
  currency: string
  suggested_shares: number | null
  stop_loss_price: number | null
  position_risk_amount: number | null
  position_size_note: string | null
  risk_per_trade: number
  min_account_needed: number | null
  avg_daily_volume: number | null
  amihud_illiquidity: number | null
  liquidity_score: string | null
  position_size_vs_volume: number | null
  estimated_slippage_pct: number | null
  total_friction_pct: number | null
  expected_edge_pct: number | null
  is_liquid_enough: boolean | null
  liquidity_failed: boolean
  liquidity_warning: string | null
  speculative_full_shares: number | null
  // Volume-Price Confirmation
  vp_ratio: number | null
  vp_confirming: boolean | null
  volume_price_data: {
    vp_ratio: number
    vp_confirming: boolean
    avg_vol_up: number
    avg_vol_down: number
    trend_for_vp: string
  } | null
  // Trade Quality Score
  trade_quality: number | null
  quality_components: {
    trend_alignment: number
    entry_timing: number
    sharpe_quality: number
    volatility_fit: number
    volume_confirmation: number
  } | null
  quality_label: string | null
  OHLC: Array<{
    Date: string
    Open: number
    Close: number
    Low: number
    High: number
  }>
  error?: string
}

export default function ResultsPage() {
  const [searchParams] = useSearchParams()
  const ticker = searchParams.get('ticker')
  const [results, setResults] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [transactionCost, setTransactionCost] = useState<number>(0.001) // Default 0.1%
  const [tradeSize, setTradeSize] = useState<number>(10000) // Default $10,000
  const [riskTolerance, setRiskTolerance] = useState<number>(2) // Default 2%
  const navigate = useNavigate()

  const handleNavigate = (ticker: string) => {
    navigate(`/results?ticker=${ticker}`)
  }

  useEffect(() => {
    const fetchResults = async () => {
      if (!ticker) return
      try {
        setLoading(true)
        const response = await fetch(
          apiUrl(`/api/analyze?ticker=${ticker}&period=5y&window_days=5`)
        )
        const data: AnalysisResult = await response.json()
        setResults(data)
      } catch (err) {
        console.error('Error fetching results:', err)
      } finally {
        setLoading(false)
      }
    }
    fetchResults()
  }, [ticker])

  // Determine market regime description
  const getRegimeDescription = (): string => {
    if (!results?.hurst) return 'Unknown'
    if (results.hurst > 0.55) return 'Trending'
    if (results.hurst < 0.45) return 'Mean-Reverting'
    return 'Random Walk'
  }

  if (loading) return <div className="loading-page"><p>Analyzing Stock...</p></div>

  if (results?.error) return (
    <div className="delisted-container" onClick={() => navigate('/')}>
      <p className='delisted-error'>{results.error}</p>
      <img src={home} alt='Landing' />
      <p className='delisted-go-back'>Go Back to Landing Page</p>
    </div>
  )

  if (!results) return <div className="no-results"><p>No results</p></div>

  return (
    <div className="results-page">
      {/* Navigation Header */}
      <div className='nav'>
        <div className='logo-dark'>
          <img src={logo} alt='Dark Logo' onClick={() => navigate('/')} />
        </div>
        <div className="ticker-info">
          <h1>{results.title || 'Undefined'}</h1>
          <h2>Ticker: {ticker}&emsp;&emsp;{results.current} {results.currency}</h2>
        </div>
        <div className='home-search'>
          <div className='search-section-landing'>
            <SearchBar
              onSelect={handleNavigate}
              placeholder='Enter a stock ticker to analyze it..'
            />
          </div>
          <div className='home'>
            <img src={home} alt='Landing' onClick={() => navigate('/')} />
          </div>
        </div>
      </div>

      {/* Trading Parameters Input */}
      <div className="trading-parameters-input">
        <div className="parameter-group">
          <label>
            <span>Trade Size:</span>
            <input
              type="number"
              value={tradeSize}
              onChange={(e) => setTradeSize(parseFloat(e.target.value) || 10000)}
              step={1000}
              min={1}
            />
            <span>$</span>
          </label>
          <small>
            Capital allocated to this trade
          </small>
        </div>

        <div className="parameter-group">
          <label>
            <span>Risk Tolerance:</span>
            <input
              type="number"
              value={riskTolerance}
              onChange={(e) => setRiskTolerance(parseFloat(e.target.value) || 2)}
              step={0.5}
              min={0.1}
              max={100}
            />
            <span>%</span>
          </label>
          <small>
            Stop loss trigger (% drawdown from entry, 100% = no stop loss)
          </small>
        </div>
        
        <div className="parameter-group">
          <label>
            <span>Transaction Cost:</span>
            <input
              type="number"
              value={(transactionCost * 100).toFixed(3)}
              onChange={(e) => setTransactionCost(parseFloat(e.target.value) / 100)}
              step={0.001}
              min={0}
            />
            <span>%</span>
          </label>
          <small>Per trade commission (IB $0.005, Robinhood $0, Traditional 0.1-1%)</small>
        </div>
      </div>

      {/* Main Results Grid - Row Layout */}
      <div className="results-grid">
        
        {/* First Row: Key Metrics and Statistical Tests side by side */}
        <div className="results-grid-row">
          {/* LEFT: Key Metrics */}
          <div className="section header-stats">
            <h3>Key Metrics</h3>
            
            {results.trend_direction && (
              <div className="stat trend-stat">
                <label>
			  Current Trend (1-Year)
			  <Tooltip content="The stock's overall price direction over the past year. UP = gaining value, DOWN = losing value, NEUTRAL = flat. Based on comparing current price to 252 trading days ago."></Tooltip>
			  </label>
                <div className={`trend-value trend-${results.trend_direction.toLowerCase()}`}>
                  {results.trend_direction === 'UP' && '📈 UP'}
                  {results.trend_direction === 'DOWN' && '📉 DOWN'}
                  {results.trend_direction === 'NEUTRAL' && '➡️ NEUTRAL'}
                </div>
                {results.recent_return_1y && (
                  <small>{(results.recent_return_1y * 100).toFixed(2)}% (1-year return)</small>
                )}
              </div>
            )}

            <div className="stat">
              <label>Predictability Score
			<Tooltip content={tooltipContent.predictabilityScore} />
			</label>
              <div className="score">{results.predictability_score}/5</div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${(results.predictability_score / 5) * 100}%` }}
                />
              </div>
            </div>

            {results.regime_stability !== null && (
              <div className="stat">
                <label>Regime Stability (OOS)
			  <Tooltip content={tooltipContent.regimeStability} />
			  </label>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${results.regime_stability * 100}%`,
                      backgroundColor: results.regime_stability >= 1.0 ? '#22c55e' : results.regime_stability >= 0.5 ? '#f59e0b' : '#ef4444'
                    }}
                  />
                </div>
                <small>{(results.regime_stability * 100).toFixed(0)}%</small>
              </div>
            )}

            {results.sharpe !== null && (
              <div className="stat">
                <label>Sharpe Ratio
			  <Tooltip content={tooltipContent.sharpeRatio}/>
			  </label>
                <div className="value">{results.sharpe.toFixed(2)}</div>
              </div>
            )}
            
            {results.volatility !== null && (
              <div className="stat">
                <label>Volatility (Annual)
			  <Tooltip content={tooltipContent.volatility}/>
			  </label>
                <div className="value">{results.volatility.toFixed(2)}%</div>
              </div>
            )}
            
            {results.Return !== null && (
              <div className="stat">
                <label>Return (Annual)
			  <Tooltip content={tooltipContent.annualReturn} />
			  </label>
                <div className={`value ${results.Return < 0 ? 'negative' : 'positive'}`}>
                  {results.Return.toFixed(2)}%
                </div>
              </div>
            )}
          </div>

          {/* RIGHT: Statistical Tests */}
          <div className="section statistical-engine">
            <h3>Statistical Tests</h3>

            <div style={{ 
              marginBottom: '1.5em', 
              padding: '0.75em', 
              background: 'rgba(255, 149, 0, 0.08)', 
              borderRadius: '6px',
              border: '1px solid rgba(255, 149, 0, 0.2)',
              fontSize: '0.85em',
              color: '#ccc'
            }}>
              These 5 tests determine the <strong style={{ color: '#ff9500' }}>Predictability Score</strong>. Each passed test = 1 point.
            </div>

            {/* Test 1: Momentum Correlation */}
            <div className="metric-box">
              <label>1. Momentum Correlation (3-Day Blocks)
                <Tooltip content={tooltipContent.momentumCorrelation} />
              </label>
              {results.momentum_corr !== null ? (
                <>
                  <div className={`status ${Math.abs(results.momentum_corr) > 0.08 ? 'significant' : 'insignificant'}`}>
                    {Math.abs(results.momentum_corr) > 0.08 
                      ? `✓ Passed (${(results.momentum_corr * 100).toFixed(1)}%, need >±8%)` 
                      : `✗ Failed (${(results.momentum_corr * 100).toFixed(1)}%, need >±8%)`}
                  </div>
                  {results.momentum_corr_oos !== null && (
                    <small style={{ display: 'block', marginTop: '0.4em' }}>
                      Out-of-Sample: {(results.momentum_corr_oos * 100).toFixed(1)}%
                      {Math.abs(results.momentum_corr - results.momentum_corr_oos) > 0.1 && (
                        <span style={{ color: '#ef4444' }}> ⚠ Degraded</span>
                      )}
                    </small>
                  )}
                </>
              ) : (
                <div className="status insignificant">✗ No data</div>
              )}
            </div>

            {/* Test 2: Hurst/DFA */}
            <div className="metric-box">
              <label>2. Market Regime (DFA / Hurst)
			<Tooltip content={tooltipContent.hurstExponent} />
			</label>
              {results.hurst !== null ? (
                <div className="gauge-container">
                  <div className={`status ${(results.hurst_significant && (results.hurst > 0.55 || results.hurst < 0.45)) ? 'significant' : 'insignificant'}`}>
                    {(results.hurst_significant && (results.hurst > 0.55 || results.hurst < 0.45))
                      ? `✓ Passed — ${results.hurst > 0.55 ? 'Trending' : 'Mean-Reverting'} (H=${results.hurst.toFixed(3)})`
                      : `✗ Failed — ${!results.hurst_significant ? 'Not distinguishable from random' : 'Near random walk'} (H=${results.hurst.toFixed(3)})`}
                  </div>
                  <div className="gauge" style={{ marginTop: '0.75em' }}>
                    <div className="gauge-fill" style={{
                      background: results.hurst > 0.55 ? '#ef4444' : results.hurst < 0.45 ? '#22c55e' : '#f59e0b',
                      width: `${Math.max(0, Math.min(100, (results.hurst - 0.3) * 200))}%`
                    }} />
                  </div>
                  <div className="gauge-labels">
                    <span>Mean Revert</span>
                    <span className="center">Neutral</span>
                    <span>Trending</span>
                  </div>
                  <div style={{ marginTop: '0.5em', fontSize: '0.85em', color: '#aaa' }}>
                    In-Sample: {results.hurst.toFixed(3)}
                    {results.hurst_oos !== null && <> | Out-Sample: {results.hurst_oos.toFixed(3)}</>}
                  </div>
                  {results.hurst_shuffled_mean !== null && (
                    <small style={{ color: '#777' }}>
                      Shuffled baseline: {results.hurst_shuffled_mean.toFixed(3)}
                    </small>
                  )}
                </div>
              ) : (
                <div className="status insignificant">✗ No data</div>
              )}
            </div>

            {/* Test 3: Mean Reversion */}
            <div className="metric-box">
              <label>3. Mean Reversion After Extremes
                <Tooltip content={tooltipContent.meanReversion} />
              </label>
              {(results.mean_rev_up !== null && results.mean_rev_down !== null) ? (
                <>
                  <div className={`status ${(Math.abs(results.mean_rev_up) > 0.003 && Math.abs(results.mean_rev_down) > 0.003) ? 'significant' : 'insignificant'}`}>
                    {(Math.abs(results.mean_rev_up) > 0.003 && Math.abs(results.mean_rev_down) > 0.003)
                      ? '✓ Passed — Significant conditional reversal detected'
                      : '✗ Failed — Weak or no reversal after extremes'}
                  </div>
                  <div style={{ marginTop: '0.5em', display: 'flex', flexWrap: 'wrap', gap: '1em', fontSize: '0.85em' }}>
                    <span style={{ color: '#aaa' }}>
                      After Up: <strong style={{ color: results.mean_rev_up < 0 ? '#22c55e' : '#f59e0b' }}>
                        {(results.mean_rev_up * 100).toFixed(2)}%
                      </strong>
                    </span>
                    <span style={{ color: '#aaa' }}>
                      After Down: <strong style={{ color: results.mean_rev_down > 0 ? '#22c55e' : '#f59e0b' }}>
                        {(results.mean_rev_down * 100).toFixed(2)}%
                      </strong>
                    </span>
                  </div>
                  {(results.mean_rev_up_oos !== null || results.mean_rev_down_oos !== null) && (
                    <div style={{ marginTop: '0.3em', fontSize: '0.8em', color: '#777' }}>
                      OOS: After Up {results.mean_rev_up_oos !== null ? `${(results.mean_rev_up_oos * 100).toFixed(2)}%` : 'N/A'}
                      {' | '}After Down {results.mean_rev_down_oos !== null ? `${(results.mean_rev_down_oos * 100).toFixed(2)}%` : 'N/A'}
                    </div>
                  )}
                </>
              ) : (
                <div className="status insignificant">✗ No data</div>
              )}
            </div>

            {/* Test 4: Regime Stability OOS */}
            <div className="metric-box">
              <label>4. Regime Stability (Out-of-Sample)
                <Tooltip content={tooltipContent.regimeStability} />
              </label>
              {results.regime_stability !== null ? (
                <>
                  <div className={`status ${results.regime_stability >= 0.5 ? 'significant' : 'insignificant'}`}>
                    {results.regime_stability >= 1.0
                      ? '✓ Passed — Pattern fully confirmed out-of-sample (100%)'
                      : results.regime_stability >= 0.5
                      ? `✓ Passed — Pattern partially holds (${(results.regime_stability * 100).toFixed(0)}%)`
                      : results.regime_stability === 0
                      ? '✗ Failed — Momentum REVERSED out-of-sample (0%)'
                      : `✗ Failed — Pattern unstable (${(results.regime_stability * 100).toFixed(0)}%)`}
                  </div>
                  <div className="progress-bar" style={{ marginTop: '0.5em' }}>
                    <div
                      className="progress-fill"
                      style={{
                        width: `${results.regime_stability * 100}%`,
                        backgroundColor: results.regime_stability >= 1.0 ? '#22c55e' : results.regime_stability >= 0.5 ? '#f59e0b' : '#ef4444'
                      }}
                    />
                  </div>
                </>
              ) : (
                <div className="status insignificant">✗ No data</div>
              )}
            </div>

            {/* Test 5: Volume-Price Confirmation */}
            {results.volume_price_data && (
              <div className="metric-box">
                <label>5. Volume-Price Confirmation
                  <Tooltip content={
                    <>
                      <strong>Volume-Price Confirmation</strong>
                      <p>Tests whether volume supports the current trend direction.</p>
                      <ul>
                        <li><strong>Uptrend:</strong> Up-day volume should exceed down-day volume by &gt;10%</li>
                        <li><strong>Downtrend:</strong> Down-day volume should exceed up-day volume by &gt;10%</li>
                        <li><strong>Neutral:</strong> No clear trend to confirm</li>
                      </ul>
                      <p>The ratio shows average volume on up days divided by average volume on down days. A ratio &gt;1.10 in an uptrend or &lt;0.90 in a downtrend means volume confirms the trend.</p>
                      <p><strong>Counts as 1 point toward predictability score if confirming.</strong></p>
                    </>
                  } />
                </label>
                <div className={`status ${results.volume_price_data.vp_confirming ? 'significant' : 'insignificant'}`}>
                  {results.volume_price_data.vp_confirming ? '✓ Passed — Volume confirms trend' : '✗ Failed — Volume does not confirm trend'}
                </div>
                <div style={{ marginTop: '0.75em', display: 'flex', flexWrap: 'wrap', gap: '1em', fontSize: '0.85em' }}>
                  <span style={{ color: '#aaa' }}>
                    Up/Down Ratio: <strong style={{ color: results.volume_price_data.vp_ratio > 1.1 ? '#22c55e' : results.volume_price_data.vp_ratio < 0.9 ? '#ef4444' : '#f59e0b' }}>
                      {results.volume_price_data.vp_ratio.toFixed(3)}
                    </strong>
                  </span>
                  <span style={{ color: '#aaa' }}>
                    Trend: <strong style={{
                      color: results.volume_price_data.trend_for_vp === 'UP' ? '#22c55e' :
                             results.volume_price_data.trend_for_vp === 'DOWN' ? '#ef4444' : '#f59e0b'
                    }}>
                      {results.volume_price_data.trend_for_vp}
                    </strong>
                  </span>
                </div>
                <div style={{ marginTop: '0.5em', display: 'flex', flexWrap: 'wrap', gap: '1em', fontSize: '0.8em', color: '#777' }}>
                  <span>Avg Vol Up Days: {results.volume_price_data.avg_vol_up.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                  <span>Avg Vol Down Days: {results.volume_price_data.avg_vol_down.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                </div>
              </div>
            )}

          </div>
        </div>

        {/* Second Row: Chart (Full Width) */}
        <div className="section visual-summary">
          <Chart ohlcData={results.OHLC} ticker={ticker || 'Unknown'} />

          {/* Summary Text */}
          <div className="summary-text">
			<p>Mkt cap: &ensp; {results?.cap}</p>
            {results.hurst !== null && (
              <p>
                <strong>{ticker}</strong> is in a <span className={getRegimeDescription().toLowerCase()}>
                  {getRegimeDescription()}
                </span> regime with {results.adf_pvalue && results.adf_pvalue < 0.05 ? 'stationary' : 'non-stationary'} price action.
                {results.trend_direction && (
                  <> Currently in a <strong>{results.trend_direction} trend</strong>.</>
                )}
                {results.hurst_significant === false && (
                  <> <span style={{ color: '#f59e0b' }}>(Note: regime not statistically significant vs random)</span></>
                )}
              </p>
            )}

            {results.Return && results.Return < -20 && (
              <p style={{ fontSize: '0.85em', color: '#ef4444', marginTop: '1em', fontWeight: 'bold' }}>
                ⚠⚠⚠ SEVERE WARNING: This stock has lost {Math.abs(results.Return).toFixed(1)}% annually. Consider bankruptcy risk before trading.
              </p>
            )}
          </div>
        </div>
      </div>

      {/* UNIFIED TRADING VERDICT */}
      <div className="trading-verdict-section">
        <TradingVerdict results={results} transactionCost={transactionCost} tradeSize={tradeSize} riskTolerance={riskTolerance / 100}/>
      </div>

      {/* DETAILED METRICS SECTION (Collapsible) */}
      <details className="detailed-metrics">
        <summary>📊 Show Additional Metrics</summary>
        
        <div className="metrics-grid">
          {results.adf_pvalue !== null && (
            <div className="metric-card">
              <label>ADF Test (Stationarity)
                <Tooltip content={tooltipContent.adfTest} />
              </label>
              <div className="value" style={{ color: results.adf_pvalue < 0.05 ? '#22c55e' : '#ef4444' }}>
                {results.adf_pvalue < 0.05 ? 'Stationary' : 'Non-Stationary'}
              </div>
              <small>p-value: {results.adf_pvalue.toFixed(4)}</small>
            </div>
          )}

          {results.lb_pvalue !== null && (
            <div className="metric-card">
              <label>Ljung-Box Test (Autocorrelation)
                <Tooltip content={tooltipContent.ljungBox} />
              </label>
              <div className="value" style={{ color: results.lb_pvalue < 0.05 ? '#22c55e' : '#ef4444' }}>
                {results.lb_pvalue < 0.05 ? 'Autocorrelated' : 'No Autocorrelation'}
              </div>
              <small>p-value: {results.lb_pvalue.toFixed(4)}</small>
            </div>
          )}

          {results.zscore !== null && (
            <div className="metric-card">
              <label>Current Z-Score
			  <Tooltip content={tooltipContent.zScore} />
			  </label>
              <div className="value">{results.zscore.toFixed(3)}</div>
              <small>
                {Math.abs(results.zscore) > 2 
                  ? '⚠ Extreme' 
                  : Math.abs(results.zscore) > 1 
                  ? 'Moderate' 
                  : 'Normal'}
              </small>
            </div>
          )}

          {results.z_ema !== null && (
            <div className="metric-card">
              <label>Z-EMA (Entry Timing)
			  <Tooltip content={tooltipContent.zEMA} />
			  </label>
              <div className="value">{results.z_ema.toFixed(3)}</div>
              <small>
                {results.z_ema > 1.0 ? 'Overbought' 
                  : results.z_ema < -1.0 ? 'Oversold'
                  : results.z_ema > -0.5 && results.z_ema < 1.0 ? 'Sweet spot'
                  : 'Moderate'}
              </small>
            </div>
          )}

          {results.volatility_category !== null && (
            <div className="metric-card">
              <label>Volatility Category</label>
              <div className="value">{results.volatility_category.replace('_', ' ')}</div>
              <small>{results.volatility?.toFixed(1)}% annualized</small>
            </div>
          )}

          <div className="metric-card">
            <label>Data Points</label>
            <div className="value">{results.data_points.toLocaleString()}</div>
            <small>Trading days analyzed ({results.period})</small>
          </div>
        </div>
      </details>
    </div>
  )
}
