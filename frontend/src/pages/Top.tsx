import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import '../styles/Top.css'
import logo from '../assets/logo-dark.png'
import home from '../assets/home.png'
import { apiUrl } from '../utils/api'

interface TopStock {
  ticker: string
  title: string
  score: number
  final_signal: string
  predictability_score: number
  regime_stability: number | null
  momentum_corr: number | null
  momentum_corr_oos: number | null
  expected_edge_pct: number | null
  total_friction_pct: number | null
  current: number
  currency: string
  trend_direction: 'UP' | 'DOWN' | 'NEUTRAL' | null
  sharpe: number | null
  volatility: number | null
  liquidity_failed: boolean
  suggested_shares: number | null
  z_ema: number | null
  hurst: number | null
  hurst_significant: boolean | null
  stop_loss_price: number | null
  lb_pvalue: number | null
  adf_pvalue: number | null
  vp_ratio: number | null
  vp_confirming: boolean | null
  trade_quality: number | null
  quality_label: string | null
}

interface TopStocksData {
  timestamp: string
  total_analyzed: number
  stocks: TopStock[]
}

export default function TopStocksPage() {
  const [data, setData] = useState<TopStocksData | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<'all' | 'buy' | 'short' | 'wait'>('all')
  const navigate = useNavigate()

  useEffect(() => {
    const loadTopStocks = async () => {
      try {
        setLoading(true)
        const response = await fetch(apiUrl('/top'))
        
        if (!response.ok) {
          throw new Error('Failed to load top stocks data')
        }
        
        const jsonData: TopStocksData = await response.json()
        // Filter out non-actionable signals — only show tradeable opportunities
		setData({
			...jsonData,
			stocks: jsonData.stocks.filter(
				s => s.final_signal !== 'NO_CLEAR_SIGNAL' && s.final_signal !== 'DO_NOT_TRADE'
			)
		})
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }

    loadTopStocks()
  }, [])

  const getSignalCategory = (signal: string): 'buy' | 'short' | 'wait' | 'none' => {
    if (signal.includes('BUY')) return 'buy'
    if (signal.includes('SHORT')) return 'short'
    if (signal.includes('WAIT')) return 'wait'
    return 'none'
  }

  const getSignalColor = (signal: string): string => {
    const category = getSignalCategory(signal)
    if (category === 'buy') return '#22c55e'
    if (category === 'short') return '#ef4444'
    if (category === 'wait') return '#f59e0b'
    return '#888'
  }

  const filteredStocks = data?.stocks.filter(stock => {
    if (filter === 'all') return true
    return getSignalCategory(stock.final_signal) === filter
  }) || []

  if (loading) {
    return (
      <div className="loading-page">
        <p>Loading top stocks...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="error-page">
        <h2>Error Loading Data</h2>
        <p>{error}</p>
        <p style={{ marginTop: '2em', fontSize: '0.9em', color: '#999' }}>
			Top Stocks Not Generated, Contact Administrator
        </p>
        <button onClick={() => navigate('/')} style={{ marginTop: '2em' }}>
          Go to Home
        </button>
      </div>
    )
  }

  if (!data) return null

  return (
    <div className="top-stocks-page">
      {/* Navigation Header */}
      <div className="nav2">
        <div className="logo-dark">
          <img src={logo} alt="Dark Logo" onClick={() => navigate('/')} />
        </div>
        <div className="page-title">
          <h1>Top Trading Opportunities</h1>
          <p className="timestamp">Updated: {new Date(data.timestamp).toLocaleString()}</p>
        </div>
        <div className="home-top">
          <img src={home} alt="Landing" onClick={() => navigate('/')} />
        </div>
      </div>

      {/* Filters */}
      <div className="filters-section">
        <div className="filter-buttons">
          <button
            className={filter === 'all' ? 'active' : ''}
            onClick={() => setFilter('all')}
          >
            All ({data.stocks.length})
          </button>
          <button
            className={filter === 'buy' ? 'active buy' : 'buy'}
            onClick={() => setFilter('buy')}
          >
            Buy Signals ({data.stocks.filter(s => getSignalCategory(s.final_signal) === 'buy').length})
          </button>
          <button
            className={filter === 'short' ? 'active short' : 'short'}
            onClick={() => setFilter('short')}
          >
            Short Signals ({data.stocks.filter(s => getSignalCategory(s.final_signal) === 'short').length})
          </button>
          <button
            className={filter === 'wait' ? 'active wait' : 'wait'}
            onClick={() => setFilter('wait')}
          >
            Wait Signals ({data.stocks.filter(s => getSignalCategory(s.final_signal) === 'wait').length})
          </button>
        </div>
      </div>

      {/* Stocks Grid */}
      <div className="stocks-grid">
        {filteredStocks.map((stock, index) => (
          <div
            key={stock.ticker}
            className="stock-card"
            onClick={() => navigate(`/results?ticker=${stock.ticker}`)}
          >
            <div className="stock-header">
              <div className="rank-badge">#{index + 1}</div>
              <div className="ticker-info-top">
                <h3>{stock.ticker}</h3>
                <p className="company-name">{stock.title}</p>
              </div>
              <div className="score-badge">
                <span className="score-value">{stock.score.toFixed(0)}</span>
                <span className="score-label">Score</span>
              </div>
            </div>

            <div className="stock-details">
              <div
                className="signal-badge"
                style={{ borderColor: getSignalColor(stock.final_signal) }}
              >
                <span style={{ color: getSignalColor(stock.final_signal) }}>
                  {stock.final_signal.replace(/_/g, ' ')}
                </span>
              </div>

              <div className="metrics-row">
                <div className="metric">
                  <span className="metric-label">Price</span>
                  <span className="metric-value">
                    {stock.current?.toFixed(2)} {stock.currency}
                  </span>
                </div>
                <div className="metric">
                  <span className="metric-label">Trend</span>
                  <span className={`metric-value trend-${stock.trend_direction?.toLowerCase()}`}>
                    {stock.trend_direction === 'UP' && '📈 UP'}
                    {stock.trend_direction === 'DOWN' && '📉 DOWN'}
                    {stock.trend_direction === 'NEUTRAL' && '➡️ NEUTRAL'}
                  </span>
                </div>
              </div>

              <div className="metrics-row">
                <div className="metric">
                  <span className="metric-label">Predictability</span>
                  <span className="metric-value">{stock.predictability_score}/5</span>
                </div>
                {stock.expected_edge_pct !== null && (
                  <div className="metric">
                    <span className="metric-label">Edge</span>
                    <span className="metric-value edge-positive">
                      {stock.expected_edge_pct.toFixed(2)}%
                    </span>
                  </div>
                )}
              </div>

              {stock.trade_quality !== null && (
                <div className="metrics-row">
                  <div className="metric">
                    <span className="metric-label">Setup Quality</span>
                    <span className="metric-value" style={{
                      color: stock.trade_quality >= 7 ? '#22c55e' :
                             stock.trade_quality >= 5 ? '#f59e0b' :
                             stock.trade_quality >= 3 ? '#f97316' : '#ef4444'
                    }}>
                      {stock.trade_quality.toFixed(1)}/10 {stock.quality_label && `(${stock.quality_label})`}
                    </span>
                  </div>
                  {stock.vp_confirming !== null && (
                    <div className="metric">
                      <span className="metric-label">Vol-Price</span>
                      <span className="metric-value" style={{
                        color: stock.vp_confirming ? '#22c55e' : '#ef4444'
                      }}>
                        {stock.vp_confirming ? '✓' : '✗'} {stock.vp_ratio?.toFixed(2)}
                      </span>
                    </div>
                  )}
                </div>
              )}

              {stock.regime_stability !== null && (
                <div className="stability-bar">
                  <span className="stability-label">Stability: {(stock.regime_stability * 100).toFixed(0)}%</span>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{
                        width: `${stock.regime_stability * 100}%`,
                        backgroundColor: stock.regime_stability > 0.7 ? '#22c55e' : stock.regime_stability > 0.6 ? '#f59e0b' : '#ef4444'
                      }}
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="card-footer">
              <span className="click-hint">Click to view full analysis →</span>
            </div>
          </div>
        ))}
      </div>

      {filteredStocks.length === 0 && (
        <div className="no-results-message">
          <p>No stocks found with {filter} signals.</p>
        </div>
      )}
    </div>
  )
}
