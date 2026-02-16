import { type AnalysisResult } from '../pages/ResultsPage.tsx'
import Tooltip from './Tooltip'
import '../styles/TradingVerdict.css'

interface TradingVerdictProps {
  results: AnalysisResult
  transactionCost: number
  tradeSize: number
  riskTolerance: number
}

export function TradingVerdict({ results, transactionCost, tradeSize, riskTolerance }: TradingVerdictProps) {
  if (!results.final_signal) return null

  const riskPerTrade = riskTolerance
  const noStopLoss = riskPerTrade >= 1.0
  
  // Determine if this is a short signal
  const isShortSignal = results.final_signal ? results.final_signal.includes('SHORT') : false

  // Calculate position sizing based on user's trade size
  const suggestedShares = Math.floor(tradeSize / results.current)
  const stopLossPrice = noStopLoss
    ? null
    : isShortSignal
      ? results.current * (1 + riskPerTrade)   // Short: stop ABOVE entry
      : results.current * (1 - riskPerTrade)    // Long: stop BELOW entry

  const positionRiskAmount = tradeSize * riskPerTrade
  const actualRiskPct = (positionRiskAmount / tradeSize) * 100
  
  // Position vs volume check
  const positionValue = suggestedShares * results.current
  const positionSizeVsVolume = results.avg_daily_volume 
    ? positionValue / (results.avg_daily_volume * results.current)
    : null

  // Check if position is executable
  const isSpeculativeSignal = results.final_signal.startsWith('SPEC_')
  const effectiveShares = isSpeculativeSignal ? Math.floor(suggestedShares / 2) : suggestedShares
  const canExecute = effectiveShares > 0
  
  // Generate position size note
  let positionSizeNote = ''
  if (!canExecute) {
    const minNeeded = Math.ceil(results.current * (isSpeculativeSignal ? 2 : 1))
    positionSizeNote = `Trade size too small. Minimum needed: $${minNeeded.toLocaleString()} (${isSpeculativeSignal ? '2 shares for half position' : '1 share'})`
  } else if (isSpeculativeSignal) {
    positionSizeNote = `⚠ SPECULATIVE SIGNAL: Position automatically halved to ${effectiveShares} shares due to lower statistical confidence (${results.predictability_score}/5 tests passed)`
  }

  // ============================================================================
  // FRICTION & EDGE CALCULATIONS
  // ============================================================================
  
  const dynamicSlippage = 0.0005
  const totalFriction = (dynamicSlippage + transactionCost) * 2
  const totalFrictionPct = totalFriction * 100

  const expectedEdge = results.expected_edge_pct || 0
  const edgeRatio = totalFrictionPct > 0 ? expectedEdge / totalFrictionPct : 0
  const edgeCoversCosts = edgeRatio > 3

  // ============================================================================
  // SIGNAL CONFIGURATION
  // ============================================================================
  
  const signalConfig: Record<string, {
    verdict: string
    action: string
    color: string
    riskLevel: string
    confidence: string
    summary: string
  }> = {
    'BUY_UPTREND': {
      verdict: 'TRADEABLE',
      action: '✓ BUY - Uptrend Continuing',
      color: '#22c55e',
      riskLevel: 'MEDIUM',
      confidence: 'HIGH',
      summary: 'Strong uptrend with positive momentum and ideal entry level'
    },
    'BUY_PULLBACK': {
      verdict: 'TRADEABLE',
      action: '✓ BUY - Pullback Entry',
      color: '#22c55e',
      riskLevel: 'MEDIUM',
      confidence: 'HIGH',
      summary: 'Strong uptrend with pullback dip - excellent entry point'
    },
    'BUY_MOMENTUM': {
      verdict: 'TRADEABLE',
      action: '✓ BUY - Momentum Play',
      color: '#22c55e',
      riskLevel: 'MEDIUM-HIGH',
      confidence: 'MEDIUM',
      summary: 'Positive momentum detected in uptrend'
    },
    'SHORT_DOWNTREND': {
      verdict: 'TRADEABLE',
      action: '✓ SHORT - Downtrend Continuing',
      color: '#ef4444',
      riskLevel: 'HIGH',
      confidence: 'HIGH',
      summary: 'Strong downtrend with persistent momentum and ideal entry level'
    },
    'SHORT_BOUNCES_ONLY': {
      verdict: 'TRADEABLE',
      action: '📉 SHORT - Bounces Only',
      color: '#ef4444',
      riskLevel: 'HIGH',
      confidence: 'MEDIUM',
      summary: 'Downtrend detected - only short bounces, avoid holding'
    },
    'SHORT_MOMENTUM': {
      verdict: 'TRADEABLE',
      action: '✓ SHORT - Momentum Play',
      color: '#ef4444',
      riskLevel: 'HIGH',
      confidence: 'MEDIUM',
      summary: 'Downtrend momentum detected'
    },
    'WAIT_PULLBACK': {
      verdict: 'WAIT',
      action: '⏸ WAIT - Pullback Expected',
      color: '#f59e0b',
      riskLevel: 'MEDIUM',
      confidence: 'MEDIUM',
      summary: 'Stock overbought (Z-EMA > 1.0) - wait for pullback before buying'
    },
    'WAIT_SHORT_BOUNCE': {
      verdict: 'WAIT',
      action: '⏸ WAIT - Bounce Expected',
      color: '#f59e0b',
      riskLevel: 'MEDIUM',
      confidence: 'MEDIUM',
      summary: 'Stock oversold (Z-EMA < -1.0) - wait for bounce to short'
    },
    'WAIT_OR_SHORT_BOUNCE': {
      verdict: 'WAIT',
      action: '⚠ WAIT and Buy the Dip or SHORT Bounce',
      color: '#f59e0b',
      riskLevel: 'MEDIUM-HIGH',
      confidence: 'MEDIUM',
      summary: 'Overbought in mean-reverting uptrend — wait for a pullback to buy the dip, or short the bounce now'
    },
    'WAIT_FOR_REVERSAL': {
      verdict: 'WAIT',
      action: '⏸ WAIT - Reversal Possible',
      color: '#f59e0b',
      riskLevel: 'MEDIUM',
      confidence: 'MEDIUM',
      summary: 'Downtrend momentum weakening - wait for reversal confirmation'
    },
    'WAIT_FOR_TREND': {
      verdict: 'WAIT',
      action: '⏸ WAIT - No Clear Trend',
      color: '#f59e0b',
      riskLevel: 'HIGH',
      confidence: 'LOW',
      summary: 'Strong momentum but no clear trend yet - wait for trend confirmation'
    },
    'NO_CLEAR_SIGNAL': {
      verdict: 'DO NOT TRADE',
      action: '⚪ NO CLEAR SIGNAL',
      color: '#888',
      riskLevel: 'N/A',
      confidence: 'LOW',
      summary: 'Insufficient evidence for trade - momentum too weak'
    },
    'DO_NOT_TRADE': {
      verdict: 'DO NOT TRADE',
      action: '❌ DO NOT TRADE',
      color: '#ef4444',
      riskLevel: 'AVOID',
      confidence: 'HIGH',
      summary: 'Pattern is unreliable or unstable - avoid trading'
    },
    // ── SPECULATIVE TIER ──────────────────────────────────────────────
    'SPEC_BUY_UPTREND': {
      verdict: 'SPECULATIVE',
      action: '⚠ SPECULATIVE BUY - Uptrend',
      color: '#f97316',
      riskLevel: 'HIGH',
      confidence: 'LOW',
      summary: 'Uptrend with momentum detected, but only 2/5 statistical tests passed — use half position size'
    },
    'SPEC_BUY_PULLBACK': {
      verdict: 'SPECULATIVE',
      action: '⚠ SPECULATIVE BUY - Pullback',
      color: '#f97316',
      riskLevel: 'HIGH',
      confidence: 'LOW',
      summary: 'Pullback entry in uptrend, but only 2/5 statistical tests passed — use half position size'
    },
    'SPEC_BUY_MOMENTUM': {
      verdict: 'SPECULATIVE',
      action: '⚠ SPECULATIVE BUY - Momentum',
      color: '#f97316',
      riskLevel: 'HIGH',
      confidence: 'LOW',
      summary: 'Momentum detected in uptrend, but only 2/5 statistical tests passed — use half position size'
    },
    'SPEC_SHORT_DOWNTREND': {
      verdict: 'SPECULATIVE',
      action: '⚠ SPECULATIVE SHORT - Downtrend',
      color: '#f97316',
      riskLevel: 'VERY HIGH',
      confidence: 'LOW',
      summary: 'Downtrend continuing, but only 2/5 statistical tests passed — use half position size'
    },
    'SPEC_SHORT_BOUNCES_ONLY': {
      verdict: 'SPECULATIVE',
      action: '⚠ SPECULATIVE SHORT - Bounces Only',
      color: '#f97316',
      riskLevel: 'VERY HIGH',
      confidence: 'LOW',
      summary: 'Downtrend bounce short, but only 2/5 statistical tests passed — use half position size'
    },
    'SPEC_SHORT_MOMENTUM': {
      verdict: 'SPECULATIVE',
      action: '⚠ SPECULATIVE SHORT - Momentum',
      color: '#f97316',
      riskLevel: 'VERY HIGH',
      confidence: 'LOW',
      summary: 'Downtrend momentum, but only 2/5 statistical tests passed — use half position size'
    },
    'SPEC_WAIT_OR_SHORT_BOUNCE': {
      verdict: 'SPECULATIVE',
      action: '⚠ SPECULATIVE - Wait/Short Bounce',
      color: '#f97316',
      riskLevel: 'HIGH',
      confidence: 'LOW',
      summary: 'Uptrend weakening, but only 2/5 statistical tests passed — use half position size if trading'
    },
    'SPEC_WAIT_FOR_REVERSAL': {
      verdict: 'SPECULATIVE',
      action: '⚠ SPECULATIVE - Reversal Possible',
      color: '#f97316',
      riskLevel: 'HIGH',
      confidence: 'LOW',
      summary: 'Downtrend weakening, but only 2/5 statistical tests passed — use half position size if trading'
    }
  }

  const config = signalConfig[results.final_signal] || {
    verdict: 'UNKNOWN',
    action: results.final_signal,
    color: '#888',
    riskLevel: 'UNKNOWN',
    confidence: 'UNKNOWN',
    summary: 'Unknown signal'
  }

  // ============================================================================
  // FAILURE REASONS & VALIDATION
  // ============================================================================
  
  const failureReasons = []
  if (results.predictability_score < 2) {
    failureReasons.push({
      metric: 'Predictability Score',
      value: `${results.predictability_score}/5 (need ≥2)`,
      detail: 'Insufficient statistical tests passed'
    })
  }
  if (results.regime_stability !== null && results.regime_stability < 0.5) {
    failureReasons.push({
      metric: 'Regime Stability',
      value: results.regime_stability === 0 
        ? '0% — momentum REVERSED out-of-sample' 
        : `${(results.regime_stability * 100).toFixed(0)}% (need ≥50%)`,
      detail: results.regime_stability === 0
        ? 'In-sample and out-of-sample momentum point in opposite directions'
        : 'Pattern degrades significantly out-of-sample'
    })
  }
  if (!edgeCoversCosts) {
    failureReasons.push({
      metric: 'Edge vs Friction',
      value: `${edgeRatio.toFixed(1)}x (need >3x)`,
      detail: 'Statistical edge too small to cover trading costs'
    })
  }
  if (results.momentum_corr !== null && Math.abs(results.momentum_corr) <= 0.08) {
    failureReasons.push({
      metric: 'Momentum',
      value: `${(results.momentum_corr * 100).toFixed(1)}% (need >8%)`,
      detail: 'No detectable multi-day momentum pattern'
    })
  }
  if (results.vp_confirming === false) {
    failureReasons.push({
      metric: 'Volume-Price',
      value: results.vp_ratio !== null ? `Ratio: ${results.vp_ratio.toFixed(2)}` : 'N/A',
      detail: 'Volume does not confirm trend direction'
    })
  }

  const hasFailed = config.verdict === 'DO NOT TRADE'
  const isSpeculative = config.verdict === 'SPECULATIVE'

  // Build speculative warning details
  const specWarnings: string[] = []
  if (isSpeculative) {
    if (results.predictability_score < 3) {
      const passed: string[] = []
      const failed: string[] = []
      if (results.momentum_corr !== null) {
        if (Math.abs(results.momentum_corr) > 0.08) passed.push('Momentum')
        else failed.push('Momentum')
      }
      if (results.hurst_significant !== null) {
        if (results.hurst_significant) passed.push('Hurst/DFA')
        else failed.push('Hurst/DFA')
      }
      if (results.mean_rev_up !== null && results.mean_rev_down !== null) {
        if (Math.abs(results.mean_rev_up) > 0.003 && Math.abs(results.mean_rev_down) > 0.003) passed.push('Mean Reversion')
        else failed.push('Mean Reversion')
      }
      if (results.regime_stability !== null) {
        if (results.regime_stability >= 0.5) passed.push('Regime Stability')
        else failed.push('Regime Stability')
      }
      if (results.vp_confirming !== null) {
        if (results.vp_confirming) passed.push('Volume-Price')
        else failed.push('Volume-Price')
      }
      if (failed.length > 0) specWarnings.push(`Failed tests: ${failed.join(', ')}`)
      if (passed.length > 0) specWarnings.push(`Passed tests: ${passed.join(', ')}`)
    }
  }

  return (
    <div className="trading-verdict-container">
      <h2 className="verdict-title">📊 Trading Verdict</h2>

      {/* Main Verdict Box */}
      <div 
        className={`verdict-main ${config.verdict.toLowerCase().replace(/ /g, '-')}`}
        style={{ borderLeftColor: config.color }}
      >
        <div className="verdict-header">
          <h3 style={{ color: config.color }}>{config.action}</h3>
          <div className="verdict-badges">
            <span className={`badge risk-${config.riskLevel.toLowerCase().replace('-', '_')}`}>
              Risk: {config.riskLevel}
            </span>
            <span className={`badge conf-${config.confidence.toLowerCase()}`}>
              Confidence: {config.confidence}
            </span>
          </div>
        </div>

        <p className="verdict-summary">{config.summary}</p>

        {/* Show failure reasons if DO NOT TRADE */}
        {hasFailed && failureReasons.length > 0 && (
          <div className="failure-reasons">
            <strong>❌ Failed Validation Checks:</strong>
            <ul>
              {failureReasons.map((reason, idx) => (
                <li key={idx}>
                  <strong>{reason.metric}:</strong> {reason.value}
                  <br />
                  <small>{reason.detail}</small>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Show what passed if it's tradeable */}
        {!hasFailed && !isSpeculative && (
          <div className="passing-metrics">
            <strong>✓ Validation Passed:</strong>
            <div className="metrics-grid-compact">
              <div className="metric-item">
                <span>Predictability</span>
                <strong style={{ color: '#22c55e' }}>{results.predictability_score}/5</strong>
              </div>
              {results.regime_stability !== null && (
                <div className="metric-item">
                  <span>Regime Stability</span>
                  <strong style={{ color: '#22c55e' }}>{(results.regime_stability * 100).toFixed(0)}%</strong>
                </div>
              )}
              <div className="metric-item">
                <span>Edge/Friction</span>
                <strong style={{ color: '#22c55e' }}>{edgeRatio.toFixed(1)}x</strong>
              </div>
              {results.momentum_corr !== null && (
                <div className="metric-item">
                  <span>Momentum</span>
                  <strong style={{ color: '#22c55e' }}>{(results.momentum_corr * 100).toFixed(1)}%</strong>
                </div>
              )}
              {results.vp_confirming !== null && (
                <div className="metric-item">
                  <span>Vol-Price</span>
                  <strong style={{ color: results.vp_confirming ? '#22c55e' : '#f59e0b' }}>
                    {results.vp_confirming ? '✓' : '✗'} {results.vp_ratio?.toFixed(2)}
                  </strong>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Trade Quality Score */}
        {!hasFailed && results.trade_quality !== null && (
          <div style={{
            marginTop: '1em',
            padding: '0.8em 1em',
            background: 'rgba(255,255,255,0.03)',
            borderRadius: '8px',
            border: '1px solid rgba(255,255,255,0.08)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.8em', marginBottom: '0.5em' }}>
              <strong style={{ color: '#e0e0e0' }}>Setup Quality:</strong>
              <span style={{
                fontSize: '1.2em',
                fontWeight: 'bold',
                color: results.trade_quality >= 7 ? '#22c55e' :
                       results.trade_quality >= 5 ? '#f59e0b' :
                       results.trade_quality >= 3 ? '#f97316' : '#ef4444'
              }}>
                {results.trade_quality.toFixed(1)}/10
              </span>
              <span style={{
                fontSize: '0.8em',
                padding: '2px 8px',
                borderRadius: '4px',
                background: results.quality_label === 'Excellent' ? 'rgba(34,197,94,0.15)' :
                             results.quality_label === 'Good' ? 'rgba(245,158,11,0.15)' :
                             results.quality_label === 'Fair' ? 'rgba(249,115,22,0.15)' : 'rgba(239,68,68,0.15)',
                color: results.quality_label === 'Excellent' ? '#22c55e' :
                       results.quality_label === 'Good' ? '#f59e0b' :
                       results.quality_label === 'Fair' ? '#f97316' : '#ef4444'
              }}>
                {results.quality_label}
              </span>
            </div>
            {results.quality_components && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5em 1.5em', fontSize: '0.8em', color: '#999' }}>
                <span>Trend Align: <strong style={{ color: '#ccc' }}>{results.quality_components.trend_alignment.toFixed(1)}/2</strong>
                  <Tooltip content={<><strong>Trend Alignment (0–2)</strong><p>Measures how well momentum direction matches the current trend.</p><p><strong>2.0:</strong> Strong agreement | <strong>1.5:</strong> Moderate | <strong>1.0:</strong> Weak | <strong>0.0:</strong> Conflicting</p></>} />
                </span>
                <span>Entry Timing: <strong style={{ color: '#ccc' }}>{results.quality_components.entry_timing.toFixed(1)}/2</strong>
                  <Tooltip content={<><strong>Entry Timing / Z-EMA (0–2)</strong><p>Evaluates entry timing using Z-score relative to exponential moving average.</p><p><strong>2.0:</strong> Sweet spot | <strong>1.5:</strong> Moderate | <strong>0.5:</strong> Overbought/oversold</p></>} />
                </span>
                <span>Sharpe: <strong style={{ color: '#ccc' }}>{results.quality_components.sharpe_quality.toFixed(1)}/2</strong>
                  <Tooltip content={<><strong>Sharpe Quality (0–2)</strong><p>Risk-adjusted return quality.</p><p><strong>2.0:</strong> Sharpe ≥ 1.5 | <strong>1.0:</strong> Sharpe ≥ 0.5 | <strong>0.5:</strong> Sharpe &gt; 0</p></>} />
                </span>
                <span>Vol Fit: <strong style={{ color: '#ccc' }}>{results.quality_components.volatility_fit.toFixed(1)}/2</strong>
                  <Tooltip content={<><strong>Volatility Fit (0–2)</strong><p>Whether volatility is in the ideal trading range.</p><p><strong>2.0:</strong> 20–35% | <strong>1.5:</strong> 15–45% | <strong>0.3:</strong> Outside range</p></>} />
                </span>
                <span>Vol-Price: <strong style={{ color: '#ccc' }}>{results.quality_components.volume_confirmation.toFixed(1)}/2</strong>
                  <Tooltip content={<><strong>Volume-Price Confirmation (0–2)</strong><p>Whether volume supports the trend direction.</p><p><strong>2.0:</strong> Strong confirmation | <strong>0.2:</strong> No confirmation</p></>} />
                </span>
              </div>
            )}
          </div>
        )}

        {/* Speculative tier details */}
        {isSpeculative && (
          <div className="speculative-details" style={{ marginTop: '1em' }}>
            <div style={{ 
              background: 'rgba(249, 115, 22, 0.1)', 
              border: '1px solid rgba(249, 115, 22, 0.3)',
              borderRadius: '8px', 
              padding: '1em',
              marginBottom: '1em'
            }}>
              <strong style={{ color: '#f97316' }}>⚠ Speculative Signal — Reduced Confidence</strong>
              <p style={{ color: '#d0d0d0', fontSize: '0.9em', margin: '0.5em 0 0 0', lineHeight: '1.5' }}>
                Only {results.predictability_score}/5 statistical tests passed (high-conviction requires 3/5). 
                <strong> Position automatically halved.</strong>
              </p>
              {specWarnings.length > 0 && (
                <ul style={{ margin: '0.5em 0 0 0', paddingLeft: '1.5em', fontSize: '0.85em', color: '#bbb' }}>
                  {specWarnings.map((w, i) => <li key={i}>{w}</li>)}
                </ul>
              )}
            </div>
            <div className="metrics-grid-compact">
              <div className="metric-item">
                <span>Predictability</span>
                <strong style={{ color: '#f97316' }}>{results.predictability_score}/5</strong>
              </div>
              {results.regime_stability !== null && (
                <div className="metric-item">
                  <span>Regime Stability</span>
                  <strong style={{ color: results.regime_stability >= 1.0 ? '#22c55e' : '#f59e0b' }}>
                    {(results.regime_stability * 100).toFixed(0)}%
                  </strong>
                </div>
              )}
              <div className="metric-item">
                <span>Edge/Friction</span>
                <strong style={{ color: '#22c55e' }}>{edgeRatio.toFixed(1)}x</strong>
              </div>
              {results.momentum_corr !== null && (
                <div className="metric-item">
                  <span>Momentum</span>
                  <strong style={{ color: '#22c55e' }}>{(results.momentum_corr * 100).toFixed(1)}%</strong>
                </div>
              )}
              {results.vp_confirming !== null && (
                <div className="metric-item">
                  <span>Vol-Price</span>
                  <strong style={{ color: results.vp_confirming ? '#22c55e' : '#f59e0b' }}>
                    {results.vp_confirming ? '✓' : '✗'} {results.vp_ratio?.toFixed(2)}
                  </strong>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Position Details - Only show if tradeable or wait */}
      {config.verdict !== 'DO NOT TRADE' && (
        <div className="position-details">
          <h4>Position Setup</h4>
          
          {canExecute ? (
            <>
              <div className="position-grid">
                <div className="position-item">
                  <label>Entry</label>
                  <span className="value">{effectiveShares} shares @ ${results.current?.toFixed(2)}</span>
                  <small>Position: ${(effectiveShares * (results.current || 0)).toFixed(2)}</small>
                </div>
                <div className="position-item">
                  <label>Stop Loss</label>
                  {noStopLoss ? (
                    <>
                      <span className="value" style={{ color: '#f59e0b' }}>None</span>
                      <small>100% risk tolerance — no stop loss</small>
                    </>
                  ) : (
                    <>
                      <span className="value" style={{ color: '#ef4444' }}>
                        ${stopLossPrice?.toFixed(2)}
                      </span>
					  <small>{(riskPerTrade * 100).toFixed(1)}% {isShortSignal ? 'above' : 'below'} entry</small>
                    </>
                  )}
                </div>
                <div className="position-item">
                  <label>Max Loss</label>
                  <span className="value">
                    ${(effectiveShares * results.current * riskPerTrade).toFixed(2)}
                  </span>
                  <small>
                    {actualRiskPct.toFixed(1)}% of ${tradeSize.toLocaleString()} trade
                    {isSpeculative && ' (halved)'}
                  </small>
                </div>
              </div>
              
              {positionSizeNote && (
                <div className="position-note">
                  <p style={{ color: isSpeculative ? '#f97316' : '#f59e0b', fontSize: '0.9em', marginTop: '1em', lineHeight: '1.5' }}>
                    💡 {positionSizeNote}
                  </p>
                </div>
              )}
            </>
          ) : (
            <div className="position-warning">
              <p style={{ color: '#ef4444', margin: '1em 0', fontSize: '1.1em' }}>
                <strong>⚠ Cannot Execute Position</strong>
              </p>
              {positionSizeNote && (
                <p style={{ color: '#d0d0d0', fontSize: '0.95em', lineHeight: '1.6', marginTop: '0.75em' }}>
                  {positionSizeNote}
                </p>
              )}
              <p style={{ color: '#999', fontSize: '0.85em', marginTop: '1em', fontStyle: 'italic' }}>
                The trading pattern is still valid. Increase your trade size to execute.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Market Context */}
      <div className="market-context">
        <h4>Market Analysis</h4>
        <div className="context-grid">
          <div className="context-item">
            <span className="context-label">Momentum Pattern</span>
            <span className="context-value">
              {results.momentum_corr !== null && results.momentum_corr > 0.08
                ? `Positive (${(results.momentum_corr * 100).toFixed(1)}%) — trends CONTINUE`
                : results.momentum_corr !== null && results.momentum_corr < -0.08
                ? `Negative (${(results.momentum_corr * 100).toFixed(1)}%) — trends REVERSE`
                : 'Weak momentum — no clear pattern'}
            </span>
          </div>
          <div className="context-item">
            <span className="context-label">Market Regime</span>
            <span className="context-value">
              {results.hurst !== null && results.hurst > 0.55
                ? `TRENDING (Hurst ${results.hurst.toFixed(3)}) — follow trends`
                : results.hurst !== null && results.hurst < 0.45
                ? `MEAN-REVERTING (Hurst ${results.hurst.toFixed(3)}) — fade extremes`
                : `RANDOM WALK (Hurst ${results.hurst?.toFixed(3) || 'N/A'}) — no pattern`}
            </span>
          </div>
          <div className="context-item">
            <span className="context-label">Price Position</span>
            <span className="context-value">
              {results.z_ema !== null && results.z_ema > 1.0
                ? `Overbought (Z-EMA ${results.z_ema.toFixed(2)})`
                : results.z_ema !== null && results.z_ema < -1.0
                ? `Oversold (Z-EMA ${results.z_ema.toFixed(2)})`
                : results.z_ema !== null && results.z_ema > -0.5 && results.z_ema < 1.0
                ? `Sweet spot (Z-EMA ${results.z_ema.toFixed(2)})`
                : `Moderate (Z-EMA ${results.z_ema?.toFixed(2) || 'N/A'})`}
            </span>
          </div>
          <div className="context-item">
            <span className="context-label">Current Trend</span>
            <span className="context-value">
              {results.trend_direction === 'UP' && '📈 UPTREND'}
              {results.trend_direction === 'DOWN' && '📉 DOWNTREND'}
              {results.trend_direction === 'NEUTRAL' && '➡️ NEUTRAL'}
              {results.recent_return_1y !== null && ` (${(results.recent_return_1y * 100).toFixed(1)}% 1Y)`}
            </span>
          </div>
        </div>
      </div>

      {/* Liquidity & Costs */}
      <div className="liquidity-costs">
        <h4>Liquidity & Trading Costs</h4>
        <div className="costs-grid">
          <div className="cost-item">
            <label>Daily Volume</label>
            <span>{results.avg_daily_volume?.toLocaleString() || 'N/A'}</span>
          </div>
          <div className="cost-item">
            <label>Position vs Volume</label>
            <span className={positionSizeVsVolume && positionSizeVsVolume > 0.02 ? 'warning' : ''}>
              {((positionSizeVsVolume || 0) * 100).toFixed(3)}%
            </span>
          </div>
          <div className="cost-item">
            <label>Est. Slippage</label>
            <span>{((dynamicSlippage * 100)).toFixed(3)}%</span>
          </div>
          <div className="cost-item">
            <label>Transaction Cost</label>
            <span>{(transactionCost * 100).toFixed(3)}%</span>
          </div>
          <div className="cost-item total">
            <label>Total Friction (Round Trip)</label>
            <span>{totalFrictionPct.toFixed(3)}%</span>
          </div>
          <div className="cost-item edge">
            <label>Expected Edge (Annual)</label>
            <span>{expectedEdge.toFixed(2)}%</span>
          </div>
          <div className="cost-item ratio">
            <label>Edge / Friction Ratio</label>
            <span style={{ color: edgeCoversCosts ? '#22c55e' : '#ef4444' }}>
              {edgeRatio.toFixed(1)}x {edgeCoversCosts ? '✓' : '✗'}
            </span>
          </div>
        </div>
        
      </div>

      {/* Disclaimer */}
      <div className="verdict-disclaimer">
        <p>
          <strong>⚠ Important:</strong> Historical patterns don't guarantee future performance. 
          {config.verdict === 'TRADEABLE' && ' Paper trade first to validate the strategy in real conditions.'}
          {config.verdict === 'WAIT' && ' Wait for the specified setup before entering.'}
          {config.verdict === 'DO NOT TRADE' && ' This stock does not meet minimum statistical requirements for reliable trading.'}
        </p>
      </div>
    </div>
  )
}
