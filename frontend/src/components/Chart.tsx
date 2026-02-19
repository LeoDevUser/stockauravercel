import { useEffect, useRef } from 'react'
import {
  createChart,
  ColorType,
  CandlestickSeries,
  type Time,
} from 'lightweight-charts'
import '../styles/Chart.css'

interface OHLCData {
  Date: string
  Open: number
  High: number
  Low: number
  Close: number
}

interface CandlestickChartProps {
  ohlcData: OHLCData[] | null
  ticker: string
}

export default function Chart({ ohlcData, ticker }: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<any>(null)

  useEffect(() => {
    if (!containerRef.current || !ohlcData || ohlcData.length === 0) return

    // Transform OHLC data to format expected by Lightweight Charts
    const candleData = ohlcData.map((candle) => {
      const date = new Date(candle.Date)
      const time = Math.floor(date.getTime() / 1000) as Time

      return {
        time,
        open: candle.Open,
        high: candle.High,
        low: candle.Low,
        close: candle.Close,
      }
    })

    // Create chart
    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#1e1e1e' },
        textColor: '#d1d5db',
        fontSize: 12,
        fontFamily: 'Inter, sans-serif',
      },
      width: containerRef.current.clientWidth,
      height: 400,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 5,
      },
      rightPriceScale: {
        autoScale: true,
        borderVisible: true,
        borderColor: '#374151',
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: '#4b5563',
          width: 1,
          style: 2,
          visible: true,
        },
        horzLine: {
          color: '#4b5563',
          width: 1,
          style: 2,
          visible: true,
        },
      },
      grid: {
        horzLines: {
          visible: true,
          color: '#2d2d2d',
        },
        vertLines: {
          visible: true,
          color: '#2d2d2d',
        },
      },
    })

    // Create candlestick series
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#1F5F7C', // Green for up
      downColor: '#F5722B', // Red for down
      borderUpColor: '#1F5F7C',
      borderDownColor: '#F5722B',
      wickUpColor: '#1F5F7C',
      wickDownColor: '#F5722B',
    })

    // Add data to series
    candlestickSeries.setData(candleData)

    // Set visible range (show last 50 candles or all if less than 50)
    const visibleRange = Math.min(50, candleData.length)
    const range = {
      from: candleData[candleData.length - visibleRange].time,
      to: candleData[candleData.length - 1].time,
    }
    chart.timeScale().setVisibleRange(range)

    // Fit content
    //chart.timeScale().fitContent()

    // Store chart reference for cleanup
    chartRef.current = chart

    // Handle window resize
    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: containerRef.current.clientWidth,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      //return the cleanup function
      window.removeEventListener('resize', handleResize)
      chart.remove()
      chartRef.current = null
    }
  }, [ohlcData])

  return (
    <div className="candlestick-chart-wrapper">
      <div className="chart-header">
        <h3>{ticker} - Price Movement</h3>
      </div>
      <div ref={containerRef} className="candlestick-chart-container" />
    </div>
  )
}
