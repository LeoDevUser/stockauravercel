import { useState, useRef, useEffect } from 'react'
import '../styles/Tooltip.css'

interface TooltipProps {
  content: string | React.ReactNode
  position?: 'top' | 'bottom' | 'left' | 'right'
}

export default function Tooltip({ content, position = 'top' }: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [isCliked, setIsClicked] = useState(false)
  const [adjustedPosition, setAdjustedPosition] = useState(position)
  const tooltipRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isVisible && tooltipRef.current) {
      const tooltip = tooltipRef.current
      const rect = tooltip.getBoundingClientRect()

      // Check if tooltip goes off screen and adjust position
      if (position === 'top' && rect.top < 0) {
        setAdjustedPosition('bottom')
      } else if (position === 'bottom' && rect.bottom > window.innerHeight) {
        setAdjustedPosition('top')
      } else if (position === 'left' && rect.left < 0) {
        setAdjustedPosition('right')
      } else if (position === 'right' && rect.right > window.innerWidth) {
        setAdjustedPosition('left')
      }
    }
  }, [isVisible, position])

  return (
    <div className="tooltip-wrapper">
      <button
        className="tooltip-trigger"
        onClick={() => {
          setIsVisible(!isVisible)
          setIsClicked(!isCliked)
        }}
        onMouseOver={() => setIsVisible(true)}
        onMouseOut={() => setIsVisible(false || isCliked)}
        onBlur={() => {
          setIsVisible(false)
          setIsClicked(false)
        }}
        aria-label="More information"
        type="button"
      >
        <svg
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
          <path
            d="M8 11.5V8M8 5.5H8.01"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
      </button>
      {isVisible && (
        <div
          ref={tooltipRef}
          className={`tooltip-content tooltip-${adjustedPosition}`}
          role="tooltip"
        >
          {typeof content === 'string' ? <p>{content}</p> : content}
        </div>
      )}
    </div>
  )
}
