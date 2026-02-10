import { useNavigate } from 'react-router-dom'
import SearchBar from '../components/SearchBar'
import logo from '../assets/logo-dark.png'
import '../styles/LandingPage.css'

export default function LandingPage() {
  const navigate = useNavigate()

  const handleNavigate = (ticker: string) => {
    navigate(`/results?ticker=${ticker}`)
  }

  return (
    <div className="landing-container">
      <div className="landing-content">
        <div className="logo">
          <img src={logo} alt="STOCKAURA LOGO"/>
        </div>

        <div className="search-section-landing">
          <SearchBar 
            onSelect={handleNavigate} 
            placeholder="Enter a stock ticker to analyze it.." 
          />
        </div>
		<div className='go-top' onClick={() => navigate('/top')}>
	  <h1>Top Trading Opportunities</h1>
		</div>
      </div>
    </div>
  )
}
