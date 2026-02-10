import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import ResultsPage from './pages/ResultsPage'
import TopStocksPage from './pages/Top'
import './App.css'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="/top" element={<TopStocksPage />} />
      </Routes>
    </Router>
  )
}

export default App
