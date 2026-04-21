import { useState } from 'react'
import SearchBar from './components/SearchBar'
import Answer from './components/Answer'
import Sources from './components/Sources'
import { querySpec } from './api'
import './App.css'

function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSearch = async (question) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await querySpec(question)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>specGPT</h1>
        <p className="subtitle">NVMe Specification Intelligence</p>
      </header>

      <main className="app-main">
        <SearchBar onSearch={handleSearch} loading={loading} />

        {error && (
          <div className="error-panel">
            {error}
          </div>
        )}

        <Answer result={result} />
        <Sources sources={result?.sources} />
      </main>

      <footer className="app-footer">
        <p>Answers sourced from NVM Express Base Specification 2.1</p>
      </footer>
    </div>
  )
}

export default App
