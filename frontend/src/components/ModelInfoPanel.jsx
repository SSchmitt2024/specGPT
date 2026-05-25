import { useState, useEffect } from 'react'
import { getModels } from '../api'

const STAGE_LABELS = {
  embedding: 'Embedding',
  reranker: 'Reranker',
  llm: 'LLM (standard)',
  agentic_llm: 'LLM (agentic)',
}

function formatCost(dollars) {
  if (dollars === null || dollars === undefined) return '—'
  if (dollars < 0.001) return '<$0.001'
  return `$${dollars.toFixed(4)}`
}

function computeCost(tokensUsed, models, isAgentic) {
  if (!tokensUsed || !models) return null
  const llm = isAgentic ? models.agentic_llm : models.llm
  if (!llm) return null
  const inputCost = (tokensUsed.prompt / 1_000_000) * llm.price_per_1m_input
  const outputCost = (tokensUsed.completion / 1_000_000) * llm.price_per_1m_output
  return {
    input: inputCost,
    output: outputCost,
    total: inputCost + outputCost,
    prompt: tokensUsed.prompt,
    completion: tokensUsed.completion,
  }
}

export default function ModelInfoPanel({ tokensUsed, isAgentic }) {
  const [models, setModels] = useState(null)
  const [open, setOpen] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    getModels()
      .then(setModels)
      .catch(e => setError(e.message))
  }, [])

  const cost = computeCost(tokensUsed, models, isAgentic)

  return (
    <div className="model-info-panel">
      <button
        className="model-info-toggle"
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
      >
        <span className="model-info-icon">⚙</span>
        Models &amp; Cost
        {cost && (
          <span className="model-info-cost-badge">
            {formatCost(cost.total)} / query
          </span>
        )}
        <span className="model-info-chevron">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="model-info-body">
          {error && <p className="model-info-error">{error}</p>}

          {models && (
            <table className="model-table">
              <thead>
                <tr>
                  <th>Stage</th>
                  <th>Model</th>
                  <th>Provider</th>
                  <th>$/1M in</th>
                  <th>$/1M out</th>
                  <th>Note</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(models).map(([key, info]) => (
                  <tr
                    key={key}
                    className={
                      (key === 'llm' && !isAgentic) || (key === 'agentic_llm' && isAgentic)
                        ? 'model-row-active'
                        : ''
                    }
                  >
                    <td>{STAGE_LABELS[key] ?? key}</td>
                    <td><code>{info.model}</code></td>
                    <td>{info.provider}</td>
                    <td>{info.price_per_1m_input === 0 ? 'free' : info.price_per_1m_input != null ? `$${info.price_per_1m_input}` : '—'}</td>
                    <td>{info.price_per_1m_output != null ? `$${info.price_per_1m_output}` : '—'}</td>
                    <td className="model-note">{info.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {cost && (
            <div className="model-cost-breakdown">
              <span className="cost-label">Last query:</span>
              <span className="cost-item">{cost.prompt.toLocaleString()} input tokens → {formatCost(cost.input)}</span>
              <span className="cost-sep">+</span>
              <span className="cost-item">{cost.completion.toLocaleString()} output tokens → {formatCost(cost.output)}</span>
              <span className="cost-sep">=</span>
              <span className="cost-total">{formatCost(cost.total)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
