import { useState } from 'react'

export default function Sources({ sources }) {
  const [expanded, setExpanded] = useState(false)

  if (!sources || sources.length === 0) return null

  return (
    <div className="sources-panel">
      <button
        className="sources-toggle"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? '▼' : '▶'} Retrieved Sources ({sources.length})
      </button>

      {expanded && (
        <div className="sources-list">
          {sources.map((source, i) => (
            <div key={i} className="source-card">
              <div className="source-header">
                <span className="source-section">{source.section_id}</span>
                <span className="source-title">{source.section_title}</span>
                <span className="source-type">{source.content_type}</span>
              </div>
              <div className="source-text">{source.text_raw}</div>
              {source.pdf_pages && source.pdf_pages.length > 0 && (
                <div className="source-pages">
                  PDF pages: {source.pdf_pages.join(', ')}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
