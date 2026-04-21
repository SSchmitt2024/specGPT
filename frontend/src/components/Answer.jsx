export default function Answer({ result }) {
  if (!result) return null

  return (
    <div className="answer-panel">
      <div className="answer-text">
        {result.answer}
      </div>

      {result.citations && result.citations.length > 0 && (
        <div className="citations">
          <h4>Citations</h4>
          <ul>
            {result.citations.map((cite, i) => (
              <li key={i}>Section {cite}</li>
            ))}
          </ul>
        </div>
      )}

      {result.confidence && (
        <div className={`confidence confidence-${result.confidence.toLowerCase()}`}>
          Confidence: {result.confidence}
        </div>
      )}
    </div>
  )
}
