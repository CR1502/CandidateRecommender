/** Circular score gauge (SVG). Replaces the WebGL ScoreOrb: no GPU context per card. */
export function ScoreRing({ score, color, size = 88 }: { score: number; color: string; size?: number }) {
  const stroke = 8
  const r = (size - stroke) / 2
  const circumference = 2 * Math.PI * r
  const clamped = Math.max(0, Math.min(100, score))

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label={`Overall match ${clamped.toFixed(0)} percent`}
    >
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#1e1e2e" strokeWidth={stroke} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={circumference * (1 - clamped / 100)}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        style={{ transition: 'stroke-dashoffset 0.8s ease-out' }}
      />
      <text x="50%" y="48%" textAnchor="middle" dominantBaseline="middle" fill="#e2e8f0" fontSize={size * 0.24} fontWeight={700}>
        {clamped.toFixed(0)}%
      </text>
      <text x="50%" y="68%" textAnchor="middle" dominantBaseline="middle" fill="#64748b" fontSize={size * 0.12}>
        match
      </text>
    </svg>
  )
}
