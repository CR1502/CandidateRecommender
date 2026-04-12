import { useMemo } from 'react'
import { Html, Line } from '@react-three/drei'
import * as THREE from 'three'

interface Props {
  semantic: number      // 0–1
  skillCoverage: number // 0–1
  experience: number    // 0–1
  color: string
}

const AXES = [
  { label: 'Semantic',   angle: Math.PI / 2 },
  { label: 'Skills',     angle: Math.PI / 2 + (2 * Math.PI) / 3 },
  { label: 'Experience', angle: Math.PI / 2 + (4 * Math.PI) / 3 },
]

function toVec(angle: number, r: number): THREE.Vector3 {
  return new THREE.Vector3(Math.cos(angle) * r, Math.sin(angle) * r, 0)
}

function toTuple(angle: number, r: number): [number, number, number] {
  return [Math.cos(angle) * r, Math.sin(angle) * r, 0]
}

export function RadarChart3D({ semantic, skillCoverage, experience, color }: Props) {
  const maxR = 1.15
  const values = [semantic, skillCoverage, experience]

  const filledGeo = useMemo(() => {
    const pts = AXES.map((ax, i) => toVec(ax.angle, values[i] * maxR))
    const verts: number[] = []
    for (let i = 0; i < 3; i++) {
      const next = (i + 1) % 3
      verts.push(0, 0, 0, pts[i].x, pts[i].y, pts[i].z, pts[next].x, pts[next].y, pts[next].z)
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3))
    return geo
  }, [semantic, skillCoverage, experience])

  // Points for Drei <Line> — outline closes back to first point
  const outlinePoints = useMemo<[number, number, number][]>(() => {
    const pts = AXES.map((ax, i) => toTuple(ax.angle, values[i] * maxR))
    return [...pts, pts[0]] as [number, number, number][]
  }, [semantic, skillCoverage, experience])

  // Grid rings at 40%, 70%, 100% of maxR
  const gridRings = useMemo(() => [0.4, 0.7, 1.0].map(frac => {
    const pts = AXES.map(ax => toTuple(ax.angle, frac * maxR))
    return [...pts, pts[0]] as [number, number, number][]
  }), [])

  return (
    <group>
      <ambientLight intensity={0.5} />

      {/* Grid rings */}
      {gridRings.map((pts, idx) => (
        <Line key={idx} points={pts} color="#1e293b" lineWidth={1} transparent opacity={0.5} />
      ))}

      {/* Axis spokes */}
      {AXES.map((ax, i) => (
        <group key={ax.label}>
          <Line
            points={[[0, 0, 0], toTuple(ax.angle, maxR)]}
            color="#334155"
            lineWidth={1}
          />
          <Html position={toTuple(ax.angle, maxR + 0.42)} center distanceFactor={4.5}>
            <div style={{ textAlign: 'center', pointerEvents: 'none', userSelect: 'none' }}>
              <div style={{ color: '#64748b', fontSize: '0.58rem', whiteSpace: 'nowrap' }}>
                {ax.label}
              </div>
              <div style={{ color, fontWeight: 700, fontSize: '0.68rem' }}>
                {(values[i] * 100).toFixed(0)}%
              </div>
            </div>
          </Html>
        </group>
      ))}

      {/* Filled polygon */}
      <mesh geometry={filledGeo}>
        <meshBasicMaterial color={color} transparent opacity={0.22} side={THREE.DoubleSide} />
      </mesh>

      {/* Outline */}
      <Line points={outlinePoints} color={color} lineWidth={1.5} />
    </group>
  )
}
