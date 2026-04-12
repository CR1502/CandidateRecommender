import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'

interface Props {
  score: number   // 0–100
  color: string   // hex
  size?: number
}

export function ScoreOrb({ score, color, size = 1 }: Props) {
  const outerRef = useRef<THREE.Mesh>(null)
  const innerRef = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    if (outerRef.current) {
      outerRef.current.rotation.y = t * 0.55
      outerRef.current.rotation.z = t * 0.28
    }
    if (innerRef.current) {
      // Subtle breathing
      const breathe = 1 + Math.sin(t * 1.8) * 0.03
      innerRef.current.scale.setScalar(breathe)
    }
  })

  return (
    <group>
      {/* Ambient light for the orb */}
      <ambientLight intensity={0.3} />
      <pointLight position={[2, 2, 2]} intensity={1.2} color={color} />

      {/* Inner glowing sphere */}
      <mesh ref={innerRef}>
        <sphereGeometry args={[size * 0.68, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.5}
          roughness={0.15}
          metalness={0.4}
        />
      </mesh>

      {/* Outer spinning wireframe */}
      <mesh ref={outerRef}>
        <sphereGeometry args={[size * 0.96, 14, 14]} />
        <meshBasicMaterial color={color} wireframe transparent opacity={0.18} />
      </mesh>

      {/* Score label via HTML overlay */}
      <Html center distanceFactor={5}>
        <div
          style={{
            color: '#ffffff',
            fontWeight: 800,
            fontSize: '1.05rem',
            textShadow: `0 0 14px ${color}, 0 0 30px ${color}`,
            whiteSpace: 'nowrap',
            userSelect: 'none',
            pointerEvents: 'none',
          }}
        >
          {score.toFixed(1)}%
        </div>
      </Html>
    </group>
  )
}
