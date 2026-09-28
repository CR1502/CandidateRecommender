import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface Props {
  fileCount: number
}

const COUNT = 800

// Generated once at module load — random values can't be produced during render.
const POSITIONS = (() => {
  const arr = new Float32Array(COUNT * 3)
  for (let i = 0; i < COUNT; i++) {
    arr[i * 3]     = (Math.random() - 0.5) * 22
    arr[i * 3 + 1] = (Math.random() - 0.5) * 22
    arr[i * 3 + 2] = (Math.random() - 0.5) * 22
  }
  return arr
})()

export function ParticleField({ fileCount }: Props) {
  const ref = useRef<THREE.Points>(null)

  useFrame(({ clock }) => {
    if (!ref.current) return
    const t = clock.getElapsedTime()
    ref.current.rotation.y = t * 0.04
    ref.current.rotation.x = t * 0.02
    const pulse = fileCount > 0 ? 1 + Math.sin(t * 2.5) * 0.04 : 1
    ref.current.scale.setScalar(pulse)
  })

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[POSITIONS, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.055}
        color={fileCount > 0 ? '#818cf8' : '#2d3748'}
        transparent
        opacity={fileCount > 0 ? 0.65 : 0.4}
        sizeAttenuation
      />
    </points>
  )
}
