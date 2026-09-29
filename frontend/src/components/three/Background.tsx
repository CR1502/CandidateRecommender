import { Canvas } from '@react-three/fiber'
import { ParticleField } from './ParticleField'

/** Decorative particle background. Loaded lazily so three.js stays out of the main bundle. */
export default function Background({ fileCount }: { fileCount: number }) {
  return (
    <Canvas camera={{ position: [0, 0, 10], fov: 60 }}>
      <ParticleField fileCount={fileCount} />
    </Canvas>
  )
}
