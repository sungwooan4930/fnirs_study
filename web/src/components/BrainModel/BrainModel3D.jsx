/**
 * BrainModel3D — Three.js 기반 3D 뇌 모델
 *
 * brain.glb 파일이 있으면 GLB 로드, 없으면 절차적 뇌 형태로 대체.
 * PFC 4채널 위치에 반투명 구체(sphere) 블롭 표시.
 *
 * props:
 *   values  {number[]}  채널별 신호 값 (hbo 또는 hbr), 4개
 *   view    'front' | 'top'  카메라 프리셋
 */
import { useRef, useMemo, Suspense } from 'react'
import { Canvas, useFrame, useLoader } from '@react-three/fiber'
import { OrbitControls, useGLTF } from '@react-three/drei'
import * as THREE from 'three'

// 전전두엽 4채널 3D 좌표 (Three.js 좌표계: x좌우, y상하, z전후)
// 뇌 크기 ~1.5 단위 기준, 전두엽 전방·상단
const CHANNEL_3D_POSITIONS = [
  [-0.30,  0.55,  0.80],  // Ch1 좌전방 PFC
  [ 0.30,  0.55,  0.80],  // Ch2 우전방 PFC
  [-0.40,  0.25,  0.65],  // Ch3 좌후방 PFC
  [ 0.40,  0.25,  0.65],  // Ch4 우후방 PFC
]

const VMIN = -5.0
const VMAX = 5.0

function valToColor(val) {
  const t = Math.max(0, Math.min(1, (val - VMIN) / (VMAX - VMIN)))
  if (t < 0.5) {
    const f = t * 2
    return new THREE.Color(
      (78  + 177 * f) / 255,
      (146 + 109 * f) / 255,
      1
    )
  }
  const f = (t - 0.5) * 2
  const gb = 1 - f
  return new THREE.Color(1, gb, gb)
}

function valToOpacity(val) {
  const t = Math.abs((val - (VMIN + VMAX) / 2) / ((VMAX - VMIN) / 2))
  return 0.30 + Math.min(t, 1) * 0.40
}

// GLB 뇌 메시 — 파일이 없으면 Suspense fallback이 절차적 메시를 보여줌
function BrainMesh() {
  try {
    const { scene } = useGLTF('/brain.glb')
    const clone = useMemo(() => scene.clone(), [scene])
    return (
      <primitive
        object={clone}
        rotation={[Math.PI / 2, Math.PI, 0]}
        scale={[1, 1, 1]}
      />
    )
  } catch {
    return null
  }
}

// GLB 없을 때 대체: 절차적 뇌 형태 (타원체 + 주름 표현)
function ProceduralBrain() {
  return (
    <group>
      {/* 주 반구: 약간 납작한 타원체 */}
      <mesh>
        <sphereGeometry args={[1, 64, 48]} />
        <meshStandardMaterial
          color="#c8a882"
          roughness={0.85}
          metalness={0.05}
          side={THREE.FrontSide}
        />
      </mesh>
      {/* 좌우 반구 구분선 */}
      <mesh rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.015, 0.015, 2.1, 16]} />
        <meshStandardMaterial color="#8a6a50" roughness={1} />
      </mesh>
      {/* 전두엽 돌출 */}
      <mesh position={[0, 0.1, 0.8]} scale={[0.85, 0.75, 0.6]}>
        <sphereGeometry args={[0.6, 32, 24]} />
        <meshStandardMaterial color="#c8a882" roughness={0.85} metalness={0.05} />
      </mesh>
    </group>
  )
}

// 채널 블롭 구체
function ChannelSphere({ position, value }) {
  const meshRef = useRef()
  const color = valToColor(value)
  const opacity = valToOpacity(value)

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.material.opacity = opacity * (0.85 + 0.15 * Math.sin(Date.now() * 0.003))
    }
  })

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[0.13, 24, 24]} />
      <meshStandardMaterial
        color={color}
        transparent
        opacity={opacity}
        emissive={color}
        emissiveIntensity={0.3}
      />
    </mesh>
  )
}

// 카메라 프리셋
const CAMERA_PRESETS = {
  front: { position: [0, 0.3, 3.5], target: [0, 0, 0] },
  top:   { position: [0, 4.0, 0.1], target: [0, 0, 0] },
}

export default function BrainModel3D({ values = [0, 0, 0, 0], view = 'front' }) {
  const { position: camPos } = CAMERA_PRESETS[view] ?? CAMERA_PRESETS.front

  return (
    <div style={{ width: '100%', maxWidth: 960, aspectRatio: '16/10', margin: '0 auto' }}>
      <Canvas
        camera={{ position: camPos, fov: 45, near: 0.1, far: 100 }}
        style={{ background: '#0f1117', borderRadius: 12 }}
        gl={{ antialias: true, alpha: false }}
      >
        {/* 조명 */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[3, 4, 5]} intensity={0.9} />
        <directionalLight position={[-3, 2, -3]} intensity={0.3} color="#4e92ff" />

        {/* 뇌 메시 */}
        <Suspense fallback={<ProceduralBrain />}>
          <BrainMesh />
          <ProceduralBrain />
        </Suspense>

        {/* 채널 블롭 */}
        {CHANNEL_3D_POSITIONS.map((pos, ch) => (
          <ChannelSphere key={ch} position={pos} value={values[ch]} />
        ))}

        {/* 축 / 그리드 (개발용, 선택) */}
        {/* <gridHelper args={[4, 10, '#1e2130', '#1e2130']} position={[0, -1.1, 0]} /> */}

        <OrbitControls
          enablePan={false}
          minDistance={2}
          maxDistance={8}
          target={[0, 0, 0]}
        />
      </Canvas>
    </div>
  )
}
