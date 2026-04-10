import { describe, it, expect } from 'vitest'
import { bwr, rbfInterpolate } from '../../lib/rbf'

describe('bwr (Blue-White-Red 컬러맵)', () => {
  it('t=0 → 파랑 [0,0,255]', () => {
    const [r, g, b] = bwr(0)
    expect(r).toBe(0); expect(g).toBe(0); expect(b).toBe(255)
  })

  it('t=0.5 → 흰색 [255,255,255]', () => {
    const [r, g, b] = bwr(0.5)
    expect(r).toBe(255); expect(g).toBe(255); expect(b).toBe(255)
  })

  it('t=1 → 빨강 [255,0,0]', () => {
    const [r, g, b] = bwr(1)
    expect(r).toBe(255); expect(g).toBe(0); expect(b).toBe(0)
  })

  it('t < 0이면 파랑으로 클리핑', () => {
    const [r, g, b] = bwr(-1)
    expect(r).toBe(0); expect(b).toBe(255)
  })

  it('t > 1이면 빨강으로 클리핑', () => {
    const [r, g, b] = bwr(2)
    expect(r).toBe(255); expect(b).toBe(0)
  })
})

describe('rbfInterpolate', () => {
  it('반환 길이가 queryPoints 수와 일치', () => {
    const points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
    const values = [1.0, -1.0, 0.0]
    const query = [[0.1, 0.1], [0.5, 0.5], [0.9, 0.1]]
    const result = rbfInterpolate(query, points, values)
    expect(result.length).toBe(3)
    result.forEach(v => expect(isFinite(v)).toBe(true))
  })

  it('채널 위치에서 해당 값에 가까운 보간값 반환', () => {
    const points = [[0.0, 0.0], [1.0, 0.0]]
    const values = [2.0, -2.0]
    const result = rbfInterpolate([[0.0, 0.0]], points, values)
    expect(result[0]).toBeCloseTo(2.0, 0)  // 채널 위치에서 정확한 값
  })

  it('4채널 PFC 위치 보간 — 숫자 반환', () => {
    const chPos = [[0.37, 0.36], [0.63, 0.36], [0.37, 0.52], [0.63, 0.52]]
    const values = [1.0, -1.0, 0.5, -0.5]
    const query = [[0.5, 0.44]]
    const result = rbfInterpolate(query, chPos, values)
    expect(result.length).toBe(1)
    expect(isFinite(result[0])).toBe(true)
  })
})
