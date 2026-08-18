import { describe, it, expect } from 'vitest'
import { modifiedBeerLambert } from '../../lib/mbll'

const EXT_HBO = [0.975, 0.901, 1.046]
const EXT_HBR = [2.755, 0.781, 0.260]
const DPF = [6.51, 5.86, 5.12]
const SDS_MM = 30.0

describe('modifiedBeerLambert', () => {
  it('정적 입력(모든 샘플 동일)에 대해 HbO/HbR ≈ 0 반환', () => {
    // shape: [n_wl][n_ch][n_samples] = [3][4][3]
    const raw = Array.from({ length: 3 }, () =>
      Array.from({ length: 4 }, () => [1.0, 1.0, 1.0])
    )
    const { hbo, hbr } = modifiedBeerLambert(raw, EXT_HBO, EXT_HBR, DPF, SDS_MM)
    expect(hbo.length).toBe(4)
    hbo.forEach(ch => ch.forEach(v => expect(Math.abs(v)).toBeLessThan(1e-9)))
    hbr.forEach(ch => ch.forEach(v => expect(Math.abs(v)).toBeLessThan(1e-9)))
  })

  it('반환 shape: hbo/hbr [n_channels][n_samples]', () => {
    const raw = Array.from({ length: 3 }, () =>
      Array.from({ length: 4 }, () => Array(5).fill(1.0))
    )
    const { hbo, hbr } = modifiedBeerLambert(raw, EXT_HBO, EXT_HBR, DPF, SDS_MM)
    expect(hbo.length).toBe(4)
    expect(hbo[0].length).toBe(5)
    expect(hbr.length).toBe(4)
    expect(hbr[0].length).toBe(5)
  })

  it('강도 변화 시 유한한 숫자 반환', () => {
    const raw = [
      [[1.0, 1.05, 1.1], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
      [[1.0, 1.05, 1.1], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
      [[1.0, 1.05, 1.1], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    ]
    const { hbo } = modifiedBeerLambert(raw, EXT_HBO, EXT_HBR, DPF, SDS_MM)
    expect(isFinite(hbo[0][2])).toBe(true)
  })

  it('sds_mm = 0이면 에러', () => {
    const raw = [[[1.0]], [[1.0]], [[1.0]]]
    expect(() => modifiedBeerLambert(raw, [1], [1], [1], 0)).toThrow('sds_mm')
  })

  it('0 강도 입력 시 NaN 없음 (1e-10으로 대체)', () => {
    const raw = Array.from({ length: 3 }, () =>
      Array.from({ length: 4 }, () => [0.0, 1.0, 1.0])
    )
    const { hbo, hbr } = modifiedBeerLambert(raw, EXT_HBO, EXT_HBR, DPF, SDS_MM)
    hbo.forEach(ch => ch.forEach(v => expect(isFinite(v)).toBe(true)))
    hbr.forEach(ch => ch.forEach(v => expect(isFinite(v)).toBe(true)))
  })
})
