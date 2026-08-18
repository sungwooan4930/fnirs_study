import { describe, it, expect } from 'vitest'
import { bandpassFilter } from '../../lib/filters'

describe('bandpassFilter', () => {
  it('반환 길이가 입력과 동일', () => {
    const data = Array.from({ length: 100 }, (_, i) => Math.sin(2 * Math.PI * 0.1 * i / 10))
    const out = bandpassFilter(data, 0.01, 0.5, 10.0)
    expect(out.length).toBe(data.length)
  })

  it('통과대역 신호 유지 — 0.1Hz, SR=10Hz', () => {
    const sr = 10.0
    const data = Array.from({ length: 300 }, (_, i) => Math.sin(2 * Math.PI * 0.1 * i / sr))
    const out = bandpassFilter(data, 0.01, 0.5, sr)
    // 끝부분(edge effect 제외) RMS가 원래의 30% 이상 유지
    const tail = out.slice(150)
    const rms = Math.sqrt(tail.reduce((s, v) => s + v * v, 0) / tail.length)
    expect(rms).toBeGreaterThan(0.25)
  })

  it('low_hz <= 0이면 에러', () => {
    expect(() => bandpassFilter([1, 2, 3], 0, 0.5, 10)).toThrow()
  })

  it('high_hz >= Nyquist이면 에러', () => {
    expect(() => bandpassFilter([1, 2, 3], 0.01, 5.0, 10)).toThrow()
  })

  it('숫자 배열 반환', () => {
    const data = Array.from({ length: 100 }, () => Math.random())
    const out = bandpassFilter(data, 0.01, 0.5, 10.0)
    out.forEach(v => expect(typeof v).toBe('number'))
  })
})
