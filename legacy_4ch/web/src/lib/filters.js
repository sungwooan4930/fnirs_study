/**
 * Butterworth 밴드패스 필터 — fili 라이브러리 사용
 * zero-phase: forward pass + backward pass (Python sosfiltfilt 동가)
 */
import Fili from 'fili'

/**
 * @param {number[]} data          1D 배열
 * @param {number}   lowHz         하한 주파수 (Hz)
 * @param {number}   highHz        상한 주파수 (Hz)
 * @param {number}   samplingRate  샘플링 레이트 (Hz)
 * @param {number}   order         필터 차수 (기본 3 — forward+backward로 effective order 6)
 * @returns {number[]}
 */
export function bandpassFilter(data, lowHz, highHz, samplingRate, order = 3) {
  if (lowHz <= 0) throw new Error('low_hz must be positive')
  const nyquist = samplingRate / 2
  if (highHz >= nyquist) throw new Error(`high_hz must be less than Nyquist (${nyquist})`)
  if (highHz <= lowHz) throw new Error('high_hz must be greater than low_hz')

  const iirCalc = new Fili.CalcCascades()
  const Fc = (lowHz + highHz) / 2        // 중심 주파수
  const BW = Math.log2(highHz / lowHz)   // 대역폭 (옥타브 단위 — fili BW 파라미터 규격)

  const coeffs = iirCalc.bandpass({
    order,
    characteristic: 'butterworth',
    Fs: samplingRate,
    Fc,
    BW,
  })

  // Forward pass
  const fwdFilter = new Fili.IirFilter(coeffs)
  const forward = fwdFilter.multiStep(data)

  // Backward pass (zero-phase)
  const bwdFilter = new Fili.IirFilter(coeffs)
  const backward = bwdFilter.multiStep([...forward].reverse())

  return backward.reverse()
}
