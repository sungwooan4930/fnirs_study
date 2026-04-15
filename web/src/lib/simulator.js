/**
 * 개발용 fNIRS BLE 시뮬레이터
 * Python FNIRSSimulator와 동일한 sin파+노이즈 패턴
 *
 * 반환 rawPacket 형식:
 * {
 *   timestamp: number,
 *   channelIntensities: number[],  // [ch0_wl0, ch0_wl1, ch0_wl2, ch1_wl0, ...]
 *   nWavelengths: 3,
 * }
 */

const N_CHANNELS = 4
const N_WAVELENGTHS = 3
const SAMPLING_RATE_HZ = 10.0
const HBO_AMPLITUDE = 0.05
const HBO_FREQ_HZ = 0.1
const NOISE_STD = 0.05

// 집중도 점진적 상승 설정
// 신호에 천천히 증가하는 트렌드를 더해 CI가 낮은 값에서 시작해 점차 올라가도록 함
const TREND_MAX = 0.07      // 최대 트렌드 크기 (강도 단위)
const TREND_TAU = 70        // 63% 수렴 시간 (초) — 약 70초에 절반 이상 오름
const FLUCT_AMP  = 0.022    // 중간 변동 진폭
const FLUCT_FREQ = 0.018    // 변동 주기 (~55초)

function gaussianNoise(std) {
  // Box-Muller transform
  const u1 = Math.random()
  const u2 = Math.random()
  return std * Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2)
}

export class FNIRSSimulator {
  constructor() {
    this._intervalId = null
    this._sampleIndex = 0
    this._startTime = null
  }

  /**
   * 시뮬레이션 시작
   * @param {(packet: object) => void} onPacket  패킷 수신 콜백
   */
  start(onPacket) {
    this._sampleIndex = 0
    this._startTime = performance.now()

    const intervalMs = 1000 / SAMPLING_RATE_HZ
    this._intervalId = setInterval(() => {
      const t = this._sampleIndex / SAMPLING_RATE_HZ
      this._sampleIndex++

      // 천천히 상승하는 트렌드 + 중간 변동
      const trend = TREND_MAX * (1 - Math.exp(-t / TREND_TAU))
      const fluct = FLUCT_AMP * Math.sin(2 * Math.PI * FLUCT_FREQ * t)

      const channelIntensities = []
      for (let ch = 0; ch < N_CHANNELS; ch++) {
        const phaseOffset = ch * (2 * Math.PI / N_CHANNELS)
        const freqOffset = [1.0, 1.3, 0.8, 1.5][ch]
        const chFluct = FLUCT_AMP * 0.6 * Math.sin(2 * Math.PI * FLUCT_FREQ * 1.3 * t + ch * 1.7)
        const hboSignal = HBO_AMPLITUDE * Math.sin(2 * Math.PI * HBO_FREQ_HZ * freqOffset * t + phaseOffset)
                        + trend + fluct + chFluct

        for (let wl = 0; wl < N_WAVELENGTHS; wl++) {
          const hboSensitivity = 0.5 + wl * 0.3
          const val = 1.0 + hboSensitivity * hboSignal + gaussianNoise(NOISE_STD)
          channelIntensities.push(Math.max(0.01, val))
        }
      }

      onPacket({
        timestamp: performance.now() / 1000,
        channelIntensities,
        nWavelengths: N_WAVELENGTHS,
      })
    }, intervalMs)
  }

  stop() {
    if (this._intervalId !== null) {
      clearInterval(this._intervalId)
      this._intervalId = null
    }
  }
}
