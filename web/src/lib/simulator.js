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
const HBO_AMPLITUDE = 0.5
const HBO_FREQ_HZ = 0.1
const NOISE_STD = 0.05

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

      const channelIntensities = []
      for (let ch = 0; ch < N_CHANNELS; ch++) {
        const phaseOffset = ch * (2 * Math.PI / N_CHANNELS)
        const hboSignal = HBO_AMPLITUDE * Math.sin(2 * Math.PI * HBO_FREQ_HZ * t + phaseOffset)

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
