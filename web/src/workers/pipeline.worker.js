/**
 * Web Worker: 슬라이딩 윈도우 + 필터 + mBLL → processedSample 반환
 *
 * 메시지 수신:
 *   { type: 'packet', data: rawPacket }
 *   { type: 'reset' }
 *
 * 메시지 발신:
 *   { type: 'sample', data: processedSample }
 */
import { bandpassFilter } from '../lib/filters.js'
import { modifiedBeerLambert } from '../lib/mbll.js'

const WINDOW_SEC = 10.0
const SAMPLING_RATE_HZ = 10.0
const MIN_SAMPLES = 40  // sosfiltfilt padlen 대응

// 설정 (config/settings.yaml 기준)
const EXT_HBO = [0.975, 0.901, 1.046]
const EXT_HBR = [2.755, 0.781, 0.260]
const DPF = [6.51, 5.86, 5.12]
const SDS_MM = 30.0
const N_CHANNELS = 4
const N_WAVELENGTHS = 3
const BANDPASS_LOW = 0.01
const BANDPASS_HIGH = 0.5

const maxSamples = Math.floor(WINDOW_SEC * SAMPLING_RATE_HZ)
let window = []  // rawPacket[]

self.onmessage = ({ data: msg }) => {
  if (msg.type === 'reset') {
    window = []
    return
  }

  if (msg.type !== 'packet') return

  const packet = msg.data
  window.push(packet)
  if (window.length > maxSamples) window.shift()
  if (window.length < MIN_SAMPLES) return

  const sample = processWindow(window)
  self.postMessage({ type: 'sample', data: sample })
}

function processWindow(packets) {
  const nT = packets.length

  // raw: [n_wavelengths][n_channels][n_samples]
  const raw = Array.from({ length: N_WAVELENGTHS }, () =>
    Array.from({ length: N_CHANNELS }, () => Array(nT).fill(0))
  )

  for (let t = 0; t < nT; t++) {
    const pkt = packets[t]
    for (let ch = 0; ch < N_CHANNELS; ch++) {
      for (let wl = 0; wl < N_WAVELENGTHS; wl++) {
        raw[wl][ch][t] = pkt.channelIntensities[ch * N_WAVELENGTHS + wl]
      }
    }
  }

  // 밴드패스 필터 (파장별, 평균 복원 포함)
  for (let wl = 0; wl < N_WAVELENGTHS; wl++) {
    for (let ch = 0; ch < N_CHANNELS; ch++) {
      const slice = raw[wl][ch]
      const mean = slice.reduce((s, v) => s + v, 0) / slice.length
      const filtered = bandpassFilter(slice, BANDPASS_LOW, BANDPASS_HIGH, SAMPLING_RATE_HZ)
      raw[wl][ch] = filtered.map(v => v + mean)
    }
  }

  // mBLL
  const { hbo, hbr } = modifiedBeerLambert(raw, EXT_HBO, EXT_HBR, DPF, SDS_MM)

  // 마지막 샘플의 값
  const hboNow = hbo.map(ch => ch[ch.length - 1])
  const hbrNow = hbr.map(ch => ch[ch.length - 1])

  // 집중도 지수: HbO 평균을 [0,1]로 정규화
  const ci = Math.max(0, Math.min(1, (hboNow.reduce((s, v) => s + v, 0) / N_CHANNELS) / 10.0 + 0.5))

  return {
    timestamp: packets[packets.length - 1].timestamp,
    hbo: hboNow,
    hbr: hbrNow,
    ci,
  }
}
