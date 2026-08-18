import { useCallback, useRef } from 'react'
import { FNIRSSimulator } from '../lib/simulator.js'

const RECONNECT_MAX = 3

/**
 * Web Bluetooth BLE 연결 훅
 * URL에 ?simulate=true 가 있으면 시뮬레이터 사용
 *
 * @param {(packet: object) => void} onPacket      패킷 수신 콜백
 * @param {(status: string) => void} onStatusChange 연결 상태 변경 콜백
 */
export function useBLE(onPacket, onStatusChange) {
  const deviceRef = useRef(null)
  const charRef = useRef(null)
  const simRef = useRef(null)
  const reconnectCount = useRef(0)

  const isSimulate = new URLSearchParams(window.location.search).get('simulate') === 'true'

  const connect = useCallback(async () => {
    if (isSimulate) {
      onStatusChange('connected')
      const sim = new FNIRSSimulator()
      simRef.current = sim
      sim.start(onPacket)
      return
    }

    try {
      onStatusChange('connecting')
      const device = await navigator.bluetooth.requestDevice({
        acceptAllDevices: true,
        optionalServices: ['generic_access'],
      })
      deviceRef.current = device

      device.addEventListener('gattserverdisconnected', () => {
        onStatusChange('error')
        if (reconnectCount.current < RECONNECT_MAX) {
          reconnectCount.current++
          setTimeout(() => connect(), 1000)
        }
      })

      const server = await device.gatt.connect()
      // NOTE: 실제 UUID는 하드웨어 팀 확정 후 교체 필요
      // 현재는 연결 성공만 확인
      reconnectCount.current = 0
      onStatusChange('connected')

      // TODO: 실제 UUID 확정 후 characteristic 구독 추가
      // const service = await server.getPrimaryService('YOUR-SERVICE-UUID')
      // const char = await service.getCharacteristic('YOUR-CHAR-UUID')
      // await char.startNotifications()
      // char.addEventListener('characteristicvaluechanged', (e) => {
      //   const raw = parsePacket(e.target.value)
      //   onPacket(raw)
      // })
    } catch (err) {
      console.error('BLE 연결 실패:', err)
      onStatusChange('error')
    }
  }, [onPacket, onStatusChange, isSimulate])

  const disconnect = useCallback(() => {
    simRef.current?.stop()
    simRef.current = null
    if (deviceRef.current?.gatt?.connected) {
      deviceRef.current.gatt.disconnect()
    }
    deviceRef.current = null
    onStatusChange('idle')
  }, [onStatusChange])

  return { connect, disconnect, isSimulate }
}
