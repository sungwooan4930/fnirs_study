import { useEffect, useRef, useCallback } from 'react'

/**
 * Web Worker를 생성하고 processedSample 콜백을 연결하는 훅.
 * @param {(sample: object) => void} onSample
 * @returns {{ sendPacket: (packet: object) => void, resetWorker: () => void }}
 */
export function usePipeline(onSample) {
  const workerRef = useRef(null)

  useEffect(() => {
    const worker = new Worker(
      new URL('../workers/pipeline.worker.js', import.meta.url),
      { type: 'module' }
    )
    worker.onmessage = ({ data: msg }) => {
      if (msg.type === 'sample') onSample(msg.data)
    }
    workerRef.current = worker
    return () => worker.terminate()
  }, [onSample])

  const sendPacket = useCallback((packet) => {
    workerRef.current?.postMessage({ type: 'packet', data: packet })
  }, [])

  const resetWorker = useCallback(() => {
    workerRef.current?.postMessage({ type: 'reset' })
  }, [])

  return { sendPacket, resetWorker }
}
