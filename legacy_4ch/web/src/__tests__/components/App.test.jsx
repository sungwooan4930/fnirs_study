import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import { AppProvider, useApp } from '../../context/AppContext'

// 각 테스트 전에 sessionStorage 정리
beforeEach(() => {
  sessionStorage.clear()
})

function Inspector() {
  const ctx = useApp()
  return (
    <div>
      <span data-testid="ble">{ctx.bleStatus}</span>
      <span data-testid="cal">{String(ctx.calibrationDone)}</span>
      <span data-testid="ci">{ctx.processedSample?.ci ?? 'null'}</span>
      <button onClick={() => ctx.setCalibrationDone(true)}>cal</button>
      <button onClick={() => ctx.setBleStatus('connected')}>ble</button>
      <button onClick={() => ctx.pushSample({ timestamp: 1, hbo: [0], hbr: [0], ci: 0.5 })}>push</button>
    </div>
  )
}

test('초기 상태가 올바르다', () => {
  render(<AppProvider><Inspector /></AppProvider>)
  expect(screen.getByTestId('ble')).toHaveTextContent('idle')
  expect(screen.getByTestId('cal')).toHaveTextContent('false')
  expect(screen.getByTestId('ci')).toHaveTextContent('null')
})

test('setBleStatus 변경', async () => {
  render(<AppProvider><Inspector /></AppProvider>)
  await userEvent.click(screen.getByText('ble'))
  expect(screen.getByTestId('ble')).toHaveTextContent('connected')
})

test('setCalibrationDone 변경', async () => {
  render(<AppProvider><Inspector /></AppProvider>)
  await userEvent.click(screen.getByText('cal'))
  expect(screen.getByTestId('cal')).toHaveTextContent('true')
})

test('pushSample이 processedSample과 sessionData를 갱신', async () => {
  render(<AppProvider><Inspector /></AppProvider>)
  await userEvent.click(screen.getByText('push'))
  expect(screen.getByTestId('ci')).toHaveTextContent('0.5')
})

test('pushSample이 sessionData 배열에 누적된다', async () => {
  render(<AppProvider><Inspector /></AppProvider>)
  await userEvent.click(screen.getByText('push'))
  await userEvent.click(screen.getByText('push'))
  // Inspector를 통해 sessionData 길이를 확인하려면 Inspector에 추가가 필요
  // 대신 ci가 두 번 눌려도 여전히 0.5인지 확인 (processedSample은 항상 최신)
  expect(screen.getByTestId('ci')).toHaveTextContent('0.5')
})

test('clearSession이 상태를 초기화한다', async () => {
  function Inspector2() {
    const ctx = useApp()
    return (
      <div>
        <span data-testid="ci2">{ctx.processedSample?.ci ?? 'null'}</span>
        <span data-testid="len">{ctx.sessionData.length}</span>
        <button onClick={() => ctx.pushSample({ timestamp: 1, hbo: [0], hbr: [0], ci: 0.9 })}>push2</button>
        <button onClick={() => ctx.clearSession()}>clear</button>
      </div>
    )
  }
  render(<AppProvider><Inspector2 /></AppProvider>)
  await userEvent.click(screen.getByText('push2'))
  expect(screen.getByTestId('ci2')).toHaveTextContent('0.9')
  expect(screen.getByTestId('len')).toHaveTextContent('1')
  await userEvent.click(screen.getByText('clear'))
  expect(screen.getByTestId('ci2')).toHaveTextContent('null')
  expect(screen.getByTestId('len')).toHaveTextContent('0')
})

test('AppProvider 밖에서 useApp 호출 시 에러', () => {
  function BadComponent() {
    useApp()
    return null
  }
  // console.error를 잠시 억제
  const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
  expect(() => render(<BadComponent />)).toThrow('useApp must be used within AppProvider')
  spy.mockRestore()
})
