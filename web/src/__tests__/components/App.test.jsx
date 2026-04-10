import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AppProvider, useApp } from '../../context/AppContext'

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
