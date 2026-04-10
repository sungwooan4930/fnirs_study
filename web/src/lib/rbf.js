/**
 * RBF 보간 + Blue-White-Red 컬러맵
 * Python brain_map_widget.py _build_overlay() 포팅
 */

/**
 * Blue(0) → White(0.5) → Red(1) 컬러맵
 * @param {number} t  [0, 1]
 * @returns {[number, number, number]} [r, g, b] 0-255
 */
export function bwr(t) {
  t = Math.max(0, Math.min(1, t))
  let r, g, b
  if (t < 0.5) {
    const s = t / 0.5
    r = Math.round(s * 255)
    g = Math.round(s * 255)
    b = 255
  } else {
    const s = (t - 0.5) / 0.5
    r = 255
    g = Math.round((1 - s) * 255)
    b = Math.round((1 - s) * 255)
  }
  return [r, g, b]
}

/**
 * Gaussian RBF 보간 (epsilon=6)
 * @param {number[][]} queryPoints  [[x,y], ...]
 * @param {number[][]} dataPoints   [[x,y], ...] 채널 위치들
 * @param {number[]}   values       채널별 값
 * @returns {number[]}
 */
export function rbfInterpolate(queryPoints, dataPoints, values) {
  const epsilon = 6
  const n = dataPoints.length

  function rbfKernel(r2) {
    return Math.exp(-epsilon * epsilon * r2)
  }

  // RBF 행렬 Phi: n × n
  const Phi = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      const dx = dataPoints[i][0] - dataPoints[j][0]
      const dy = dataPoints[i][1] - dataPoints[j][1]
      return rbfKernel(dx * dx + dy * dy)
    })
  )

  // 가중치 w = Phi^{-1} · values (가우스 소거)
  const w = gaussianElimination(Phi, [...values])

  // 쿼리 포인트에서 보간
  return queryPoints.map(([qx, qy]) => {
    let val = 0
    for (let j = 0; j < n; j++) {
      const dx = qx - dataPoints[j][0]
      const dy = qy - dataPoints[j][1]
      val += w[j] * rbfKernel(dx * dx + dy * dy)
    }
    return val
  })
}

/** 부분 피벗 가우스 소거 — Ax = b 풀기 */
function gaussianElimination(A, b) {
  const n = b.length
  const M = A.map((row, i) => [...row, b[i]])

  for (let col = 0; col < n; col++) {
    let maxRow = col
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row
    }
    ;[M[col], M[maxRow]] = [M[maxRow], M[col]]

    const pivot = M[col][col]
    if (Math.abs(pivot) < 1e-12) continue

    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / pivot
      for (let k = col; k <= n; k++) {
        M[row][k] -= factor * M[col][k]
      }
    }
  }

  const x = Array(n).fill(0)
  for (let i = n - 1; i >= 0; i--) {
    x[i] = M[i][n]
    for (let j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j]
    x[i] /= M[i][i]
  }
  return x
}
