/**
 * Modified Beer-Lambert Law — JS port of src/processing/mbll.py
 *
 * @param {number[][][]} raw  [n_wavelengths][n_channels][n_samples]
 * @param {number[]} extHbo   HbO extinction coefficients (L/mmol·cm), one per wavelength
 * @param {number[]} extHbr   HbR extinction coefficients (L/mmol·cm), one per wavelength
 * @param {number[]} dpf      Differential path length factors, one per wavelength
 * @param {number}   sdsMm    Source-detector separation in mm
 * @returns {{ hbo: number[][], hbr: number[][] }}  each [n_channels][n_samples] in μmol/L
 */
export function modifiedBeerLambert(raw, extHbo, extHbr, dpf, sdsMm) {
  if (sdsMm <= 0) throw new Error('sds_mm must be positive')

  const nWl = raw.length
  const nCh = raw[0].length
  const nSamples = raw[0][0].length
  const sdsCm = sdsMm / 10.0

  // Extinction matrix A: nWl × 2
  const A = Array.from({ length: nWl }, (_, i) => [extHbo[i], extHbr[i]])

  // Pseudo-inverse: Apinv = (A^T A)^{-1} A^T  (2 × nWl)
  const Apinv = pseudoInverse(A, nWl)

  const hbo = Array.from({ length: nCh }, () => Array(nSamples).fill(0))
  const hbr = Array.from({ length: nCh }, () => Array(nSamples).fill(0))

  for (let ch = 0; ch < nCh; ch++) {
    // Baseline: first sample per wavelength (guard against 0)
    const baseline = Array.from({ length: nWl }, (_, wl) => {
      const v = raw[wl][ch][0]
      return v === 0 ? 1e-10 : v
    })

    for (let s = 0; s < nSamples; s++) {
      // od_normalized[wl] = -log(I/I0) / (dpf[wl] * sdsCm)
      const odNorm = Array.from({ length: nWl }, (_, wl) => {
        const I = raw[wl][ch][s] === 0 ? 1e-10 : raw[wl][ch][s]
        return -Math.log(I / baseline[wl]) / (dpf[wl] * sdsCm)
      })

      // concentrations = Apinv (2×nWl) · odNorm (nWl) = [cHbo, cHbr]
      let cHbo = 0, cHbr = 0
      for (let wl = 0; wl < nWl; wl++) {
        cHbo += Apinv[0][wl] * odNorm[wl]
        cHbr += Apinv[1][wl] * odNorm[wl]
      }
      // mmol/L → μmol/L
      hbo[ch][s] = cHbo * 1000.0
      hbr[ch][s] = cHbr * 1000.0
    }
  }

  return { hbo, hbr }
}

/** Pseudo-inverse of A (nWl×2) via (A^T A)^{-1} A^T → returns 2×nWl */
function pseudoInverse(A, nWl) {
  // AT: 2 × nWl
  const AT = [A.map(r => r[0]), A.map(r => r[1])]

  // ATA: 2 × 2
  const ATA = [[0, 0], [0, 0]]
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 2; j++)
      for (let k = 0; k < nWl; k++)
        ATA[i][j] += AT[i][k] * A[k][j]

  // Inverse of 2×2: [[d,-b],[-c,a]] / det
  const det = ATA[0][0] * ATA[1][1] - ATA[0][1] * ATA[1][0]
  if (Math.abs(det) < 1e-12) throw new Error('Singular extinction matrix')
  const ATAinv = [
    [ ATA[1][1] / det, -ATA[0][1] / det],
    [-ATA[1][0] / det,  ATA[0][0] / det],
  ]

  // pinv = ATAinv (2×2) · AT (2×nWl) → 2×nWl
  const pinv = Array.from({ length: 2 }, () => Array(nWl).fill(0))
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < nWl; j++)
      for (let k = 0; k < 2; k++)
        pinv[i][j] += ATAinv[i][k] * AT[k][j]

  return pinv
}
