/**
 * generatePDF — jsPDF + html2canvas로 A4 포맷 레포트 PDF 생성
 *
 * @param {HTMLElement} el  캡처할 DOM 엘리먼트 (ReportPDFTemplate)
 * @param {string}      filename  저장 파일명
 */
import jsPDF from 'jspdf'
import html2canvas from 'html2canvas'

const A4_W_MM = 210
const A4_H_MM = 297
const DPI     = 150          // 출력 해상도 (높을수록 선명, 느려짐)
const SCALE   = DPI / 96     // 브라우저 96dpi 기준 배율

export async function generatePDF(el, filename = 'fnirs_report.pdf') {
  // 캡처 전 잠깐 보이게
  const prev = el.style.cssText
  el.style.cssText = 'position:fixed;top:-9999px;left:0;z-index:-1;visibility:visible;'

  let canvas
  try {
    canvas = await html2canvas(el, {
      scale: SCALE,
      useCORS: true,
      allowTaint: true,
      backgroundColor: '#ffffff',
      logging: false,
    })
  } finally {
    el.style.cssText = prev
  }

  const imgData = canvas.toDataURL('image/jpeg', 0.92)

  // 캔버스 비율에 맞춰 PDF 높이 결정 (여러 페이지 지원)
  const canvasW = canvas.width
  const canvasH = canvas.height
  const pxPerMm = canvasW / A4_W_MM

  const totalMmH = canvasH / pxPerMm
  const pageCount = Math.ceil(totalMmH / A4_H_MM)

  const pdf = new jsPDF({ unit: 'mm', format: 'a4', orientation: 'portrait' })

  for (let p = 0; p < pageCount; p++) {
    if (p > 0) pdf.addPage()

    const srcY = p * A4_H_MM * pxPerMm
    const srcH = Math.min(A4_H_MM * pxPerMm, canvasH - srcY)

    // 해당 페이지 슬라이스를 임시 캔버스에 그림
    const sliceCanvas = document.createElement('canvas')
    sliceCanvas.width  = canvasW
    sliceCanvas.height = srcH
    sliceCanvas.getContext('2d').drawImage(canvas, 0, srcY, canvasW, srcH, 0, 0, canvasW, srcH)

    const sliceImg  = sliceCanvas.toDataURL('image/jpeg', 0.92)
    const sliceHMm  = srcH / pxPerMm
    pdf.addImage(sliceImg, 'JPEG', 0, 0, A4_W_MM, sliceHMm)
  }

  pdf.save(filename)
}
