/**
 * BrainViews: PNG 뷰별 채널 좌표 정의
 *
 * x, y: 이미지 위 퍼센트 위치 (0~100)
 * r:    radialGradient 반경 (px 기준 %)
 *
 * 전전두엽(PFC) 4채널:
 *   Ch1 좌전방 PFC  Ch2 우전방 PFC
 *   Ch3 좌후방 PFC  Ch4 우후방 PFC
 */

export const BRAIN_FRONT = {
  // front-view 이미지 파일명 (web/public/ 아래)
  image: '/brain.png',

  channels: [
    { id: 'ch1', label: '좌전방 PFC', x: 36, y: 52, r: 18 },
    { id: 'ch2', label: '우전방 PFC', x: 64, y: 52, r: 18 },
    { id: 'ch3', label: '좌후방 PFC', x: 38, y: 65, r: 15 },
    { id: 'ch4', label: '우후방 PFC', x: 62, y: 65, r: 15 },
  ],
}
