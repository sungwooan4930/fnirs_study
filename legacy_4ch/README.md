# legacy_4ch — 폐기된 이전 프로젝트 (참조 전용)

**2026-08-18 아카이브.** 이 디렉토리의 코드는 **현재 연구과제와 다른 프로젝트**다.

## 무엇이었나

4채널 커스텀 fNIRS 장치로 전전두엽 신호를 측정해 1차원 집중도 지수를 실시간 표시하는
측정기 소프트웨어. Python Core(PySide6) → React 웹앱으로 전환된 상태에서 중단.

- `src/` — Python Core (mBLL, Butterworth 필터, HDF5 저장, PySide6 UI). 73 tests
- `web/` — React + Vite 웹앱 (Web Bluetooth, Web Worker 신호처리). 25 tests
- `tests/` — Python 단위 테스트
- `config/settings.yaml` — 4채널·3파장(780/850/950nm) 하드웨어 설정
- `refer/` — 구 UI 디자인 참조 이미지
- `lovable-*.md` — 구 웹 UI 디자인 시스템 문서

## 왜 폐기됐나

연구계획서(`docs/plan/`)와 대조한 결과 사실상 다른 프로젝트임이 확인됐다.

| | legacy_4ch | 현 연구과제 |
|---|---|---|
| fNIRS | 4ch, 3파장, 커스텀 BLE | 48ch, 전두·두정·운동·후두, 10.4 Hz |
| EEG | 없음 | 30ch, 10-5, 1,000 Hz |
| 동기화/표준 | 자체 타임스탬프 / 자체 HDF5 | LSL / BIDS |
| 출력 | 집중도 1차원 | 6차원 인지상태 벡터 |
| 모델 | 규칙식 | 융합 3전략 + 크로스모달 어텐션, LOSO-CV |
| 사용자 | 측정 대상자 본인 | 교수자 (다수 학습자 모니터링) |

## 재사용 가능한 것 — 오직 이것뿐

- `src/processing/mbll.py` — 수정 Beer-Lambert 변환 **로직 참조**
  (단, 신규 파이프라인은 MNE-NIRS의 `beer_lambert_law`를 쓴다)
- `src/processing/filters.py` — Butterworth 대역통과 구현 참조
  (특히 `sosfiltfilt` padlen 대응 노하우)

## 절대 하지 말 것

- ❌ 4채널·3파장(780/850/950nm)·BLE 스펙을 신규 코드에 도입
- ❌ 자체 HDF5 스키마를 BIDS 대신 사용
- ❌ `SimpleHbOIndex` 같은 1차원 집중도 지수를 6차원 벡터 대용으로 사용
- ❌ 이 디렉토리의 코드를 신규 모듈에서 import

상세 배경: `../run_logging.md` 2026-08-18 항목, `../CLAUDE.md` §1.
