"""특징 추출기 레지스트리.

`features.extractor`는 config 스키마에 있고 모든 config가 값을 적고
있었지만 `src/` 어디에서도 읽지 않았다. 그 상태에서는 B(전처리)가
완성되어 누군가 `extractor: real`로 바꿔도 최소 추출기가 조용히 돌고
초록불이 뜬다 — 엄격 스키마가 막으려던 "조용한 기본값" 실패가 스키마
안쪽에서 그대로 재현되는 셈이다. 그래서 키를 실제로 소비한다.
"""

from __future__ import annotations

from collections.abc import Callable

from src.datasets.features_minimal import extract_features as _minimal

#: 이름 → 추출기 함수. 시그니처는 (rec, windows) -> dict[str, np.ndarray].
EXTRACTORS: dict[str, Callable] = {
    "minimal": _minimal,
}


def get_extractor(name: str) -> Callable:
    """이름으로 특징 추출기를 고른다. 모르는 이름이면 실행을 거부한다."""
    try:
        return EXTRACTORS[name]
    except KeyError:
        raise ValueError(
            f"unknown feature extractor '{name}'; expected one of "
            f"{sorted(EXTRACTORS)}. B(전처리) 완성 전까지 'minimal'만 유효하다"
        ) from None
