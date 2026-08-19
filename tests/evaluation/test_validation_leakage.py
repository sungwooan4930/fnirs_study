"""T3(누수 검출) — 스펙 8절. 이 서브프로젝트의 존재 이유."""

import json

import pytest

from src.datasets.contract import LeakageError
from src.evaluation.runner import run_experiment

PILOT = "config/experiments/pilot.yaml"

# [2026-08-19 실측치 — 세션 통합(Task 13) 이전] 아래 7.55%p·0.6542·0.7297은
# 단일 세션 시절 측정값이다. Task 13에서 pilot.yaml이 n_sessions=3으로
# 바뀌고 세션 베이스라인 정규화가 실제로 켜졌으므로 절대 수치는 더 이상
# 정확하지 않다 — LOSO 기준값 자체가 tests/baselines/t2_pilot.json에서
# **두 번** 이동해 0.6542 → 0.6283 → **0.5959**(현재 값. 2026-08-19 최종
# 리뷰에서 실측 재확인, git_commit=2b9ec0f)다.
#   - 0.6542 → 0.6283 (commit 1bf95ed, Task 13): 러너에 세션 루프·세션
#     베이스라인 정규화가 처음 end-to-end로 통합되며 이동. n_sessions
#     1→3 구조 변경과 함께 일어나 단일 원인으로 분리되지 않는다.
#   - 0.6283 → 0.5959 (재기록 commit 76c9896, 값 자체는 이 사이에 발생):
#     **§6.2 정규화 정정**(commit 47d1bba, `concentration_delta`를 뺄셈만이
#     아니라 베이스라인 산포로도 나누도록 수정)이 원인이다 — 분류기에
#     들어가는 특징값 자체가 바뀌므로 정확도가 움직인다. 같은 구간에
#     **§6.3 드리프트 분모 정정**(commit 2b9ec0f, `compute_drift`의 분모를
#     `|mean(start)|`→`std(start, ddof=1)`로 수정)도 착지했지만, `drift_flag`/
#     `compute_drift`는 `session_quality.csv` 리포팅에만 쓰이고 분류기가 보는
#     윈도우를 걸러내지 않으므로(§6.3 참조) 이 수치에는 기여하지 않는다 —
#     §6.3을 공동 원인으로 적으면 부정확하다.
# 재측정하지 않은 이유: INFLATION_THRESHOLD
# (0.05)가 지키려는 성질(누수가 있으면 LOSO보다 유의하게 높다)은 재확인됐고
# (test_t3b는 여전히 통과한다), 아래 서술은 문턱을 고른 *논리*를 설명하는
# 것이지 재현해야 하는 값이 아니다. 문턱 자체를 다시 조정할 근거가 생기면
# 그때 재측정해 이 블록 전체를 갱신할 것.
# 파일럿 실측(2026-08-19)에 근거해 5%p로 보정됨 — 원래 스펙의 15%p는
# 데이터 없이 사전에 정한 값이었다. 실제로 측정된 부풀림은 7.55%p
# (LOSO 0.6542 → window_random 누수 0.7297)이다.
#
# 15%p가 아니라 7.55%p인 이유: pilot.yaml은 block_duration_s=60,
# window_s=5·step_s=1이므로 한 블록 안에 약 52개 창이 들어가고 이 창들은
# 전부 같은 라벨을 공유한다. 따라서 "인접 창이 train/test 양쪽에 걸치는"
# 누수의 대부분은 이미 합법적인 신호(블록 수준 판별)가 제공하는 정보를
# 중복할 뿐이다 — 분류기는 어차피 블록 단위로 구분하고 있다. 실제로
# 벌어들이는 7.55%p는 주로 "이웃 창을 알아보는" 것이 아니라 "피험자를
# 알아보는"(subject-identity) 누수다: window_random은 피험자 중첩 가드가
# 걸리는 분할기이므로, 같은 피험자의 창이 train과 test에 흩어져 들어가면
# 분류기가 피험자 고유의 특징(개인차)을 이용해 맞힐 수 있다.
# KFold(shuffle=True)가 5-fold라서 일부 인접 창이 우연히 같은 fold(둘 다
# train 혹은 둘 다 test)에 남는 것도 부풀림을 깎지만, 이는 부차적 효과다.
#
# 결론적으로 이 config는 "창-중첩 누수"를 과소평가한다: 블록이 더 짧거나
# 블록 내에서 라벨이 바뀌는 설계였다면 훨씬 크게 부풀었을 것이다. 즉
# 가드가 지키는 값은 여기서 최소 7.55%p이지 딱 7.55%p가 아니다.
#
# 5%p를 문턱으로 쓰는 이유: 널 테스트(T1, n=6000)의 95% 신뢰구간
# 반폭은 약 1.19%p다 (`tests/evaluation/test_validation_effect.py`의
# test_t2_matches_recorded_baseline docstring에 기록된 값 — 그 값을
# 다시 재는 대신 여기서는 그것을 인용한다). 5%p는 그 노이즈의 약 4배로,
# 우연으로는 설명되지 않는 진짜 부풀림임을 확인하면서도, 관측된
# 7.55%p에 여유를 두어 실행 간 노이즈에 test가 깨지지 않게 한다.
INFLATION_THRESHOLD = 0.05


def _run(tmp_path, name, **overrides):
    ov = {"output": {"results_dir": str(tmp_path / name)}}
    ov.update(overrides)
    return run_experiment(PILOT, overrides=ov)


def _acc(out_dir):
    return json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))["pooled_accuracy"]


@pytest.mark.slow
def test_t3a_guards_block_window_random_split(tmp_path):
    with pytest.raises(LeakageError, match="subjects appear in both"):
        _run(tmp_path, "blocked", evaluation={"splitter": "window_random"})


@pytest.mark.slow
def test_t3b_leakage_actually_inflates_accuracy(tmp_path):
    """창 단위 무작위 분할이 LOSO 대비 정확도를 부풀린다는 것을 확인한다.

    파일럿 실측(2026-08-19): LOSO 0.6542, window_random 누수 0.7297 —
    부풀림 7.55%p. 문턱은 15%p가 아니라 5%p다(위 INFLATION_THRESHOLD 주석
    참조). 60초 블록 안에 5초/1초 슬라이딩 윈도우 창이 약 52개 들어가고
    모두 같은 라벨을 공유하므로, 인접 창 누수의 상당 부분은 이미 합법적인
    블록 수준 신호와 겹친다 — 그래서 15%p까지는 부풀지 않는다. 남은
    7.55%p는 주로 subject-identity 누수(분류기가 이웃 창이 아니라
    피험자 개인의 특징을 알아보는 것)로 해석된다. 이 config는 블록이 길어
    창-중첩 누수 자체는 과소평가하고 있다는 점에 유의할 것 — 블록이
    짧거나 블록 내 라벨이 바뀌는 설계라면 부풀림은 이보다 커진다.
    """
    loso = _acc(_run(tmp_path, "loso"))
    leaky = _acc(
        _run(
            tmp_path, "leaky",
            evaluation={
                "splitter": "window_random",
                "guards": {
                    "check_subject_overlap": False,
                    "check_window_overlap": False,
                    "check_session_overlap": False,
                },
            },
        )
    )
    assert leaky - loso >= INFLATION_THRESHOLD, (
        f"창 단위 무작위 분할이 LOSO({loso:.3f}) 대비 {leaky - loso:.3f}만 "
        f"부풀렸다 ({INFLATION_THRESHOLD:.2f} 미만). 파일럿 실측 기준값은 "
        "7.55%p(0.6542→0.7297)이므로 이보다 크게 벗어나면 재현성이 깨졌거나 "
        "pilot.yaml·window_random·가드 로직이 바뀐 것이다 — 원인을 규명하지 "
        "않고 이 문턱을 더 낮추지 말 것. (참고: 60초 블록 안에 창이 약 52개 "
        "들어가 같은 라벨을 공유하므로 인접-창 누수 대부분은 블록 수준 "
        "신호와 겹친다. 남는 부풀림은 주로 subject-identity 누수다. 이 "
        "config는 블록이 길어 창-중첩 누수를 과소평가하는 쪽으로 치우쳐 "
        "있으므로, 가드가 막는 실제 위험은 여기 측정치보다 크면 컸지 작지 "
        "않다.)"
    )


@pytest.mark.slow
def test_t3_unsafe_run_is_marked_in_directory_name(tmp_path):
    out = _run(
        tmp_path, "marked",
        evaluation={
            "splitter": "window_random",
            "guards": {
                "check_subject_overlap": False,
                "check_window_overlap": False,
                "check_session_overlap": False,
            },
        },
    )
    assert out.name.endswith("-UNSAFE")
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["guards_disabled"] is True


@pytest.mark.slow
def test_t3_testview_fit_is_blocked_regardless_of_guard_config(tmp_path):
    """TestView.fit 차단은 설정으로 끌 수 없는 구조적 장치다."""
    from src.common.config import load_config
    from src.common.seeding import set_all_seeds
    from src.evaluation.runner import build_dataset
    from src.evaluation.splitters import get_splitter
    from sklearn.preprocessing import StandardScaler

    cfg = load_config(PILOT)
    cfg["evaluation"]["guards"] = {
        "check_subject_overlap": False,
        "check_window_overlap": False,
    }
    cfg["simulation"]["n_subjects"] = 3
    cfg["simulation"]["task"]["n_blocks_per_level"] = 1
    cfg["simulation"]["task"]["block_duration_s"] = 20
    cfg["simulation"]["eeg"]["sfreq_hz"] = 100

    dataset, _, _ = build_dataset(cfg, set_all_seeds(0))
    fold = next(iter(dataset.iter_folds(get_splitter("loso"))))

    with pytest.raises(LeakageError):
        fold.test.fit(StandardScaler())
