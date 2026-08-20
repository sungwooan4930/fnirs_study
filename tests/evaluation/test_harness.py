import numpy as np
import pytest

from src.datasets.contract import LeakageError, WindowedDataset
from src.evaluation.harness import make_model, run_folds
from src.evaluation.splitters import get_splitter

GUARDS_ON = {
    "check_subject_overlap": True,
    "check_window_overlap": True,
    "check_session_overlap": True,
    "check_normalization_source": True,
}
GUARDS_OFF = {
    "check_subject_overlap": False,
    "check_window_overlap": False,
    "check_session_overlap": False,
    "check_normalization_source": False,
}


def make_dataset(n_subjects=4, n_per=30, separable=True, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys, subs, times, trials = [], [], [], [], []
    for s in range(n_subjects):
        for i in range(n_per):
            level = i % 3
            offset = level * 3.0 if separable else 0.0
            xs.append(rng.normal(offset, 1.0, size=4))
            ys.append(level)
            subs.append(f"sub-{s + 1:02d}")
            times.append([float(i), float(i) + 5.0])
            trials.append(s * 1000 + i // 10)
    n = n_subjects * n_per
    return WindowedDataset(
        X={"eeg": np.array(xs)},
        y={"cognitive_load": np.array(ys)},
        subject_ids=np.array(subs),
        session_ids=np.zeros(n, dtype=int),
        window_times=np.array(times, dtype=float),
        trial_ids=np.array(trials),
        normalization_source=np.zeros(n, dtype=bool),
    )


def _dataset_with_two_sessions():
    n = 4
    # y=[0,1,1,0] (브리프 원안의 [0,1,0,1]이 아니다): 세션 가드 테스트는
    # train=[0,2]/test=[1,3]을 그대로 run_folds에 태운다. [0,1,0,1]이면
    # train={y[0],y[2]}={0,0}으로 단일 클래스가 되어 run_folds의 (가드와
    # 무관한) 단일 클래스 거부가 먼저 터진다 — 세션 가드를 시험하기도
    # 전에 다른 이유로 실패한다. [0,1,1,0]은 train={0,1}, test={1,0}으로
    # 두 fold 모두 두 클래스를 유지해 세션 가드만을 격리해서 시험한다.
    return WindowedDataset(
        X={"m": np.array([[0.0], [1.0], [0.1], [1.1]])},
        y={"t": np.array([0, 1, 1, 0])},
        subject_ids=np.array(["a", "a", "a", "a"]),
        session_ids=np.array([0, 0, 1, 1]),
        window_times=np.column_stack([np.arange(n, dtype=float) * 10,
                                      np.arange(n, dtype=float) * 10 + 5]),
        trial_ids=np.array([0, 1, 2, 3]),
        normalization_source=np.zeros(n, dtype=bool),
    )


def test_returns_one_result_per_fold():
    ds = make_dataset()
    results = run_folds(
        ds, get_splitter("loso"), target="cognitive_load",
        modalities=["eeg"], guards=GUARDS_ON, seed=0,
    )
    assert len(results) == 4


def test_result_records_test_subjects_and_sizes():
    ds = make_dataset()
    r = run_folds(
        ds, get_splitter("loso"), target="cognitive_load",
        modalities=["eeg"], guards=GUARDS_ON, seed=0,
    )[0]
    assert len(r.test_subjects) == 1
    assert r.n_train + r.n_test == ds.n_windows
    assert len(r.y_true) == len(r.y_pred) == r.n_test


def test_separable_data_beats_chance():
    ds = make_dataset(separable=True)
    results = run_folds(
        ds, get_splitter("loso"), target="cognitive_load",
        modalities=["eeg"], guards=GUARDS_ON, seed=0,
    )
    assert np.mean([r.accuracy for r in results]) > 0.8


def test_non_separable_data_is_near_chance():
    ds = make_dataset(separable=False)
    results = run_folds(
        ds, get_splitter("loso"), target="cognitive_load",
        modalities=["eeg"], guards=GUARDS_ON, seed=0,
    )
    assert np.mean([r.accuracy for r in results]) < 0.5


def test_guards_on_reject_window_random_splitter():
    ds = make_dataset()
    with pytest.raises(LeakageError):
        run_folds(
            ds, get_splitter("window_random", seed=0), target="cognitive_load",
            modalities=["eeg"], guards=GUARDS_ON, seed=0,
        )


def test_guards_off_allow_window_random_splitter():
    ds = make_dataset()
    results = run_folds(
        ds, get_splitter("window_random", seed=0), target="cognitive_load",
        modalities=["eeg"], guards=GUARDS_OFF, seed=0,
    )
    assert len(results) == 5


def test_make_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        make_model("bogus", seed=0)


def test_single_class_fold_is_rejected():
    ds = WindowedDataset(
        X={"eeg": np.random.default_rng(0).normal(size=(20, 3))},
        y={"cognitive_load": np.array([0] * 10 + [1] * 10)},
        subject_ids=np.array(["sub-01"] * 10 + ["sub-02"] * 10),
        session_ids=np.zeros(20, dtype=int),
        window_times=np.column_stack(
            [np.arange(20, dtype=float), np.arange(20, dtype=float) + 5.0]
        ),
        trial_ids=np.arange(20) // 5,
        normalization_source=np.zeros(20, dtype=bool),
    )
    with pytest.raises(ValueError, match="single class"):
        run_folds(
            ds, get_splitter("loso"), target="cognitive_load",
            modalities=["eeg"], guards=GUARDS_ON, seed=0,
        )


def test_single_class_test_fold_is_rejected():
    """test가 단일 클래스인 fold도 거부한다 (스펙 §9는 'fold'라고 적혀 있다).

    sub-01·sub-02는 두 클래스를 갖고 sub-03은 클래스 0만 갖는다. LOSO에서
    sub-03이 test인 fold는 train이 멀쩡하므로 train 쪽 검사만으로는 통과하고,
    그 fold의 정확도는 "다수 클래스를 얼마나 맞혔는가"가 되어 평균과
    accuracy_worst를 동시에 오염시킨다.
    """
    y = np.array([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5 + [0] * 10)
    ds = WindowedDataset(
        X={"eeg": np.random.default_rng(0).normal(size=(30, 3))},
        y={"cognitive_load": y},
        subject_ids=np.array(["sub-01"] * 10 + ["sub-02"] * 10 + ["sub-03"] * 10),
        session_ids=np.zeros(30, dtype=int),
        window_times=np.column_stack(
            [np.arange(30, dtype=float), np.arange(30, dtype=float) + 5.0]
        ),
        trial_ids=np.arange(30) // 5,
        normalization_source=np.zeros(30, dtype=bool),
    )
    with pytest.raises(ValueError, match="test has a single class"):
        run_folds(
            ds, get_splitter("loso"), target="cognitive_load",
            modalities=["eeg"], guards=GUARDS_ON, seed=0,
        )


def make_block_dataset(n_subjects=4, n_trials=3, n_per_trial=10, seed=0):
    """Dataset whose trial blocks are separated by a wide time gap.

    WithinSubjectSplitter splits by trial, not by window, so it makes
    no promise that windows are far apart *within* a trial. But a real
    experiment's trial blocks are separated in time (different task
    blocks), so windows never straddle a trial boundary. This fixture
    encodes that gap explicitly (100s between blocks vs. a 5s window),
    unlike `make_dataset`, whose trials are time-contiguous and would
    make even a correct within-subject split fail the window-overlap
    guard on boundary windows — that would be a real leak, not a false
    positive, so it belongs in a separate fixture.
    """
    rng = np.random.default_rng(seed)
    xs, ys, subs, times, trials = [], [], [], [], []
    for s in range(n_subjects):
        for t in range(n_trials):
            base = t * 100.0
            for i in range(n_per_trial):
                level = i % 3
                xs.append(rng.normal(level * 3.0, 1.0, size=4))
                ys.append(level)
                subs.append(f"sub-{s + 1:02d}")
                times.append([base + i, base + i + 5.0])
                trials.append(s * 1000 + t)
    n = n_subjects * n_trials * n_per_trial
    return WindowedDataset(
        X={"eeg": np.array(xs)},
        y={"cognitive_load": np.array(ys)},
        subject_ids=np.array(subs),
        session_ids=np.zeros(n, dtype=int),
        window_times=np.array(times, dtype=float),
        trial_ids=np.array(trials),
        normalization_source=np.zeros(n, dtype=bool),
    )


def test_within_subject_runs_with_both_guards_on():
    """allows_same_subject=True splitter must not be rejected by the
    subject-overlap guard — that is the whole point of the controller
    decision in Task 16: the guard enforces what a splitter *claims*,
    not a blanket "no shared subjects" rule. The window-overlap guard
    stays unconditional and must still pass because within-subject
    splits are made along trial boundaries, so windows never cross them.
    """
    ds = make_block_dataset()
    results = run_folds(
        ds, get_splitter("within_subject"), target="cognitive_load",
        modalities=["eeg"], guards=GUARDS_ON, seed=0,
    )
    assert len(results) > 0
    for r in results:
        assert r.n_train > 0
        assert r.n_test > 0
        assert len(r.y_true) == len(r.y_pred) == r.n_test


def test_run_folds_runs_the_session_guard_for_a_splitter_that_forbids_sharing():
    """세션 공유를 금지한다고 선언한 분할기가 실제로 공유하면 실패시킨다.

    _dataset_with_two_sessions()는 세션을 [0,0,1,1]로 준다. 이 분할은
    양쪽에 세션 0과 1이 다 들어가므로 세션 중첩이 발생한다 — 인덱스까지
    중첩되면 세션 가드보다 먼저 다른 가드가 터질 수 있으므로, train/test가
    인덱스는 겹치지 않되 세션은 겹치는 [0,2]/[1,3]을 쓴다.
    """
    ds = _dataset_with_two_sessions()

    class _Bad:
        allows_same_subject = True
        allows_same_session = False

        def split(self, keys):
            yield np.array([0, 2]), np.array([1, 3])

    with pytest.raises(LeakageError, match="sessions appear in both"):
        run_folds(
            ds, _Bad(), target="t", modalities=["m"],
            guards={"check_subject_overlap": True, "check_window_overlap": False,
                    "check_session_overlap": True, "check_normalization_source": True},
            seed=0,
        )


def test_run_folds_skips_the_session_guard_when_the_splitter_allows_sharing():
    """같은 분할([0,2]/[1,3])이라도 allows_same_session=True면 통과한다.

    선언만 다를 뿐 분할 자체는 위 테스트와 동일하다 — 세션 가드가
    분할기의 주장을 존중한다는 것이 이 쌍의 요점이다.
    """
    ds = _dataset_with_two_sessions()

    class _WithinLike:
        allows_same_subject = True
        allows_same_session = True

        def split(self, keys):
            yield np.array([0, 2]), np.array([1, 3])

    results = run_folds(
        ds, _WithinLike(), target="t", modalities=["m"],
        guards={"check_subject_overlap": True, "check_window_overlap": False,
                "check_session_overlap": True, "check_normalization_source": True},
        seed=0,
    )
    assert len(results) == 1
