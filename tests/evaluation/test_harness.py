import numpy as np
import pytest

from src.datasets.contract import LeakageError, WindowedDataset
from src.evaluation.harness import make_model, run_folds
from src.evaluation.splitters import get_splitter

GUARDS_ON = {"check_subject_overlap": True, "check_window_overlap": True}
GUARDS_OFF = {"check_subject_overlap": False, "check_window_overlap": False}


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
            trials.append(i // 10)
    return WindowedDataset(
        X={"eeg": np.array(xs)},
        y={"cognitive_load": np.array(ys)},
        subject_ids=np.array(subs),
        window_times=np.array(times, dtype=float),
        trial_ids=np.array(trials),
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
        window_times=np.column_stack(
            [np.arange(20, dtype=float), np.arange(20, dtype=float) + 5.0]
        ),
        trial_ids=np.arange(20) // 5,
    )
    with pytest.raises(ValueError, match="single class"):
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
                trials.append(t)
    return WindowedDataset(
        X={"eeg": np.array(xs)},
        y={"cognitive_load": np.array(ys)},
        subject_ids=np.array(subs),
        window_times=np.array(times, dtype=float),
        trial_ids=np.array(trials),
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
