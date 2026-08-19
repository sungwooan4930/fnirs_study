import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.datasets.contract import LeakageError, TestView, TrainView, WindowedDataset


def make_dataset(n=12):
    rng = np.random.default_rng(0)
    return WindowedDataset(
        X={"eeg": rng.normal(size=(n, 4)), "fnirs": rng.normal(size=(n, 3))},
        y={"cognitive_load": np.arange(n) % 3},
        subject_ids=np.array([f"sub-{i // 4 + 1:02d}" for i in range(n)]),
        session_ids=np.zeros(n, dtype=int),
        window_times=np.column_stack([np.arange(n, dtype=float), np.arange(n, dtype=float) + 5.0]),
        trial_ids=np.arange(n) // 2,
        normalization_source=np.zeros(n, dtype=bool),
    )


class DummySplitter:
    """앞 8개를 train, 뒤 4개를 test로 주는 고정 분할기."""

    allows_same_subject = False
    allows_same_session = False

    def split(self, keys):
        idx = np.arange(len(keys.subject_ids))
        yield idx[:8], idx[8:]


def test_no_public_x_or_y_attribute():
    ds = make_dataset()
    assert not hasattr(ds, "X")
    assert not hasattr(ds, "y")


def test_basic_properties():
    ds = make_dataset()
    assert ds.n_windows == 12
    assert sorted(ds.modalities) == ["eeg", "fnirs"]
    assert ds.targets == ["cognitive_load"]


def test_subject_ids_are_exposed_but_copied():
    ds = make_dataset()
    a = ds.get_subject_ids()
    a[0] = "TAMPERED"
    assert ds.get_subject_ids()[0] == "sub-01"


def test_iter_folds_yields_views():
    ds = make_dataset()
    folds = list(ds.iter_folds(DummySplitter()))
    assert len(folds) == 1
    assert isinstance(folds[0].train, TrainView)
    assert isinstance(folds[0].test, TestView)
    assert folds[0].fold_id == 0


def test_view_x_concatenates_modalities_in_given_order():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    x = fold.train.X(["eeg", "fnirs"])
    assert x.shape == (8, 7)


def test_view_x_respects_caller_order_not_sorted_order():
    # eeg has width 4, fnirs has width 3 (different widths, so a
    # transposition of the blocks is unmistakable in the output shape/values).
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    forward = fold.train.X(["eeg", "fnirs"])
    reversed_ = fold.train.X(["fnirs", "eeg"])
    expected_reversed = np.hstack([fold.train.X(["fnirs"]), fold.train.X(["eeg"])])
    assert np.array_equal(reversed_, expected_reversed)
    assert not np.array_equal(reversed_, forward)


def test_view_x_rejects_unknown_modality():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    with pytest.raises(KeyError, match="bogus"):
        fold.train.X(["bogus"])


def test_view_y_rejects_unknown_target():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    with pytest.raises(KeyError, match="bogus"):
        fold.train.y("bogus")


def test_train_and_test_have_disjoint_rows():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    assert fold.train.X(["eeg"]).shape[0] == 8
    assert fold.test.X(["eeg"]).shape[0] == 4


def test_testview_fit_raises_leakage_error():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    with pytest.raises(LeakageError, match="fit"):
        fold.test.fit(StandardScaler())


def test_testview_fit_transform_raises_leakage_error():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    with pytest.raises(LeakageError, match="fit"):
        fold.test.fit_transform(StandardScaler())


def test_testview_transform_with_fitted_transformer_is_allowed():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    scaler = StandardScaler().fit(fold.train.X(["eeg"]))
    out = fold.test.transform(scaler, ["eeg"])
    assert out.shape == (4, 4)


def test_trainview_groups_returns_subject_ids():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    assert np.array_equal(fold.train.groups(), fold.train.subject_ids())


def test_rejects_length_mismatch():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="length"):
        WindowedDataset(
            X={"eeg": rng.normal(size=(10, 4))},
            y={"cognitive_load": np.arange(9)},
            subject_ids=np.array(["sub-01"] * 10),
            session_ids=np.zeros(10, dtype=int),
            window_times=np.zeros((10, 2)),
            trial_ids=np.zeros(10, dtype=int),
            normalization_source=np.zeros(10, dtype=bool),
        )


def test_rejects_empty_dataset():
    with pytest.raises(ValueError, match="empty"):
        WindowedDataset(
            X={"eeg": np.zeros((0, 4))},
            y={"cognitive_load": np.zeros(0, dtype=int)},
            subject_ids=np.array([], dtype="<U6"),
            session_ids=np.array([], dtype=int),
            window_times=np.zeros((0, 2)),
            trial_ids=np.zeros(0, dtype=int),
            normalization_source=np.array([], dtype=bool),
        )


from src.datasets.contract import SplitKeys, WindowedDataset


def _make_dataset(
    subjects=("a", "a", "b", "b"),
    sessions=(0, 1, 0, 1),
    trials=(0, 1, 2, 3),
    norm_src=(False, False, False, False),
):
    n = len(subjects)
    return WindowedDataset(
        X={"m": np.arange(n * 2, dtype=float).reshape(n, 2)},
        y={"t": np.array([0, 1, 0, 1])},
        subject_ids=np.array(subjects),
        session_ids=np.array(sessions),
        window_times=np.column_stack([np.arange(n, dtype=float), np.arange(n, dtype=float) + 5]),
        trial_ids=np.array(trials),
        normalization_source=np.array(norm_src),
    )


def test_dataset_exposes_session_ids():
    ds = _make_dataset()
    assert ds.get_session_ids().tolist() == [0, 1, 0, 1]


def test_trial_id_shared_across_two_sessions_is_rejected():
    """trial_id가 녹화마다 0부터 세면 세션 간 충돌한다.

    충돌하면 WithinSubjectSplitter가 두 세션의 블록을 한 덩어리로 묶어
    시행이 더 이상 일관된 단위가 아니게 된다.
    """
    with pytest.raises(ValueError, match="trial_id .* appears under"):
        _make_dataset(trials=(0, 0, 1, 1))


def test_trial_id_shared_across_two_subjects_is_rejected():
    with pytest.raises(ValueError, match="trial_id .* appears under"):
        _make_dataset(subjects=("a", "a", "b", "b"), sessions=(0, 0, 0, 0), trials=(0, 1, 0, 1))


def test_views_expose_session_ids_and_normalization_source():
    ds = _make_dataset(norm_src=(True, False, False, False))

    class _AllSplit:
        allows_same_subject = True
        allows_same_session = True

        def split(self, keys):
            yield np.array([0, 1]), np.array([2, 3])

    fold = next(ds.iter_folds(_AllSplit()))
    assert fold.train.session_ids().tolist() == [0, 1]
    assert fold.test.session_ids().tolist() == [0, 1]
    assert fold.train.normalization_source().tolist() == [True, False]
    assert fold.test.normalization_source().tolist() == [False, False]


def test_splitter_receives_a_splitkeys_object():
    ds = _make_dataset()
    seen = {}

    class _Spy:
        allows_same_subject = True
        allows_same_session = True

        def split(self, keys):
            seen["keys"] = keys
            yield np.array([0, 1]), np.array([2, 3])

    next(ds.iter_folds(_Spy(), stratify_target="t"))
    keys = seen["keys"]
    assert isinstance(keys, SplitKeys)
    assert keys.subject_ids.tolist() == ["a", "a", "b", "b"]
    assert keys.session_ids.tolist() == [0, 1, 0, 1]
    assert keys.trial_ids.tolist() == [0, 1, 2, 3]
    assert keys.labels.tolist() == [0, 1, 0, 1]


def test_labels_are_none_without_stratify_target():
    ds = _make_dataset()
    seen = {}

    class _Spy:
        allows_same_subject = True
        allows_same_session = True

        def split(self, keys):
            seen["keys"] = keys
            yield np.array([0]), np.array([1])

    next(ds.iter_folds(_Spy()))
    assert seen["keys"].labels is None
