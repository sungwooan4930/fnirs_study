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
        window_times=np.column_stack([np.arange(n, dtype=float), np.arange(n, dtype=float) + 5.0]),
        trial_ids=np.arange(n) // 2,
    )


class DummySplitter:
    """앞 8개를 train, 뒤 4개를 test로 주는 고정 분할기."""

    def split(self, subject_ids, trial_ids):
        idx = np.arange(len(subject_ids))
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
            window_times=np.zeros((10, 2)),
            trial_ids=np.zeros(10, dtype=int),
        )


def test_rejects_empty_dataset():
    with pytest.raises(ValueError, match="empty"):
        WindowedDataset(
            X={"eeg": np.zeros((0, 4))},
            y={"cognitive_load": np.zeros(0, dtype=int)},
            subject_ids=np.array([], dtype="<U6"),
            window_times=np.zeros((0, 2)),
            trial_ids=np.zeros(0, dtype=int),
        )
