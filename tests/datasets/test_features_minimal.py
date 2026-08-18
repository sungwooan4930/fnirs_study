import numpy as np

from src.common.seeding import set_all_seeds
from src.datasets.features_minimal import extract_features
from src.datasets.windowing import make_windows
from src.simulation.recording import generate_recording
from src.simulation.subject import make_subjects

SIM_CFG = {
    "n_subjects": 1,
    "subject_variance": 0.0,
    "effect_size": 0.8,
    "lead_delta_s": 1.2,
    "task": {
        "nback_levels": [0, 2, 3],
        "block_duration_s": 30,
        "n_blocks_per_level": 2,
        "stim_interval_s": 2.0,
    },
    "eeg": {"n_channels": 30, "sfreq_hz": 250},
    "fnirs": {"n_channels": 48, "sfreq_hz": 10.4, "hbr_coupling": -0.33},
}


def _setup(effect_size=0.8):
    cfg = {**SIM_CFG, "effect_size": effect_size}
    rng = set_all_seeds(0)
    sub = make_subjects(1, 0.0, rng)[0]
    rec = generate_recording(sub, cfg, rng)
    win = make_windows(rec.timeline, 5.0, 1.0)
    return rec, win, extract_features(rec, win)


def test_feature_shapes():
    rec, win, feats = _setup()
    n = len(win.start_s)
    assert feats["eeg"].shape == (n, 60)
    assert feats["fnirs"].shape == (n, 96)
    assert feats["behavior"].shape == (n, 2)


def test_no_nans():
    _, _, feats = _setup()
    for name, arr in feats.items():
        assert np.isfinite(arr).all(), f"{name} has non-finite values"


def test_theta_feature_separates_load_levels():
    _, win, feats = _setup(effect_size=0.8)
    theta_ch0 = feats["eeg"][:, 0]
    assert theta_ch0[win.load_level == 2].mean() > theta_ch0[win.load_level == 0].mean()


def test_hbo_feature_separates_load_levels():
    _, win, feats = _setup(effect_size=0.8)
    hbo_mean_ch0 = feats["fnirs"][:, 0]
    assert hbo_mean_ch0[win.load_level == 2].mean() > hbo_mean_ch0[win.load_level == 0].mean()


def test_zero_effect_size_collapses_separation():
    _, win, feats = _setup(effect_size=0.0)
    hbo_mean_ch0 = feats["fnirs"][:, 0]
    hi = hbo_mean_ch0[win.load_level == 2].mean()
    lo = hbo_mean_ch0[win.load_level == 0].mean()
    assert abs(hi - lo) < 0.1


def test_behavior_accuracy_is_a_rate():
    _, _, feats = _setup()
    acc = feats["behavior"][:, 0]
    assert acc.min() >= 0.0 and acc.max() <= 1.0
