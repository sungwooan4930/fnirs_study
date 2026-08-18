import json

import numpy as np

from src.common.seeding import set_all_seeds
from src.simulation.recording import generate_dataset, generate_recording, write_bids_metadata
from src.simulation.subject import make_subjects

SIM_CFG = {
    "n_subjects": 3,
    "subject_variance": 0.5,
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


def test_recording_shapes():
    rng = set_all_seeds(0)
    sub = make_subjects(1, 0.5, rng)[0]
    rec = generate_recording(sub, SIM_CFG, rng)
    assert rec.eeg.shape[0] == 30
    assert rec.hbo.shape[0] == 48
    assert rec.hbr.shape == rec.hbo.shape
    assert rec.eeg.shape[1] == int(round(rec.timeline.duration_s * 250))
    assert rec.hbo.shape[1] == int(round(rec.timeline.duration_s * 10.4))


def test_recording_has_no_nans():
    rng = set_all_seeds(0)
    sub = make_subjects(1, 0.5, rng)[0]
    rec = generate_recording(sub, SIM_CFG, rng)
    assert np.isfinite(rec.eeg).all()
    assert np.isfinite(rec.hbo).all()
    assert np.isfinite(rec.hbr).all()


def test_dataset_has_requested_subject_count():
    recs = generate_dataset(SIM_CFG, set_all_seeds(0))
    assert len(recs) == 3
    assert [r.subject_id for r in recs] == ["sub-01", "sub-02", "sub-03"]


def test_subjects_differ_from_each_other():
    recs = generate_dataset(SIM_CFG, set_all_seeds(0))
    assert not np.array_equal(recs[0].eeg, recs[1].eeg)


def test_dataset_is_reproducible():
    a = generate_dataset(SIM_CFG, set_all_seeds(11))
    b = generate_dataset(SIM_CFG, set_all_seeds(11))
    assert np.array_equal(a[0].eeg, b[0].eeg)
    assert np.array_equal(a[2].hbo, b[2].hbo)


def test_bids_metadata_written(tmp_path):
    recs = generate_dataset(SIM_CFG, set_all_seeds(0))
    write_bids_metadata(recs, tmp_path)

    desc = json.loads((tmp_path / "dataset_description.json").read_text(encoding="utf-8"))
    assert desc["BIDSVersion"]
    assert "synthetic" in desc["Name"].lower()

    lines = (tmp_path / "participants.tsv").read_text(encoding="utf-8").strip().split("\n")
    assert lines[0].split("\t")[0] == "participant_id"
    assert len(lines) == 4  # 헤더 + 피험자 3명
