"""실험 러너 — 오케스트레이션과 결과 기록.

CLAUDE.md 5.2가 요구하는 "실험 1회 = 결과 디렉토리 1개"를 구현한다.
config 스냅샷·시드·git commit·환경을 함께 저장하지 않으면 몇 달 뒤
그 숫자가 어디서 나왔는지 알 수 없게 된다.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from src.common.config import load_config
from src.common.seeding import set_all_seeds
from src.datasets.contract import WindowedDataset
from src.datasets.features_minimal import extract_features
from src.datasets.labels import build_labels
from src.datasets.windowing import make_windows
from src.evaluation.harness import run_folds
from src.evaluation.metrics import aggregate
from src.evaluation.splitters import get_splitter
from src.simulation.recording import generate_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    """git 명령을 실행한다. 실패하면 빈 문자열이 아니라 None을 돌려줘서
    "git이 없다"와 "git 출력이 비어 있다"를 구분한다 — 후자를 clean으로
    오인하면 실제로는 알 수 없는 상태를 clean/tracked로 위장하게 된다.

    인코딩은 명시적으로 utf-8을 쓴다. Windows 콘솔 코드페이지(cp949)에
    맡기면 `git diff`에 한글 커밋 메시지·주석이 섞였을 때 디코딩이
    실패해 별도 스레드에서 조용히 죽고 stdout이 None으로 돌아온다 —
    "git 상태 조회 실패"가 아니라 "결과 없음"으로 위장되는 것과 같은
    문제라서 여기서도 errors="replace"로 방어하고, stdout이 그래도
    None이면 실패로 취급한다.
    """
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.stdout is None:
        return None
    return result.stdout.strip()


def _deep_update(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def build_dataset(cfg: dict, rng: np.random.Generator) -> tuple[WindowedDataset, int]:
    """합성 녹화를 만들고 창·특징·라벨을 거쳐 계약 객체를 조립한다."""
    win_cfg = cfg["windowing"]
    ds_cfg = cfg["dataset"]
    lead_delta_s = float(cfg["simulation"]["lead_delta_s"])

    recordings = generate_dataset(cfg["simulation"], rng)

    feature_blocks: dict[str, list[np.ndarray]] = {}
    label_blocks: dict[str, list[np.ndarray]] = {}
    subjects, times, trials = [], [], []
    n_dropped = 0

    for rec in recordings:
        windows = make_windows(rec.timeline, win_cfg["window_s"], win_cfg["step_s"])
        feats = extract_features(rec, windows)
        labels, keep = build_labels(
            rec, windows,
            lead_delta_s=lead_delta_s,
            rt_bins=list(ds_cfg["rt_bins"]),
        )
        n_dropped += int((~keep).sum())

        for name, arr in feats.items():
            feature_blocks.setdefault(name, []).append(arr[keep])
        for name, arr in labels.items():
            label_blocks.setdefault(name, []).append(arr[keep])

        subjects.append(np.full(int(keep.sum()), rec.subject_id))
        times.append(np.column_stack([windows.start_s, windows.end_s])[keep])
        trials.append(windows.trial_id[keep])

    dataset = WindowedDataset(
        X={k: np.vstack(v) for k, v in feature_blocks.items()},
        y={k: np.concatenate(v) for k, v in label_blocks.items()},
        subject_ids=np.concatenate(subjects),
        window_times=np.vstack(times),
        trial_ids=np.concatenate(trials),
    )
    return dataset, n_dropped


def run_experiment(config_path, *, overrides: dict | None = None) -> Path:
    """config 하나를 실행하고 결과 디렉토리 경로를 반환한다."""
    cfg = load_config(config_path)
    if overrides:
        cfg = _deep_update(cfg, overrides)

    seed = int(cfg["seed"])
    rng = set_all_seeds(seed)

    guards = cfg["evaluation"]["guards"]
    guards_disabled = not all(bool(v) for v in guards.values())

    commit_raw = _git("rev-parse", "--short", "HEAD")
    status_raw = _git("status", "--porcelain")
    git_available = commit_raw is not None and status_raw is not None
    commit = commit_raw or "nogit"
    # dirty는 3값 논리다: git 조회가 실패하면 "깨끗함"을 조작해내지 않고
    # None(알 수 없음)으로 남긴다. run_id에는 여전히 "nogit"이 노출되어
    # 디렉토리 이름만 봐도 provenance가 불확실하다는 것이 드러난다.
    dirty: bool | None = bool(status_raw) if git_available else None

    run_id = f"{cfg['run_name']}_seed{seed}_{commit}"
    if dirty:
        run_id += "-dirty"
    if guards_disabled:
        run_id += "-UNSAFE"

    out_dir = Path(cfg["output"]["results_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, n_dropped = build_dataset(cfg, rng)

    target = cfg["dataset"]["targets"][0]
    n_classes = int(len(cfg["simulation"]["task"]["nback_levels"]))

    splitter = get_splitter(cfg["evaluation"]["splitter"], seed=seed)
    fold_results = run_folds(
        dataset, splitter,
        target=target,
        modalities=list(cfg["dataset"]["modalities"]),
        guards=guards,
        seed=seed,
        model_name=cfg["evaluation"]["model"],
    )

    metrics = aggregate(
        fold_results, n_classes=n_classes, cv_method=cfg["evaluation"]["splitter"]
    )
    metrics["n_windows_dropped"] = n_dropped
    metrics["guards_disabled"] = guards_disabled
    metrics["target"] = target

    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "env.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "git_available": git_available,
                "git_commit": commit,
                "git_dirty": dirty,
                "git_diff": (_git("diff") or "") if dirty else "",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with (out_dir / "per_fold.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold_id", "test_subjects", "n_train", "n_test", "accuracy"])
        for r in fold_results:
            writer.writerow(
                [r.fold_id, ";".join(r.test_subjects), r.n_train, r.n_test, f"{r.accuracy:.6f}"]
            )

    (out_dir / "log.txt").write_text(
        f"run_id={run_id}\n"
        f"cv_method={metrics['cv_method']}\n"
        f"chance_level={metrics['chance_level']:.4f}\n"
        f"accuracy_mean={metrics['accuracy_mean']:.4f}\n"
        f"accuracy_worst={metrics['accuracy_worst']:.4f}\n"
        f"n_windows_dropped={n_dropped}\n"
        f"guards_disabled={guards_disabled}\n",
        encoding="utf-8",
    )

    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="합성 테스트베드 실험 실행")
    parser.add_argument("config", help="config/experiments/*.yaml 경로")
    args = parser.parse_args()

    result_dir = run_experiment(args.config)
    print(result_dir)
    print((result_dir / "log.txt").read_text(encoding="utf-8"))
