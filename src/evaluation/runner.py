"""실험 러너 — 오케스트레이션과 결과 기록.

CLAUDE.md 5.2가 요구하는 "실험 1회 = 결과 디렉토리 1개"를 구현한다.
config 스냅샷·시드·git commit·환경을 함께 저장하지 않으면 몇 달 뒤
그 숫자가 어디서 나왔는지 알 수 없게 된다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from src.common.config import load_config, validate_config
from src.common.seeding import set_all_seeds
from src.datasets.contract import WindowedDataset
from src.datasets.extractors import get_extractor
from src.datasets.labels import build_labels
from src.datasets.windowing import make_windows
from src.evaluation.harness import run_folds
from src.evaluation.metrics import aggregate
from src.evaluation.splitters import get_splitter
from src.preprocessing.baseline import (
    MODALITY_KIND,
    SessionBaseline,
    SessionQuality,
    compute_drift,
    flag_drift,
)
from src.simulation.recording import generate_dataset
from src.simulation.state import BASELINE

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


def _config_hash(cfg: dict) -> str:
    """오버라이드가 병합된 *실효* config의 짧은 해시.

    run_id가 run_name·seed·commit만 담으면 오버라이드로 조건을 바꾼
    실행들이 전부 같은 디렉토리를 가리킨다. mkdir(exist_ok=True)가
    조용히 통과하고 metrics.json·config.yaml·per_fold.csv가 덮어써지므로,
    T4 스윕 6개 조건을 기본 results/에 돌리면 5개가 파괴되고 마지막
    하나만 살아남는다 — 게다가 살아남은 디렉토리는 내부적으로 일관되어
    아무것도 잘못돼 보이지 않는다. 스펙 7.2와 CLAUDE.md 5.2
    ("실험 1회 = 결과 디렉토리 1개") 위반이다.

    output(결과 저장 위치)은 해시에서 제외한다. 어디에 쓰느냐는 실험의
    정체성이 아니라 보관 위치일 뿐이고, 포함시키면 같은 실험을 다른
    폴더에 쓰기만 해도 다른 실험처럼 보인다.
    """
    payload = {k: v for k, v in cfg.items() if k != "output"}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return "cfg" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def build_dataset(
    cfg: dict, rng: np.random.Generator
) -> tuple[WindowedDataset, int, list[SessionQuality]]:
    """합성 녹화를 만들고 창·특징·정규화·라벨을 거쳐 계약 객체를 조립한다.

    정규화는 **녹화(= 한 피험자의 한 세션) 단위로 독립 수행**한다. 기준은
    그 세션의 시작 베이스라인이고, 종료 베이스라인은 드리프트 추정에만
    쓴다 (CLAUDE.md §3.8).
    """
    win_cfg = cfg["windowing"]
    ds_cfg = cfg["dataset"]
    base_cfg = cfg["preprocessing"]["baseline"]
    lead_delta_s = float(cfg["simulation"]["lead_delta_s"])

    normalize = bool(base_cfg["normalize"])
    zero_atol = float(base_cfg["zero_atol"])
    drift_threshold = float(base_cfg["drift_threshold_relative"])
    include_baseline = bool(ds_cfg["include_baseline"])

    extract_features = get_extractor(cfg["features"]["extractor"])
    lead_targets = list(ds_cfg["lead_targets"])

    recordings = generate_dataset(cfg["simulation"], rng)

    feature_blocks: dict[str, list[np.ndarray]] = {}
    label_blocks: dict[str, list[np.ndarray]] = {}
    subjects, sessions, times, trials, norm_src = [], [], [], [], []
    qualities: list[SessionQuality] = []
    n_dropped = 0
    trial_offset = 0

    for rec in recordings:
        windows = make_windows(rec.timeline, win_cfg["window_s"], win_cfg["step_s"])
        feats = extract_features(rec, windows)

        is_base = windows.block_kind == BASELINE
        start_mask = is_base & (windows.trial_id == windows.trial_id.min())
        end_mask = is_base & (windows.trial_id == windows.trial_id.max())
        if not start_mask.any() or not end_mask.any():
            raise ValueError(
                f"{rec.subject_id} ses-{rec.session_idx}: 시작 또는 종료 "
                "베이스라인에서 창이 나오지 않았다. CLAUDE.md §2.4는 두 "
                "베이스라인을 생략 불가로 규정한다"
            )

        # 드리프트는 **정규화 전 원 단위**에서 잰다. dB 변환 후에는 기준값이
        # 정의상 0이 되어 상대 비율이 성립하지 않는다.
        drift_by_modality: dict[str, float] = {}
        n_excluded_by_modality: dict[str, int] = {}
        for name, arr in feats.items():
            if MODALITY_KIND[name] == "absolute":
                continue  # 행동은 장비 재부착의 영향을 받지 않으므로 드리프트 대상이 아니다
            drift = compute_drift(arr[start_mask], arr[end_mask], zero_atol=zero_atol)
            drift_by_modality[name] = drift.aggregate
            n_excluded_by_modality[name] = drift.n_excluded

        qualities.append(
            SessionQuality(
                subject_id=rec.subject_id,
                session_idx=rec.session_idx,
                drift_by_modality=drift_by_modality,
                n_excluded_by_modality=n_excluded_by_modality,
                drift_flag=flag_drift(drift_by_modality, drift_threshold),
            )
        )

        if normalize:
            for name in list(feats):
                baseline = SessionBaseline.fit(
                    feats[name][start_mask], MODALITY_KIND[name]
                )
                feats[name] = baseline.apply(feats[name])

        labels, keep = build_labels(
            rec, windows,
            lead_delta_s=lead_delta_s,
            rt_bins=list(ds_cfg["rt_bins"]),
            lead_targets=lead_targets,
            include_baseline=include_baseline,
        )
        n_dropped += int((~keep).sum())

        for name, arr in feats.items():
            feature_blocks.setdefault(name, []).append(arr[keep])
        for name, arr in labels.items():
            label_blocks.setdefault(name, []).append(arr[keep])

        n_kept = int(keep.sum())
        subjects.append(np.full(n_kept, rec.subject_id))
        sessions.append(np.full(n_kept, rec.session_idx))
        times.append(np.column_stack([windows.start_s, windows.end_s])[keep])
        # trial_id는 세션마다 0부터 세므로 전역 오프셋을 더한다. 계약이
        # 고유성을 검증하지만, 만드는 책임은 여기에 있다.
        trials.append(windows.trial_id[keep] + trial_offset)
        # 정규화에 실제로 쓰인 창만 표식한다. normalize가 꺼져 있으면
        # 아무것도 fit되지 않았으므로 표식도 없다.
        norm_src.append(
            start_mask[keep] if normalize else np.zeros(n_kept, dtype=bool)
        )
        trial_offset += int(windows.trial_id.max()) + 1

    dataset = WindowedDataset(
        X={k: np.vstack(v) for k, v in feature_blocks.items()},
        y={k: np.concatenate(v) for k, v in label_blocks.items()},
        subject_ids=np.concatenate(subjects),
        session_ids=np.concatenate(sessions),
        window_times=np.vstack(times),
        trial_ids=np.concatenate(trials),
        normalization_source=np.concatenate(norm_src),
    )
    return dataset, n_dropped, qualities


def run_experiment(config_path, *, overrides: dict | None = None) -> Path:
    """config 하나를 실행하고 결과 디렉토리 경로를 반환한다."""
    cfg = load_config(config_path)
    if overrides:
        cfg = _deep_update(cfg, overrides)
        # P1: 병합 후 반드시 재검증한다. 하지 않으면 오타 난 오버라이드 키가
        # 조용히 통과하고, _config_hash만 바뀌어 새 결과 디렉토리가 생긴다 —
        # 아무 손잡이도 돌리지 않은 실행이 별개 조건처럼 기록된다.
        validate_config(cfg)

    targets = list(cfg["dataset"]["targets"])
    if len(targets) != 1:
        raise ValueError(
            f"dataset.targets has {len(targets)} entries {targets}; multi-target "
            "evaluation is not supported yet — 러너는 targets[0] 하나만 평가하고 "
            "나머지를 조용히 버린다. 여러 타깃을 평가하려면 config를 나눠 "
            "각각 실행하라"
        )

    if bool(cfg["dataset"]["include_baseline"]) and "cognitive_load" in targets:
        raise ValueError(
            "dataset.include_baseline=true 인데 target이 'cognitive_load'다. "
            "베이스라인 창은 과제 조건이 없어 센티넬 라벨(-1)을 갖는다 — 부하 "
            "분류에 넣으면 클래스가 하나 늘고 chance level이 33.3%에서 25%로 "
            "조용히 바뀐다. CLAUDE.md §9.2 4번('3수준 분류'의 정의)이 미확정이므로 "
            "코드가 임의로 확정하지 않는다."
        )

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

    run_id = f"{cfg['run_name']}_seed{seed}_{commit}_{_config_hash(cfg)}"
    if dirty:
        run_id += "-dirty"
    if guards_disabled:
        run_id += "-UNSAFE"

    out_dir = Path(cfg["output"]["results_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, n_dropped, qualities = build_dataset(cfg, rng)

    target = targets[0]

    # n_classes는 과제 config가 아니라 실제로 평가하는 라벨 배열에서 유도한다.
    # len(nback_levels)로 두면 targets를 accuracy(이진)로 바꾸는 것만으로
    # chance_level이 0.3333으로 기록되고 binomtest가 틀린 귀무가설을 쓰는데,
    # 아무 오류도 나지 않는다 — config 변경이 CLAUDE.md 5.4 위반을
    # 제조하는 셈이다.
    class_labels = dataset.class_labels(target)
    n_classes = int(len(class_labels))
    expected = np.arange(n_classes)
    if not np.array_equal(class_labels, expected):
        raise ValueError(
            f"target '{target}' has non-contiguous class labels "
            f"{class_labels.tolist()}; chance level과 혼동행렬 축이 "
            "0..n-1 정수 라벨을 전제한다"
        )

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
    metrics["n_sessions_flagged"] = sum(1 for q in qualities if q.drift_flag)
    metrics["n_sessions"] = len(qualities)

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

    with (out_dir / "session_quality.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        modalities = sorted(qualities[0].drift_by_modality) if qualities else []
        writer.writerow(
            ["subject_id", "session_idx", "drift_flag"]
            + [f"drift_{m}" for m in modalities]
            + [f"n_excluded_{m}" for m in modalities]
        )
        for q in qualities:
            writer.writerow(
                [q.subject_id, q.session_idx, q.drift_flag]
                + [f"{q.drift_by_modality[m]:.6f}" for m in modalities]
                + [q.n_excluded_by_modality[m] for m in modalities]
            )

    (out_dir / "log.txt").write_text(
        f"run_id={run_id}\n"
        f"cv_method={metrics['cv_method']}\n"
        f"chance_level={metrics['chance_level']:.4f}\n"
        f"accuracy_mean={metrics['accuracy_mean']:.4f}\n"
        f"accuracy_worst={metrics['accuracy_worst']:.4f}\n"
        f"fold_ci=[{metrics['fold_ci_low']:.4f}, {metrics['fold_ci_high']:.4f}]\n"
        f"n_windows_dropped={n_dropped}\n"
        f"guards_disabled={guards_disabled}\n"
        f"n_sessions_flagged={metrics['n_sessions_flagged']}/{metrics['n_sessions']}\n",
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
