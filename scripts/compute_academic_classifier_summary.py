#!/usr/bin/env python3

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

from academic_sentiment_evaluator import AcademicSentimentEvaluator, EvaluationRecord
from huggingface_hub import snapshot_download


def wilson_ci(success: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = success / total
    denom = 1 + (z * z / total)
    center = p + (z * z / (2 * total))
    margin = z * math.sqrt((p * (1 - p) + (z * z) / (4 * total)) / total)
    low = max(0.0, (center - margin) / denom)
    high = min(1.0, (center + margin) / denom)
    return (low, high)


def t_critical_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.16,
        14: 2.145,
        15: 2.131,
        16: 2.12,
        17: 2.11,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.08,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.06,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    if df < 1:
        return 1.96
    return table.get(df, 1.96)


def build_records(results_data: dict[str, Any]) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for attack_key, attack_data in results_data.items():
        for request_type, results in attack_data.items():
            for result in results:
                attack_type = str(result.get("attack_type", ""))
                if "steering" not in attack_type:
                    continue
                expected_sentiment = (
                    "positive" if "pos_steering" in attack_type else "negative"
                )
                records.append(
                    EvaluationRecord(
                        attack_key=attack_key,
                        attack_type=attack_type,
                        request_type=request_type,
                        response=str(result.get("response", "")),
                        pdf_file=str(result.get("pdf_file", "")),
                        expected_sentiment=expected_sentiment,
                        original_data=result,
                    )
                )
    return records


def annotate_results_data(
    results_data: dict[str, Any], evaluator: AcademicSentimentEvaluator
) -> list[dict[str, Any]]:
    records = build_records(results_data)
    if not records:
        return []

    eval_results = evaluator.evaluate_steering_attacks(records)
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in eval_results.detailed_results:
        key = (
            str(row.get("attack_key", "")),
            str(row.get("request_type", "")),
            str(row.get("pdf_file", "")),
        )
        by_key[key] = row

    for attack_key, attack_data in results_data.items():
        for request_type, results in attack_data.items():
            for result in results:
                attack_type = str(result.get("attack_type", ""))
                if "steering" not in attack_type:
                    continue
                key = (attack_key, request_type, str(result.get("pdf_file", "")))
                row = by_key.get(key)
                if not row:
                    continue
                result["academic_classifier_success"] = bool(
                    row.get("attack_successful", False)
                )
                result["academic_classifier_confidence"] = float(
                    row.get("confidence", 0.0)
                )
                result["academic_classifier_predicted_sentiment"] = str(
                    row.get("predicted_sentiment", "unknown")
                )

    return eval_results.detailed_results


def compute_bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    success = sum(1 for r in records if bool(r.get("attack_successful", False)))
    rate = (success / total) if total else 0.0
    ci_low, ci_high = wilson_ci(success, total)
    return {
        "success": success,
        "total": total,
        "rate": rate,
        "wilson_ci_95": [ci_low, ci_high],
    }


def iter_records(results_data: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for attack_key, attack_data in results_data.items():
        for request_type, results in attack_data.items():
            for result in results:
                rec = dict(result)
                rec["attack_key"] = attack_key
                rec["request_type"] = request_type
                records.append(rec)
    return records


def channel_success(record: dict[str, Any], channel: str) -> bool | None:
    attack_type = str(record.get("attack_type", ""))
    if channel == "llm_primary":
        return bool(record.get("evaluation_success", False))
    if channel == "llm_judge_a":
        if "steering" in attack_type:
            return bool(record.get("llm_judge_a_success", False))
        return bool(record.get("evaluation_success", False))
    if channel == "llm_judge_b":
        if "steering" in attack_type:
            return bool(record.get("llm_judge_b_success", False))
        return None
    if channel == "llm_consensus":
        if "steering" in attack_type:
            return bool(record.get("llm_consensus_success", False))
        return None
    if channel == "vader":
        if "steering" in attack_type:
            return bool(record.get("vader_sentiment_success", False))
        return None
    if channel == "academic_classifier":
        if "steering" in attack_type:
            val = record.get("academic_classifier_success")
            return bool(val) if val is not None else None
        return None
    return None


def aggregate_runs(run_buckets: list[dict[str, Any]]) -> dict[str, Any]:
    run_values = [bucket["rate"] for bucket in run_buckets if bucket["total"] > 0]
    pooled_success = sum(bucket["success"] for bucket in run_buckets)
    pooled_total = sum(bucket["total"] for bucket in run_buckets)
    mean = statistics.mean(run_values) if run_values else 0.0
    std = statistics.stdev(run_values) if len(run_values) > 1 else 0.0
    sem = std / math.sqrt(len(run_values)) if run_values else 0.0
    tcrit = t_critical_95(len(run_values) - 1)
    t_low = max(0.0, mean - tcrit * sem)
    t_high = min(1.0, mean + tcrit * sem)
    w_low, w_high = wilson_ci(pooled_success, pooled_total)
    return {
        "run_values": run_values,
        "mean_rate": mean,
        "std_dev": std,
        "sem": sem,
        "t_ci_95": [t_low, t_high],
        "pooled_success": pooled_success,
        "pooled_total": pooled_total,
        "pooled_rate": (pooled_success / pooled_total) if pooled_total else 0.0,
        "pooled_wilson_ci_95": [w_low, w_high],
    }


def compute_stats(
    run_values: list[float], pooled_success: int, pooled_total: int
) -> dict[str, Any]:
    n = len(run_values)
    mean = statistics.mean(run_values) if run_values else 0.0
    std = statistics.stdev(run_values) if len(run_values) > 1 else 0.0
    sem = std / math.sqrt(n) if n > 0 else 0.0
    tcrit = t_critical_95(n - 1)
    low = max(0.0, mean - tcrit * sem)
    high = min(1.0, mean + tcrit * sem)
    wilson_low, wilson_high = wilson_ci(pooled_success, pooled_total)
    return {
        "run_values": run_values,
        "mean_rate": mean,
        "std_dev": std,
        "sem": sem,
        "t_ci_95": [low, high],
        "pooled_success": pooled_success,
        "pooled_total": pooled_total,
        "pooled_rate": (pooled_success / pooled_total) if pooled_total else 0.0,
        "pooled_wilson_ci_95": [wilson_low, wilson_high],
    }


def aggregate_full_summary(
    services: list[str],
    run_start: int,
    run_end: int,
    evaluation_dir: Path,
) -> dict[str, Any]:
    channels = [
        "llm_primary",
        "llm_judge_a",
        "llm_judge_b",
        "llm_consensus",
        "vader",
        "academic_classifier",
    ]
    summary: dict[str, Any] = {
        "metadata": {
            "services": services,
            "run_start": run_start,
            "run_end": run_end,
        },
        "services": {},
    }

    for service in services:
        service_bucket: dict[str, Any] = {
            "overall": {},
            "by_attack_type": {},
            "by_attack_key": {},
            "by_attack_type_request_type": {},
        }
        loaded_runs: dict[int, list[dict[str, Any]]] = {}

        for run_id in range(run_start, run_end + 1):
            path = evaluation_dir / f"all_results_{service}_run{run_id}_evaluated.json"
            if not path.exists():
                continue
            loaded_runs[run_id] = iter_records(
                json.loads(path.read_text(encoding="utf-8"))
            )

        if not loaded_runs:
            summary["services"][service] = service_bucket
            continue

        for channel in channels:
            run_rates: list[float] = []
            pooled_success = 0
            pooled_total = 0
            for run_id in sorted(loaded_runs):
                vals = [
                    channel_success(rec, channel)
                    for rec in loaded_runs[run_id]
                    if channel_success(rec, channel) is not None
                ]
                if not vals:
                    continue
                success = sum(1 for v in vals if v)
                total = len(vals)
                pooled_success += success
                pooled_total += total
                run_rates.append(success / total)
            if run_rates:
                service_bucket["overall"][channel] = compute_stats(
                    run_rates, pooled_success, pooled_total
                )

        attack_types = sorted(
            {
                str(rec.get("attack_type", "unknown"))
                for run_recs in loaded_runs.values()
                for rec in run_recs
            }
        )
        for attack_type in attack_types:
            service_bucket["by_attack_type"][attack_type] = {}
            for channel in channels:
                run_rates = []
                pooled_success = 0
                pooled_total = 0
                for run_recs in loaded_runs.values():
                    subset = [
                        r
                        for r in run_recs
                        if str(r.get("attack_type", "")) == attack_type
                    ]
                    vals = [
                        channel_success(rec, channel)
                        for rec in subset
                        if channel_success(rec, channel) is not None
                    ]
                    if not vals:
                        continue
                    success = sum(1 for v in vals if v)
                    total = len(vals)
                    pooled_success += success
                    pooled_total += total
                    run_rates.append(success / total)
                if run_rates:
                    service_bucket["by_attack_type"][attack_type][channel] = (
                        compute_stats(run_rates, pooled_success, pooled_total)
                    )

        attack_keys = sorted(
            {
                str(rec.get("attack_key", "unknown"))
                for run_recs in loaded_runs.values()
                for rec in run_recs
            }
        )
        for attack_key in attack_keys:
            service_bucket["by_attack_key"][attack_key] = {}
            for channel in channels:
                run_rates = []
                pooled_success = 0
                pooled_total = 0
                for run_recs in loaded_runs.values():
                    subset = [
                        r
                        for r in run_recs
                        if str(r.get("attack_key", "")) == attack_key
                    ]
                    vals = [
                        channel_success(rec, channel)
                        for rec in subset
                        if channel_success(rec, channel) is not None
                    ]
                    if not vals:
                        continue
                    success = sum(1 for v in vals if v)
                    total = len(vals)
                    pooled_success += success
                    pooled_total += total
                    run_rates.append(success / total)
                if run_rates:
                    service_bucket["by_attack_key"][attack_key][channel] = (
                        compute_stats(run_rates, pooled_success, pooled_total)
                    )

        combo_keys = sorted(
            {
                f"{rec.get('attack_type', 'unknown')}|{rec.get('request_type', 'unknown')}"
                for run_recs in loaded_runs.values()
                for rec in run_recs
            }
        )
        for combo_key in combo_keys:
            attack_type, request_type = combo_key.split("|", 1)
            service_bucket["by_attack_type_request_type"][combo_key] = {}
            for channel in channels:
                run_rates = []
                pooled_success = 0
                pooled_total = 0
                for run_recs in loaded_runs.values():
                    subset = [
                        r
                        for r in run_recs
                        if str(r.get("attack_type", "")) == attack_type
                        and str(r.get("request_type", "")) == request_type
                    ]
                    vals = [
                        channel_success(rec, channel)
                        for rec in subset
                        if channel_success(rec, channel) is not None
                    ]
                    if not vals:
                        continue
                    success = sum(1 for v in vals if v)
                    total = len(vals)
                    pooled_success += success
                    pooled_total += total
                    run_rates.append(success / total)
                if run_rates:
                    service_bucket["by_attack_type_request_type"][combo_key][
                        channel
                    ] = compute_stats(run_rates, pooled_success, pooled_total)

        summary["services"][service] = service_bucket

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute 5-run academic sentiment classifier summary from evaluated files"
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=ROOT / "results" / "evaluation",
        help="Directory containing all_results_{service}_runN_evaluated.json",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(ROOT / "data" / "models" / "academic-sentiment-classifier"),
        help="Local path to the academic sentiment classifier checkpoint",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="EvilScript/academic-sentiment-classifier",
        help="Hugging Face repo to download if the local checkpoint is missing",
    )
    parser.add_argument(
        "--services",
        nargs="+",
        default=["chatgpt", "gemini"],
        help="Services to process",
    )
    parser.add_argument("--run-start", type=int, default=1)
    parser.add_argument("--run-end", type=int, default=5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "results" / "evaluation" / "academic_classifier_summary.json",
        help="Where to write the aggregated classifier summary",
    )
    parser.add_argument(
        "--full-summary-output",
        type=Path,
        default=ROOT / "results" / "evaluation" / "self_consistency_summary.json",
        help="Where to write the rebuilt full self-consistency summary",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for classifier inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device override, e.g. cuda or cpu",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download the model if the local checkpoint is missing",
    )
    parser.add_argument(
        "--no-write-back",
        action="store_true",
        help="Do not write classifier fields back into the evaluated JSON files",
    )
    parser.add_argument(
        "--no-full-summary",
        action="store_true",
        help="Do not rebuild the full self-consistency summary",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_path)
    if not model_dir.exists():
        if args.skip_download:
            raise FileNotFoundError(
                f"Model checkpoint not found and --skip-download was set: {model_dir}"
            )
        print(f"Downloading classifier from {args.model_repo} to {model_dir}")
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=args.model_repo, local_dir=str(model_dir))

    evaluator = AcademicSentimentEvaluator(
        model_path=str(model_dir),
        batch_size=args.batch_size,
        device=args.device,
    )

    summary: dict[str, Any] = {
        "metadata": {
            "services": args.services,
            "run_start": args.run_start,
            "run_end": args.run_end,
            "model_path": args.model_path,
        },
        "services": {},
    }

    for service in args.services:
        per_run: dict[int, Any] = {}
        for run_id in range(args.run_start, args.run_end + 1):
            input_path = (
                args.evaluation_dir
                / f"all_results_{service}_run{run_id}_evaluated.json"
            )
            if not input_path.exists():
                print(f"Skipping missing file: {input_path}")
                continue

            results_data = json.loads(input_path.read_text(encoding="utf-8"))
            detailed = annotate_results_data(results_data, evaluator)

            if not args.no_write_back:
                input_path.write_text(
                    json.dumps(results_data, indent=2), encoding="utf-8"
                )

            per_run[run_id] = {
                "overall": compute_bucket(detailed),
                "by_attack_type": {},
                "by_attack_type_request_type": {},
            }

            for attack_type in ["pos_steering_attack", "neg_steering_attack"]:
                subset = [r for r in detailed if r.get("attack_type") == attack_type]
                per_run[run_id]["by_attack_type"][attack_type] = compute_bucket(subset)

            combos = [
                ("pos_steering_attack", "standard_request"),
                ("pos_steering_attack", "negative_request"),
                ("neg_steering_attack", "standard_request"),
                ("neg_steering_attack", "positive_request"),
            ]
            for attack_type, request_type in combos:
                key = f"{attack_type}|{request_type}"
                subset = [
                    r
                    for r in detailed
                    if r.get("attack_type") == attack_type
                    and r.get("request_type") == request_type
                ]
                per_run[run_id]["by_attack_type_request_type"][key] = compute_bucket(
                    subset
                )

        service_summary: dict[str, Any] = {
            "runs": per_run,
            "overall": {},
            "by_attack_type": {},
            "by_attack_type_request_type": {},
        }

        if per_run:
            ordered_run_ids = sorted(per_run)
            service_summary["overall"] = aggregate_runs(
                [per_run[run_id]["overall"] for run_id in ordered_run_ids]
            )

            for attack_type in ["pos_steering_attack", "neg_steering_attack"]:
                service_summary["by_attack_type"][attack_type] = aggregate_runs(
                    [
                        per_run[run_id]["by_attack_type"][attack_type]
                        for run_id in ordered_run_ids
                    ]
                )

            for key in per_run[ordered_run_ids[0]]["by_attack_type_request_type"]:
                service_summary["by_attack_type_request_type"][key] = aggregate_runs(
                    [
                        per_run[run_id]["by_attack_type_request_type"][key]
                        for run_id in ordered_run_ids
                    ]
                )

        summary["services"][service] = service_summary

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary to {args.output_json}")

    if not args.no_full_summary:
        full_summary = aggregate_full_summary(
            services=args.services,
            run_start=args.run_start,
            run_end=args.run_end,
            evaluation_dir=args.evaluation_dir,
        )
        args.full_summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.full_summary_output.write_text(
            json.dumps(full_summary, indent=2), encoding="utf-8"
        )
        print(f"Wrote rebuilt full summary to {args.full_summary_output}")


if __name__ == "__main__":
    main()
