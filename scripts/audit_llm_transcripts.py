#!/usr/bin/env python3
"""Reconstruct the Phi LLM transcript audit path from bundled prompt files."""

from __future__ import annotations

import argparse
import json
import math
import re
from json import JSONDecoder
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "prompts" / "llm_audit_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the bundled Phi LLM transcripts and reconstruct the manuscript summaries."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to stdout when omitted.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_transcript(path: Path) -> dict[str, list[dict]]:
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"={6,}\nPROMPT \d+: ", text)[1:]
    decoder = JSONDecoder()
    parsed: dict[str, list[dict]] = {}

    for section in sections:
        _, body = section.split("\n", 1)
        objects = []
        index = 0
        while index < len(body):
            if body[index] != "{":
                index += 1
                continue
            try:
                obj, end = decoder.raw_decode(body, index)
            except json.JSONDecodeError:
                index += 1
                continue
            if isinstance(obj, dict) and "batch" in obj and "results" in obj:
                objects.append(obj)
            index = end

        if not objects:
            raise ValueError(f"No transcript JSON payload found in {path}")

        payload = objects[-1]
        parsed[payload["batch"]] = payload["results"]

    return parsed


def entropy(weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return 0.0
    value = 0.0
    for weight in weights:
        if weight > 0:
            probability = weight / total
            value -= probability * math.log2(probability)
    return value


def round3(value: float) -> float:
    return round(value, 3)


def summarize_model_results(results: list[dict], included_ids: list[str]) -> tuple[list[dict], dict]:
    id_to_result = {entry["id"]: entry for entry in results}
    sentence_rows = []
    for sentence_id in included_ids:
        if sentence_id not in id_to_result:
            raise ValueError(f"Missing sentence {sentence_id} in transcript payload")
        entry = id_to_result[sentence_id]
        weights = [item["confidence"] for item in entry["interpretations"]]
        sentence_rows.append(
            {
                "id": sentence_id,
                "text": entry["text"],
                "state_size": len(entry["interpretations"]),
                "entropy": round3(entropy(weights)),
            }
        )

    summary = {
        "n": len(sentence_rows),
        "mean_state_size": round3(
            sum(row["state_size"] for row in sentence_rows) / len(sentence_rows)
        ),
        "mean_entropy": round3(
            sum(row["entropy"] for row in sentence_rows) / len(sentence_rows)
        ),
    }
    return sentence_rows, summary


def build_audit_report(manifest: dict) -> dict:
    transcript_files = {
        model: REPO_ROOT / rel_path
        for model, rel_path in manifest["transcript_files"].items()
    }
    parsed = {model: parse_transcript(path) for model, path in transcript_files.items()}

    report = {
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "transcript_files": manifest["transcript_files"],
        "model_category_summaries": {},
        "sentence_level_llm_entropy": {},
        "category_means": {},
        "combined_summary": {},
    }

    llm_entropy_total = 0.0
    llm_state_size_total = 0.0
    llm_sentence_count = 0

    for category, sentence_ids in manifest["llm_sentence_groups"].items():
        report["model_category_summaries"][category] = {}
        per_model_rows: dict[str, list[dict]] = {}
        per_model_means = []
        per_model_state_sizes = []

        for model, batches in parsed.items():
            rows, summary = summarize_model_results(batches[category], sentence_ids)
            per_model_rows[model] = rows
            report["model_category_summaries"][category][model] = summary
            per_model_means.append(summary["mean_entropy"])
            per_model_state_sizes.append(summary["mean_state_size"])

        sentence_level_rows = []
        for sentence_id in sentence_ids:
            model_rows = {model: next(row for row in rows if row["id"] == sentence_id) for model, rows in per_model_rows.items()}
            sentence_level_rows.append(
                {
                    "id": sentence_id,
                    "text": model_rows["chatgpt"]["text"],
                    "entropy_by_model": {
                        model: model_rows[model]["entropy"] for model in parsed
                    },
                    "state_size_by_model": {
                        model: model_rows[model]["state_size"] for model in parsed
                    },
                    "three_model_average_entropy": round3(
                        sum(model_rows[model]["entropy"] for model in parsed) / len(parsed)
                    ),
                    "three_model_average_state_size": round3(
                        sum(model_rows[model]["state_size"] for model in parsed) / len(parsed)
                    ),
                }
            )

        category_mean_entropy = round3(
            sum(row["three_model_average_entropy"] for row in sentence_level_rows)
            / len(sentence_level_rows)
        )
        category_mean_state_size = round3(
            sum(row["three_model_average_state_size"] for row in sentence_level_rows)
            / len(sentence_level_rows)
        )

        report["sentence_level_llm_entropy"][category] = sentence_level_rows
        report["category_means"][category] = {
            "n": len(sentence_level_rows),
            "mean_state_size": category_mean_state_size,
            "mean_entropy": category_mean_entropy,
        }

        llm_entropy_total += sum(
            row["three_model_average_entropy"] for row in sentence_level_rows
        )
        llm_state_size_total += sum(
            row["three_model_average_state_size"] for row in sentence_level_rows
        )
        llm_sentence_count += len(sentence_level_rows)

    rule_means = manifest["rule_based_category_means"]
    overall_entropy_numerator = llm_entropy_total
    overall_state_size_numerator = llm_state_size_total
    total_n = llm_sentence_count

    for payload in rule_means.values():
        overall_entropy_numerator += payload["mean_entropy"] * payload["n"]
        overall_state_size_numerator += payload["mean_state_size"] * payload["n"]
        total_n += payload["n"]

    report["combined_summary"] = {
        "rule_based_categories": rule_means,
        "llm_only_mean_entropy": round3(llm_entropy_total / llm_sentence_count),
        "llm_only_mean_state_size": round3(llm_state_size_total / llm_sentence_count),
        "overall_mean_entropy": round3(overall_entropy_numerator / total_n),
        "overall_mean_state_size": round3(overall_state_size_numerator / total_n),
        "overall_n": total_n,
    }

    return report


def main() -> int:
    args = parse_args()
    manifest = load_manifest(MANIFEST_PATH)
    report = build_audit_report(manifest)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"

    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
