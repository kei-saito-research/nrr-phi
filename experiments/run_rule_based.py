#!/usr/bin/env python3
"""
Run rule-based φ-mapping experiment.

Applies the φ function to 40 sentences (20 adversative + 20 hedging)
in English and Japanese, and verifies results against Paper 2 Table 2.

Usage:
    python experiments/run_rule_based.py
    python experiments/run_rule_based.py --output results/rule_based_output.json

Reference: Saito (2026), Paper 2, Section 5 (Rule-Based Extraction).
"""

import argparse
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phi_mapping import phi, phi_batch
from src.state import NRRState


# ─────────────────────────────────────────────────────────────────────
# Test sentences (identical to Paper 2 dataset)
# ─────────────────────────────────────────────────────────────────────

ADVERSATIVE_SENTENCES = [
    # English (10)
    {"id": "adv_en_01", "text": "I want to quit my job, but I also don't want to quit.", "lang": "EN"},
    {"id": "adv_en_02", "text": "She is talented, however she lacks confidence.", "lang": "EN"},
    {"id": "adv_en_03", "text": "The plan seems good, yet I have doubts.", "lang": "EN"},
    {"id": "adv_en_04", "text": "Although I love him, I can't forgive him.", "lang": "EN"},
    {"id": "adv_en_05", "text": "I succeeded, but it doesn't feel like success.", "lang": "EN"},
    {"id": "adv_en_06", "text": "He apologized, however I'm still angry.", "lang": "EN"},
    {"id": "adv_en_07", "text": "I should stay, yet something tells me to leave.", "lang": "EN"},
    {"id": "adv_en_08", "text": "The job pays well, but it's destroying my health.", "lang": "EN"},
    {"id": "adv_en_09", "text": "I want to help, but I don't know how, yet I feel I must try.", "lang": "EN", "triple": True},
    {"id": "adv_en_10", "text": "She said yes, but her eyes said no, however she stayed.", "lang": "EN", "triple": True},
    # Japanese (10)
    {"id": "adv_jp_01", "text": "辞めたいけど、辞めたくない。", "lang": "JP"},
    {"id": "adv_jp_02", "text": "才能はある。しかし自信がない。", "lang": "JP"},
    {"id": "adv_jp_03", "text": "計画は良さそうだが、疑問が残る。", "lang": "JP"},
    {"id": "adv_jp_04", "text": "彼を愛しているけど、許せない。", "lang": "JP"},
    {"id": "adv_jp_05", "text": "成功した。でも成功した気がしない。", "lang": "JP"},
    {"id": "adv_jp_06", "text": "謝ってくれた。しかしまだ怒っている。", "lang": "JP"},
    {"id": "adv_jp_07", "text": "残るべきだが、何かが去れと言っている。", "lang": "JP"},
    {"id": "adv_jp_08", "text": "給料はいいけど、健康を壊している。", "lang": "JP"},
    {"id": "adv_jp_09", "text": "助けたい。でもどうすればいいかわからない。", "lang": "JP", "unequal": True},
    {"id": "adv_jp_10", "text": "彼女はイエスと言ったが、目はノーと言っていた。", "lang": "JP", "unequal": True},
]

HEDGING_SENTENCES = [
    # English (10)
    {"id": "hdg_en_01", "text": "Maybe I should apply for that position.", "lang": "EN"},
    {"id": "hdg_en_02", "text": "Perhaps this is the right time to change.", "lang": "EN"},
    {"id": "hdg_en_03", "text": "I might be overreacting.", "lang": "EN"},
    {"id": "hdg_en_04", "text": "It's probably not as bad as I think.", "lang": "EN"},
    {"id": "hdg_en_05", "text": "Maybe they didn't mean it that way.", "lang": "EN"},
    {"id": "hdg_en_06", "text": "Perhaps I was wrong about him.", "lang": "EN"},
    {"id": "hdg_en_07", "text": "I might need to reconsider my decision.", "lang": "EN"},
    {"id": "hdg_en_08", "text": "It's probably my fault somehow.", "lang": "EN"},
    {"id": "hdg_en_09", "text": "Maybe things will get better eventually.", "lang": "EN"},
    {"id": "hdg_en_10", "text": "Perhaps I'm not seeing the whole picture, maybe I should ask.", "lang": "EN", "triple": True},
    # Japanese (10)
    {"id": "hdg_jp_01", "text": "その仕事に応募すべきかもしれない。", "lang": "JP"},
    {"id": "hdg_jp_02", "text": "たぶん、今が変わる時だ。", "lang": "JP"},
    {"id": "hdg_jp_03", "text": "過剰反応しているのかもしれない。", "lang": "JP"},
    {"id": "hdg_jp_04", "text": "思っているほど悪くないだろう。", "lang": "JP"},
    {"id": "hdg_jp_05", "text": "彼らはそういう意味じゃなかったのかもしれない。", "lang": "JP"},
    {"id": "hdg_jp_06", "text": "たぶん、彼について間違っていた。", "lang": "JP"},
    {"id": "hdg_jp_07", "text": "決断を考え直す必要があるかもしれない。", "lang": "JP"},
    {"id": "hdg_jp_08", "text": "どこかで私のせいなのだろう。", "lang": "JP"},
    {"id": "hdg_jp_09", "text": "いつかは良くなるかもしれない。", "lang": "JP"},
    {"id": "hdg_jp_10", "text": "全体像が見えていないのかもしれない。", "lang": "JP"},
]

# Paper 2 Table 2 reference values
PAPER2_REFERENCE = {
    "adversative": {"mean_S": 2.10, "mean_H": 1.037},
    "hedging": {"mean_S": 2.05, "mean_H": 1.002},
}


def run_experiment(verbose: bool = True) -> dict:
    """Run the full rule-based extraction experiment.

    Returns:
        Dict with results for each category and overall summary.
    """
    results = {}

    for cat_name, sentences in [
        ("adversative", ADVERSATIVE_SENTENCES),
        ("hedging", HEDGING_SENTENCES),
    ]:
        states = phi_batch(sentences, category=cat_name)
        state_dicts = []

        for sent, state in zip(sentences, states):
            entry = {
                "id": sent["id"],
                "text": sent["text"],
                "lang": sent["lang"],
                "|S|": state.state_size,
                "H(S)": round(state.entropy, 3),
                "interpretations": [i.to_dict() for i in state.interpretations],
            }
            state_dicts.append(entry)

        sizes = [s["|S|"] for s in state_dicts]
        entropies = [s["H(S)"] for s in state_dicts]
        mean_S = round(sum(sizes) / len(sizes), 2)
        mean_H = round(sum(entropies) / len(entropies), 3)

        results[cat_name] = {
            "states": state_dicts,
            "mean_S": mean_S,
            "mean_H": mean_H,
            "count": len(state_dicts),
        }

    # Overall
    all_S = []
    all_H = []
    for cat in results.values():
        all_S.extend(s["|S|"] for s in cat["states"])
        all_H.extend(s["H(S)"] for s in cat["states"])

    results["overall"] = {
        "total_sentences": len(all_S),
        "mean_S": round(sum(all_S) / len(all_S), 2),
        "mean_H": round(sum(all_H) / len(all_H), 3),
    }

    if verbose:
        _print_results(results)

    return results


def _print_results(results: dict):
    """Pretty-print experiment results."""
    print("=" * 70)
    print("RULE-BASED φ-MAPPING EXPERIMENT RESULTS")
    print("=" * 70)

    for cat_name in ["adversative", "hedging"]:
        cat = results[cat_name]
        print(f"\n{cat_name.upper()} ({cat['count']} sentences)")
        print("-" * 50)

        for state in cat["states"]:
            interp_str = " | ".join(
                f"{i['context']}:{i['weight']}" for i in state["interpretations"]
            )
            print(f"  {state['id']:12s}  |S|={state['|S|']}  H={state['H(S)']:.3f}  [{interp_str}]")

        print(f"\n  Mean |S|: {cat['mean_S']}")
        print(f"  Mean H(S): {cat['mean_H']}")

    # Comparison with Paper 2
    print("\n" + "=" * 70)
    print("VERIFICATION AGAINST PAPER 2 TABLE 2")
    print("=" * 70)
    print(f"\n{'Category':<15} {'Paper 2 |S|':>12} {'Actual |S|':>12} {'Match':>7}  "
          f"{'Paper 2 H':>12} {'Actual H':>12} {'Match':>7}")
    print("-" * 85)

    all_match = True
    for cat_name in ["adversative", "hedging"]:
        ref = PAPER2_REFERENCE[cat_name]
        act = results[cat_name]
        s_match = abs(ref["mean_S"] - act["mean_S"]) < 0.01
        h_match = abs(ref["mean_H"] - act["mean_H"]) < 0.01
        s_mark = "✓" if s_match else "✗"
        h_mark = "✓" if h_match else "✗"
        if not (s_match and h_match):
            all_match = False

        print(f"  {cat_name:<13} {ref['mean_S']:>12.2f} {act['mean_S']:>12.2f} {s_mark:>7}  "
              f"{ref['mean_H']:>12.3f} {act['mean_H']:>12.3f} {h_mark:>7}")

    print("-" * 85)
    if all_match:
        print("  ✓ All values match Paper 2 Table 2.")
    else:
        print("  ✗ Some values differ from Paper 2 Table 2.")

    print(f"\nOverall: {results['overall']['total_sentences']} sentences, "
          f"mean |S|={results['overall']['mean_S']}, "
          f"mean H={results['overall']['mean_H']}")


def main():
    parser = argparse.ArgumentParser(
        description="Run rule-based φ-mapping experiment (Paper 2)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save JSON results (optional)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    results = run_experiment(verbose=not args.quiet)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
