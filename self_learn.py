#!/usr/bin/env python3
"""
AI Text Optimizer v6.0 — Auto ML + Headless Detection

Generates ML variants, auto-detects with 3 detectors
(ZeroGPT + ContentDetector + Winston AI), records feedback, and iterates.

Usage:
    # Auto optimize (default)
    python self_learn.py gpt4_ai.txt --source-model gpt4

    # Manual feedback override
    python self_learn.py gpt4_ai.txt --feedback --variant variant_1.txt \
        --zerogpt 0.12 --winston 0.20 --contentdetector 0.15

Setup:
    pip install python-dotenv scikit-learn sentence-transformers playwright
    playwright install chromium
"""

import os
import sys
import signal
import json
import time
import random
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ml import TextQualityAnalyzer, StrategyQLearner
from detector import AIDetector

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
TARGET_SCORE = float(os.getenv("TARGET_SCORE", "0.30"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))
SIMILARITY_THRESHOLD = 0.70
TOP_N = 2
EARLY_STOP_PATIENCE = 2

# ── Graceful exit ────────────────────────────────────────────────────────────
_interrupted = False

def _on_sigint(sig, frame):
    global _interrupted
    print("\n⚠️  Ctrl+C — stopping after current detection...")
    _interrupted = True

signal.signal(signal.SIGINT, _on_sigint)


def text_similarity(original: str, rewritten: str) -> float:
    """Cosine similarity via TF-IDF."""
    try:
        vecs = TfidfVectorizer().fit_transform([original, rewritten])
        return cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
    except Exception:
        return 1.0


# ═════════════════════════════════════════════════════════════════════════════
# AUTO OPTIMIZATION LOOP
# ═════════════════════════════════════════════════════════════════════════════
def optimize(
    text: str,
    detector: AIDetector,
    source_model: str = "unknown",
) -> tuple:
    """Generate ML variants → detect → auto-feedback → iterate."""

    # ── Initial detection ────────────────────────────────────────────────
    print("\n🔍 Initial analysis...")
    init = detector.detect(text)
    init_score = init["ensemble"]

    if init_score < 0:
        print("❌ All detectors failed on initial text. Cannot proceed.")
        return text, -1.0, -1.0, []

    print(f"\n   Ensemble: {init_score:.1%}")
    if init_score <= TARGET_SCORE:
        print(f"\n✅ Already below target ({TARGET_SCORE:.0%})")
        return text, init_score, init_score, []

    # ── Initialize ML engine ─────────────────────────────────────────────
    analyzer = TextQualityAnalyzer(enable_learning=True)
    engine = analyzer.learning_engine
    q_agent = StrategyQLearner()
    domain, confidence = analyzer.detect_content_domain(text)
    analyzer.current_domain = domain

    print(f"   Domain: {domain} (confidence: {confidence:.2f})")
    print(f"   Q-Learning: epsilon={q_agent.epsilon:.2f}, "
          f"{len(q_agent.q_table)} states learned")
    print(f"   Target: ≤{TARGET_SCORE:.0%}")
    print(f"   Iterations: {MAX_ITERATIONS} (top {TOP_N} per iter)")

    if engine:
        stats = engine.get_learning_report()
        if stats["total_interactions"] > 0:
            print(f"   Learning: {stats['total_interactions']} past interactions, "
                  f"{stats['success_rate']:.0%} success rate")

    # ── Optimization loop ────────────────────────────────────────────────
    current = text
    best_text = text
    best_score = init_score
    history = []
    no_improve_streak = 0

    for i in range(MAX_ITERATIONS):
        if _interrupted:
            break

        print(f"\n{'─' * 55}")
        print(f"Iteration {i + 1}/{MAX_ITERATIONS} (best so far: {best_score:.1%})")

        # Generate 5 ML variants
        print(f"  🔧 Generating 5 ML variants...")
        variants = analyzer.generate_variants(current)

        # Q-Learning: order by expected value
        strategy_order = q_agent.select_strategy_order(domain, best_score, i, source_model)
        variant_map = {strategy: (var_text, metrics, strategy, changes)
                       for var_text, metrics, strategy, changes in variants}
        ordered_variants = [variant_map[s] for s in strategy_order if s in variant_map]
        seen_strategies = set(strategy_order)
        for v in variants:
            if v[2] not in seen_strategies:
                ordered_variants.append(v)

        is_exploring = random.random() < q_agent.epsilon
        print(f"  🎲 Q-Learning: {'EXPLORE (random)' if is_exploring else 'EXPLOIT → ' + strategy_order[0]}")

        # ── Similarity filter ─────────────────────────────────────────
        candidates = []
        for var_text, metrics, strategy, changes in ordered_variants:
            if not changes and var_text == current:
                print(f"    {strategy:12} — no changes, skipping")
                continue
            sim = text_similarity(text, var_text)
            if sim < SIMILARITY_THRESHOLD:
                print(f"    {strategy:12} — similarity {sim:.2f} < {SIMILARITY_THRESHOLD}, skipping")
                continue
            candidates.append((var_text, sim, strategy, changes))

        if not candidates:
            print(f"  📊 No viable candidates this iteration")
            history.append({
                "iteration": i + 1, "best_score": best_score,
                "strategy": "none", "improved": False,
            })
            continue

        # Sort by similarity DESC, take top-N
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:TOP_N]

        # Show candidates
        for var_text, sim, strategy, changes in top:
            print(f"    {strategy:12} — sim={sim:.2f}, {len(changes)} swaps")

        # ── Detect each candidate ────────────────────────────────────
        iter_best = None
        for var_text, sim, strategy, changes in top:
            if _interrupted:
                break

            print(f"\n  🌐 Detecting {strategy} ({len(changes)} swaps, sim={sim:.2f})...")
            result = detector.detect(var_text)
            score = result["ensemble"]

            if result["detector_count"] < 2:
                print(f"    ⚠️  Only {result['detector_count']} detector(s) — skipping")
                continue

            if score < 0:
                print(f"    ❌ Detection failed")
                continue

            improved = score < best_score
            delta = best_score - score
            print(f"    📊 Ensemble: {score:.1%} ({'▼' + f'{delta:.1%}' if improved else '▲ worse'})")

            # Q-Learning update
            q_agent.update(
                domain,
                best_score if not improved else (best_score + score) / 2,
                i, strategy, score, source_model,
            )

            # Auto-feedback to ML engine
            if engine:
                engine.record_user_feedback(text, var_text, improved, changes, domain)

            if improved:
                iter_best = (var_text, score, strategy, changes, sim)
                best_text = var_text
                best_score = score

            time.sleep(1)

        # Record iteration result
        if iter_best:
            no_improve_streak = 0
            current = iter_best[0]
            history.append({
                "iteration": i + 1,
                "score": iter_best[1],
                "best_score": best_score,
                "strategy": iter_best[2],
                "changes_count": len(iter_best[3]),
                "similarity": round(iter_best[4], 3),
                "improved": True,
            })
            print(f"\n  ✅ Best this iter: {iter_best[2]} → {iter_best[1]:.1%}")
        else:
            no_improve_streak += 1
            history.append({
                "iteration": i + 1,
                "best_score": best_score,
                "strategy": "none",
                "improved": False,
            })
            print(f"\n  📊 No improvement ({no_improve_streak}/{EARLY_STOP_PATIENCE})")

            if no_improve_streak >= EARLY_STOP_PATIENCE:
                print(f"\n⏹  Early stop — plateau detected after {i + 1} iterations")
                break

        if best_score <= TARGET_SCORE:
            print(f"\n🎉 Target reached at iteration {i + 1}!")
            break

        time.sleep(2)

    return best_text, init_score, best_score, history


# ═════════════════════════════════════════════════════════════════════════════
# FEEDBACK MODE — Manual detector score input
# ═════════════════════════════════════════════════════════════════════════════
def _handle_feedback(args):
    """Accept manual detector scores and feed them into Q-Learning + ML engine."""
    scores = {}
    if args.zerogpt is not None:
        scores["zerogpt"] = args.zerogpt
    if args.contentdetector is not None:
        scores["contentdetector"] = args.contentdetector
    if args.winston is not None:
        scores["winston"] = args.winston

    if not scores:
        print("❌ --feedback requires at least one detector score (--zerogpt, --winston, --contentdetector)")
        sys.exit(1)

    # Load original text
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            original = f.read().strip()
    except FileNotFoundError:
        print(f"❌ File not found: {args.file}")
        sys.exit(1)

    # Load variant text (if provided)
    variant = original
    if args.variant:
        try:
            with open(args.variant, "r", encoding="utf-8") as f:
                variant = f.read().strip()
        except FileNotFoundError:
            print(f"❌ Variant file not found: {args.variant}")
            sys.exit(1)

    avg_score = sum(scores.values()) / len(scores)

    print("=" * 55)
    print("  Manual Feedback Mode")
    print(f"  Source model: {args.source_model}")
    print(f"  Original:     {args.file}")
    if args.variant:
        print(f"  Variant:      {args.variant}")
    print(f"  Scores:       {', '.join(f'{k}: {v:.1%}' for k, v in scores.items())}")
    print(f"  Ensemble avg: {avg_score:.1%}")
    print("=" * 55)

    analyzer = TextQualityAnalyzer(enable_learning=True)
    engine = analyzer.learning_engine
    domain, confidence = analyzer.detect_content_domain(original)
    print(f"\n  Domain: {domain} (confidence: {confidence:.2f})")

    # Load variant metadata
    strategy = "unknown"
    changes = []
    meta_file = Path("variants_meta.json")
    if meta_file.exists() and args.variant:
        try:
            meta = json.loads(meta_file.read_text())
            variant_name = os.path.basename(args.variant)
            for entry in meta:
                if entry["file"] == variant_name:
                    strategy = entry.get("strategy", "unknown")
                    changes = entry.get("changes", [])
                    break
        except Exception:
            pass

    q_agent = StrategyQLearner()
    q_agent.update(domain, 1.0, 0, strategy, avg_score, args.source_model)
    print(f"  Q-Learning: updated ({domain}|1.0|0|{args.source_model}) strategy={strategy}")

    if args.variant and variant != original:
        sim = text_similarity(original, variant)
        print(f"  Similarity:   {sim:.2f}")
        if engine:
            improved = avg_score < 0.5
            change_tuples = [(c[0], c[1]) for c in changes] if changes else []
            engine.record_user_feedback(original, variant, improved, change_tuples, domain)
            print(f"  ML feedback:  {'approved' if improved else 'disapproved'} ({len(change_tuples)} swaps)")

    print(f"\n✅ Feedback recorded for source_model={args.source_model}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI Text Optimizer v6.0 — Auto ML + Headless Detection")
    parser.add_argument("file", nargs="?", default="gpt4_ai.txt", help="Input text file")
    parser.add_argument("--target", type=float, help="Target AI score (0.0-1.0)")
    parser.add_argument("--max-iter", type=int, default=5, help="Max iterations (default 5)")
    parser.add_argument("--top-n", type=int, help="Top-N variants to detect per iteration (default 2)")
    parser.add_argument(
        "--source-model", type=str, default="unknown",
        choices=["gpt4", "claude", "deepseek", "gemini", "unknown"],
        help="Which AI model generated the input text",
    )
    # ── Feedback mode ──
    parser.add_argument("--feedback", action="store_true", help="Manual feedback mode")
    parser.add_argument("--variant", type=str, help="Path to variant text file (for --feedback)")
    parser.add_argument("--zerogpt", type=float, help="Manual ZeroGPT score 0.0-1.0")
    parser.add_argument("--contentdetector", type=float, help="Manual ContentDetector score 0.0-1.0")
    parser.add_argument("--winston", type=float, help="Manual Winston AI score 0.0-1.0")
    args = parser.parse_args()

    global TARGET_SCORE, MAX_ITERATIONS, TOP_N
    if args.target is not None:
        TARGET_SCORE = args.target
    if args.max_iter is not None:
        MAX_ITERATIONS = args.max_iter
    if args.top_n is not None:
        TOP_N = args.top_n

    # ── Feedback mode ────────────────────────────────────────────────────
    if args.feedback:
        _handle_feedback(args)
        return

    # ── Load input ───────────────────────────────────────────────────────
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            original = f.read().strip()
    except FileNotFoundError:
        print(f"❌ File not found: {args.file}")
        sys.exit(1)

    # ── Banner ───────────────────────────────────────────────────────────
    print("=" * 55)
    print("  AI Text Optimizer v6.0 — Auto ML + Detection")
    print(f"  Detectors:   ZeroGPT + ContentDetector + Winston AI")
    if args.source_model != "unknown":
        print(f"  Source model: {args.source_model}")
    print(f"  Rewrite:     ML Word Selector (free)")
    print(f"  Iterations:  {MAX_ITERATIONS} (top {TOP_N} detected per iter)")
    print(f"  Target:      ≤{TARGET_SCORE:.0%}")
    print("=" * 55)

    words = len(original.split())
    print(f"📄 {args.file}: {len(original)} chars, {words} words")

    # ── Run optimization ─────────────────────────────────────────────────
    with AIDetector() as detector:
        best_text, init_score, final_score, history = optimize(
            original, detector,
            source_model=args.source_model,
        )

    # ── Final report ─────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("📊 FINAL REPORT")
    print(f"{'=' * 55}")
    if init_score >= 0:
        print(f"  Initial:     {init_score:.1%}")
        print(f"  Final:       {final_score:.1%}")
        print(f"  Improvement: {init_score - final_score:+.1%}")
    print(f"  Iterations:  {len(history)}")
    if final_score >= 0:
        print(f"  Target:      {'✅ REACHED' if final_score <= TARGET_SCORE else '❌ NOT REACHED'}")

    # Learning stats
    try:
        q = StrategyQLearner()
        q_report = q.get_q_report()
        print(f"  Q-Learning:  {q_report['states']} states, epsilon={q_report['epsilon']:.2f}")
        if q_report.get('strategy_wins'):
            wins_str = ", ".join(f"{k}: {v}" for k, v in q_report['strategy_wins'].items())
            print(f"  Best strats: {wins_str}")
    except Exception:
        pass
    try:
        a = TextQualityAnalyzer(enable_learning=True)
        stats = a.learning_engine.get_learning_report()
        print(f"  Learned:     {stats['total_interactions']} interactions, "
              f"{stats['success_rate']:.0%} success rate")
    except Exception:
        pass

    # Save output
    if best_text != original:
        out_file = "optimized_" + os.path.basename(args.file)
        Path(out_file).write_text(best_text, encoding="utf-8")
        print(f"\n💾 Output → {out_file}")

    if history:
        Path("history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Preview
    if best_text != original:
        print(f"\n{'─' * 55}")
        preview = best_text[:1200]
        print(preview + ("..." if len(best_text) > 1200 else ""))
        print(f"{'─' * 55}")


if __name__ == "__main__":
    main()
