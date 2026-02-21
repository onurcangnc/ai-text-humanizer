#!/usr/bin/env python3
"""
Training Data Generator — Generate AI text samples via multiple LLMs.

Topics generated dynamically with cross-model rotation:
  GPT-4.5 suggests topics → Claude writes them
  Claude Sonnet 4.6 suggests topics → DeepSeek writes them
  DeepSeek suggests topics → GPT-4.5 writes them

State persists across runs — topics never repeat.

Usage:
    python generate_training_data.py --max-texts 10 --train
    python generate_training_data.py --max-texts 3 --models claude --generate-only
    python generate_training_data.py --max-texts 20 --topics-per-batch 5 --train
"""

import os
import sys
import json
import random
import argparse
import re
from pathlib import Path
from itertools import cycle
from dotenv import load_dotenv

load_dotenv()

# ── Fallback seed topics (used only if no LLM can generate topics) ────────────
SEED_TOPICS = [
    "climate change policy", "neural networks in medicine",
    "cold war geopolitics", "quantum computing fundamentals",
    "economic inequality", "CRISPR gene editing ethics",
    "social media and democracy", "nuclear energy policy",
    "antibiotic resistance", "space colonization",
    "cryptocurrency regulation", "urban planning",
    "artificial general intelligence risks", "vaccine hesitancy",
    "biodiversity loss", "digital privacy rights",
    "automation and unemployment", "water scarcity",
    "misinformation spread", "renewable energy transition",
]

SYSTEM_PROMPT = "You are an academic writer. Write clear, formal prose."

TEXT_PROMPT = (
    "Write a 350-word academic analysis about: {topic}\n"
    "Use formal language and structured paragraphs.\n"
    "Do not use bullet points or headers.\n"
    "Write in continuous prose only."
)

TOPIC_PROMPT = (
    "Suggest {n} diverse academic topics for AI text generation testing.\n"
    "Already used topics: {used}\n\n"
    "Requirements:\n"
    "- Must be completely different from used topics\n"
    "- Cover varied domains: science, history, economics,\n"
    "  politics, ethics, technology, environment, medicine, law, philosophy\n"
    "- Each topic: 3-7 words\n"
    "- Return ONLY a JSON array of strings, nothing else\n"
    'Example: ["quantum computing ethics", "medieval trade routes"]'
)

# ── Model configs ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "gpt4": {
        "env_key": "OPENAI_API_KEY",
        "model_id": "gpt-4o",
        "source_model": "gpt4",
    },
    "claude": {
        "env_key": "ANTHROPIC_API_KEY",
        "model_id": "claude-sonnet-4-6",
        "source_model": "claude",
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "model_id": "deepseek-chat",
        "source_model": "deepseek",
    },
}

# Topic generator rotation: each model suggests topics for another
# Index 0: gpt4 suggests → for claude's texts
# Index 1: claude suggests → for deepseek's texts
# Index 2: deepseek suggests → for gpt4's texts
ROTATION_ORDER = ["gpt4", "claude", "deepseek"]


def topic_slug(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    return slug[:40]


# ── State persistence ─────────────────────────────────────────────────────────

def load_state(out_dir: Path) -> dict:
    state_file = out_dir / "state.json"
    if state_file.exists():
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            # Ensure all keys exist
            defaults = {
                "used_topics": [],
                "topic_generator_rotation": 0,
                "generated_files": [],
                "trained_files": [],
                "total_trained": 0,
                "total_generated": 0,
            }
            for k, v in defaults.items():
                data.setdefault(k, v)
            return data
        except Exception:
            pass
    return {
        "used_topics": [],
        "topic_generator_rotation": 0,
        "generated_files": [],
        "trained_files": [],
        "total_trained": 0,
        "total_generated": 0,
    }


def save_state(state: dict, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    state_file = out_dir / "state.json"
    state_file.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── API callers ───────────────────────────────────────────────────────────────

def call_openai(prompt: str, api_key: str, system: str = None) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=600,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def call_anthropic(prompt: str, api_key: str, system: str = None) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    kwargs = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 600,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content[0].text.strip()


def call_deepseek(prompt: str, api_key: str, system: str = None) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=600,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


CALLERS = {
    "gpt4": call_openai,
    "claude": call_anthropic,
    "deepseek": call_deepseek,
}


# ── Dynamic topic generation with rotation ────────────────────────────────────

def get_next_topics(
    used_topics: list,
    generator_model: str,
    api_key: str,
    n: int = 5,
) -> list:
    """Ask a specific model to suggest n new topics."""

    used_str = ", ".join(used_topics[-30:]) if used_topics else "none"
    prompt = TOPIC_PROMPT.format(n=n, used=used_str)

    caller = CALLERS[generator_model]
    label = MODEL_CONFIGS[generator_model]["model_id"]

    try:
        raw = caller(prompt, api_key)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON array in response")
        topics = json.loads(match.group())
        used_lower = {t.lower() for t in used_topics}
        fresh = [t for t in topics if isinstance(t, str) and t.lower() not in used_lower]
        if fresh:
            return fresh[:n]
        raise ValueError("All suggested topics were duplicates")
    except Exception as e:
        print(f"    {label} topic generation failed: {str(e)[:120]}")
        return []


def get_fallback_topics(used_topics: list, n: int) -> list:
    """Pick from seed list when LLM topic generation fails."""
    used_lower = {t.lower() for t in used_topics}
    remaining = [t for t in SEED_TOPICS if t.lower() not in used_lower]
    if remaining:
        pick = random.sample(remaining, min(n, len(remaining)))
        print(f"    Fallback seed topics: {pick}")
        return pick
    fallback = [f"academic analysis topic {len(used_topics) + i + 1}" for i in range(n)]
    print(f"    All seeds exhausted, using generic: {fallback}")
    return fallback


def get_rotation_model(state: dict, available_models: list) -> str:
    """Get the next topic-generator model from rotation."""
    idx = state["topic_generator_rotation"] % len(ROTATION_ORDER)
    # Walk rotation until we find an available model
    for offset in range(len(ROTATION_ORDER)):
        model = ROTATION_ORDER[(idx + offset) % len(ROTATION_ORDER)]
        if model in available_models:
            return model
    return available_models[0]


def advance_rotation(state: dict):
    state["topic_generator_rotation"] = (
        state["topic_generator_rotation"] + 1
    ) % len(ROTATION_ORDER)


# ── Detect available models ───────────────────────────────────────────────────

def detect_available_models(requested: list) -> dict:
    """Return {model_name: api_key} for models with keys present."""
    available = {}
    for model in requested:
        cfg = MODEL_CONFIGS.get(model)
        if not cfg:
            print(f"    Unknown model: {model}")
            continue
        key = os.getenv(cfg["env_key"], "")
        if key:
            available[model] = key
            print(f"    {model:10} {cfg['model_id']:24} {cfg['env_key']} found")
        else:
            print(f"    {model:10} {cfg['env_key']} NOT SET — skipping")
    return available


# ── Text generation ───────────────────────────────────────────────────────────

def generate_text(topic: str, model: str, api_key: str) -> str:
    """Generate academic text about a topic using the specified model."""
    prompt = TEXT_PROMPT.format(topic=topic)
    return CALLERS[model](prompt, api_key, system=SYSTEM_PROMPT)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Training Data Generator — dynamic topics + multi-LLM"
    )
    parser.add_argument(
        "--max-texts", type=int, default=10,
        help="Total texts to generate (default: 10)",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated: gpt4,claude,deepseek (default: all available)",
    )
    parser.add_argument(
        "--topics-per-batch", type=int, default=5,
        help="New topics to generate at a time (default: 5)",
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Only generate texts, skip training",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Auto-run self_learn.py on each generated file",
    )
    parser.add_argument(
        "--max-iter", type=int, default=5,
        help="Max self_learn.py iterations per file (default: 5)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="training_data",
        help="Output directory (default: training_data)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    print("=" * 55)
    print("  Training Data Generator")
    print("=" * 55)

    # Load state
    state = load_state(out_dir)
    if state["total_generated"] > 0:
        print(f"\n  Resuming: {len(state['used_topics'])} topics used, "
              f"{state['total_generated']} generated, "
              f"{state['total_trained']} trained")

    # Available models
    if args.models:
        requested = [m.strip() for m in args.models.split(",")]
    else:
        requested = ["gpt4", "claude", "deepseek"]

    print(f"\n  Checking API keys...")
    available = detect_available_models(requested)

    if not available:
        print("\n  No API keys found. Set at least one in .env:")
        print("    OPENAI_API_KEY=sk-...")
        print("    ANTHROPIC_API_KEY=sk-ant-...")
        print("    DEEPSEEK_API_KEY=sk-...")
        sys.exit(1)

    model_names = list(available.keys())
    model_cycle = cycle(model_names)

    max_texts = args.max_texts
    batch_size = args.topics_per_batch
    generated_this_run = 0
    available_topics = []

    print(f"\n  Models:    {', '.join(model_names)}")
    print(f"  Max texts: {max_texts}")
    print(f"  Batch:     {batch_size} topics per batch")
    print(f"  Mode:      {'generate-only' if args.generate_only else 'generate + train' if args.train else 'generate'}")

    # ── Main loop ─────────────────────────────────────────────────────────
    consecutive_failures = 0
    while generated_this_run < max_texts:
        # Refill topics if empty
        if not available_topics:
            gen_model = get_rotation_model(state, model_names)
            gen_label = MODEL_CONFIGS[gen_model]["model_id"]
            print(f"\n  Generating {batch_size} topics via {gen_label}...")

            new_topics = get_next_topics(
                state["used_topics"],
                gen_model,
                available[gen_model],
                n=batch_size,
            )

            if not new_topics:
                new_topics = get_fallback_topics(state["used_topics"], batch_size)

            if not new_topics:
                print("  No more topics available. Stopping.")
                break

            print(f"  Topics: {new_topics}")

            for t in new_topics:
                if t.lower() not in {u.lower() for u in state["used_topics"]}:
                    state["used_topics"].append(t)
            available_topics.extend(new_topics)
            advance_rotation(state)
            save_state(state, out_dir)

        # Pick next topic + model
        topic = available_topics.pop(0)
        model = next(model_cycle)
        slug = topic_slug(topic)
        filename = f"{model}_{slug}.txt"
        filepath = out_dir / filename

        # Skip if already generated
        if filename in state["generated_files"]:
            print(f"\n  {filename} — already exists, skipping")
            continue

        # Generate text
        count_label = f"[{generated_this_run + 1}/{max_texts}]"
        print(f"\n  {count_label} [{model}] Writing about: {topic}")

        out_dir.mkdir(exist_ok=True)
        try:
            text = generate_text(topic, model, available[model])
            filepath.write_text(text, encoding="utf-8")
            word_count = len(text.split())
            char_count = len(text)

            print(f"  Generated: {filename} ({word_count} words, {char_count} chars)")

            state["generated_files"].append(filename)
            state["total_generated"] += 1
            generated_this_run += 1
            consecutive_failures = 0
            save_state(state, out_dir)

            # Train immediately if requested
            if args.train and not args.generate_only:
                source = MODEL_CONFIGS[model]["source_model"]
                print(f"  Training ({state['total_trained'] + 1})...")
                cmd = (
                    f"{sys.executable} self_learn.py \"{filepath}\" "
                    f"--source-model {source} --max-iter {args.max_iter}"
                )
                ret = os.system(cmd)
                if ret == 0:
                    state["trained_files"].append(filename)
                    state["total_trained"] += 1
                    save_state(state, out_dir)
                else:
                    print(f"  Training failed (exit code {ret})")

        except Exception as e:
            print(f"  FAILED: {str(e)[:150]}")
            consecutive_failures += 1
            if consecutive_failures >= 5:
                print(f"\n  5 consecutive failures — stopping. Check API keys and packages.")
                break

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("  REPORT")
    print(f"{'=' * 55}")
    print(f"  Generated this run: {generated_this_run}")
    print(f"  Total generated:    {state['total_generated']}")
    print(f"  Total trained:      {state['total_trained']}")
    print(f"  Topics used ever:   {len(state['used_topics'])}")
    print(f"{'=' * 55}")

    # Preview generated files
    success_files = [
        f for f in state["generated_files"][-generated_this_run:]
        if (out_dir / f).exists()
    ]
    if success_files:
        print(f"\n  Previews:")
        for fname in success_files:
            text = (out_dir / fname).read_text(encoding="utf-8")
            preview = text[:200].replace("\n", " ")
            print(f"\n  [{fname}]")
            print(f"  {preview}...")

    if generated_this_run > 0 and not args.train and not args.generate_only:
        print(f"\n  To train on generated files:")
        print(f"    python generate_training_data.py --max-texts {max_texts} --train")

    print(f"\n  Done. Total generated: {state['total_generated']}")


if __name__ == "__main__":
    main()
