#!/usr/bin/env python3
"""
AI Detector — All-Local GPU Ensemble (no Playwright, no APIs).

Detectors (all local, GPU-accelerated):
  1. Binoculars  — Falcon-7B × 2 (4-bit) perplexity ratio
  2. GPT-2 PPL   — GPTZero-style perplexity scoring
  3. GLTR        — top-k token distribution analysis

Usage:
    with AIDetector() as d:
        result = d.detect("some text here")
        print(result["ensemble"])  # 0.0-1.0
"""

import math
import time


class AIDetector:
    """All-local GPU AI detection ensemble — 3 detectors."""

    WEIGHTS = {
        "binoculars": 0.35,
        "gpt2_ppl": 0.35,
        "gltr": 0.30,
    }

    def __init__(self, use_binoculars: bool = True):
        self._available = {}
        self._binoculars = None
        self._gpt2 = None
        self._last_known = {}

        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            vram_gb = gpu.total_memory / 1024**3
            print(f"  GPU: {gpu.name} ({vram_gb:.0f}GB VRAM)")
        else:
            print("  ⚠️  No CUDA GPU — models will run on CPU (slow)")

        t0 = time.time()

        if use_binoculars:
            self._init_binoculars()
        self._init_gpt2()

        elapsed = time.time() - t0
        active = sum(1 for v in self._available.values() if v)
        vram_used = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
        print(f"  {active}/{len(self.WEIGHTS)} detectors active ({elapsed:.1f}s, {vram_used:.1f}GB VRAM)")

    # ── Model loading ─────────────────────────────────────────────────────

    def _init_binoculars(self):
        """Load Binoculars Falcon-7B detector on GPU (4-bit quantized)."""
        try:
            import torch

            if not torch.cuda.is_available():
                print("  ❌ Binoculars — no CUDA GPU")
                return

            print(f"  ⏳ Binoculars (Falcon-7B × 2, 4-bit)...")

            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

            observer = AutoModelForCausalLM.from_pretrained(
                "tiiuae/falcon-7b",
                quantization_config=bnb_config,
                device_map="auto",
            )
            performer = AutoModelForCausalLM.from_pretrained(
                "tiiuae/falcon-7b-instruct",
                quantization_config=bnb_config,
                device_map="auto",
            )
            observer.eval()
            performer.eval()

            tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token

            self._binoculars = {
                "observer": observer,
                "performer": performer,
                "tokenizer": tokenizer,
                "threshold": 0.8536432310785527,
                "max_tokens": 512,
            }
            self._available["binoculars"] = True

            vram_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  ✅ Binoculars ({vram_used:.1f}GB)")
        except Exception as e:
            print(f"  ❌ Binoculars failed: {str(e)[:200]}")

    def _init_gpt2(self):
        """Load GPT-2 for perplexity + GLTR detectors (shared model)."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"  ⏳ GPT-2 (perplexity + GLTR)...")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                "openai-community/gpt2",
                torch_dtype=dtype,
            ).to(device)
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token

            self._gpt2 = {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "max_tokens": 1024,
            }
            self._available["gpt2_ppl"] = True
            self._available["gltr"] = True

            vram_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  ✅ GPT-2 ({vram_used:.1f}GB)")
        except Exception as e:
            print(f"  ❌ GPT-2 failed: {str(e)[:200]}")

    # ── Public API ────────────────────────────────────────────────────────

    def detect(self, text: str) -> dict:
        """Run all detectors, return ensemble result."""
        results = {}

        for name, fn in [
            ("binoculars", self._detect_binoculars),
            ("gpt2_ppl", self._detect_gpt2_ppl),
            ("gltr", self._detect_gltr),
        ]:
            if name not in self._available:
                continue

            print(f"  🔍 {name}...")
            try:
                score = fn(text)
                if score is not None and 0.0 <= score <= 1.0:
                    results[name] = round(score, 4)
                    self._last_known[name] = results[name]
                    print(f"  ✅ {name}: {results[name]:.0%}")
                else:
                    raise ValueError(f"bad score: {score}")
            except Exception as e:
                if name in self._last_known:
                    results[name] = self._last_known[name]
                    print(f"  ⚠️  {name} error — using last: {self._last_known[name]:.0%}")
                else:
                    print(f"  ❌ {name} failed: {str(e)[:100]}")

        count = len(results)
        if count < 1:
            print("  ⚠️  No detectors responded")

        if results:
            tw = sum(self.WEIGHTS[k] for k in results)
            ensemble = sum(results[k] * self.WEIGHTS[k] for k in results) / tw
        else:
            ensemble = -1.0

        return {
            "ensemble": round(ensemble, 4) if ensemble >= 0 else -1.0,
            "binoculars": results.get("binoculars"),
            "gpt2_ppl": results.get("gpt2_ppl"),
            "gltr": results.get("gltr"),
            "detector_count": count,
        }

    def close(self):
        """Release GPU memory."""
        import torch

        if self._binoculars:
            del self._binoculars["observer"]
            del self._binoculars["performer"]
            self._binoculars = None
        if self._gpt2:
            del self._gpt2["model"]
            self._gpt2 = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  GPU memory released")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ── Binoculars (Falcon-7B 4-bit) ─────────────────────────────────────

    def _detect_binoculars(self, text: str):
        """Binoculars: perplexity / cross-perplexity ratio."""
        if self._binoculars is None:
            return None

        import torch

        bino = self._binoculars
        tokenizer = bino["tokenizer"]
        observer = bino["observer"]
        performer = bino["performer"]
        threshold = bino["threshold"]

        encodings = tokenizer(
            text[:3000],
            return_tensors="pt",
            truncation=True,
            max_length=bino["max_tokens"],
            return_token_type_ids=False,
        ).to(observer.device)

        with torch.inference_mode():
            observer_logits = observer(**encodings).logits
            performer_logits = performer(**encodings).logits

        labels = encodings["input_ids"][:, 1:]
        logits_shifted = performer_logits[:, :-1]
        log_probs = torch.nn.functional.log_softmax(logits_shifted, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        if "attention_mask" in encodings:
            mask = encodings["attention_mask"][:, 1:].float()
            token_log_probs = token_log_probs * mask
            ppl = torch.exp(-token_log_probs.sum(-1) / mask.sum(-1))
        else:
            ppl = torch.exp(-token_log_probs.mean(-1))

        obs_log_probs = torch.nn.functional.log_softmax(observer_logits[:, :-1], dim=-1)
        perf_probs = torch.nn.functional.softmax(
            performer_logits[:, :-1].to(observer_logits.device), dim=-1
        )
        cross_entropy = -(perf_probs * obs_log_probs).sum(-1)

        if "attention_mask" in encodings:
            cross_entropy = cross_entropy * mask
            x_ppl = torch.exp(cross_entropy.sum(-1) / mask.sum(-1))
        else:
            x_ppl = torch.exp(cross_entropy.mean(-1))

        bino_score = (ppl / x_ppl).item()
        ai_prob = 1.0 / (1.0 + math.exp(12.0 * (bino_score - threshold)))
        return max(0.0, min(1.0, ai_prob))

    # ── GPT-2 Perplexity (GPTZero-style) ──────────────────────────────────

    def _detect_gpt2_ppl(self, text: str):
        """Low perplexity under GPT-2 = likely AI-generated."""
        if self._gpt2 is None:
            return None

        import torch

        g = self._gpt2
        model = g["model"]
        tokenizer = g["tokenizer"]

        encodings = tokenizer(
            text[:3000],
            return_tensors="pt",
            truncation=True,
            max_length=g["max_tokens"],
        ).to(g["device"])

        with torch.inference_mode():
            outputs = model(**encodings, labels=encodings["input_ids"])
            ppl = torch.exp(outputs.loss).item()

        threshold = 60.0
        steepness = 0.08
        ai_prob = 1.0 / (1.0 + math.exp(steepness * (ppl - threshold)))
        return max(0.0, min(1.0, ai_prob))

    # ── GLTR Log-likelihood ───────────────────────────────────────────────

    def _detect_gltr(self, text: str):
        """GLTR: fraction of tokens in GPT-2's top-k predictions."""
        if self._gpt2 is None:
            return None

        import torch

        g = self._gpt2
        model = g["model"]
        tokenizer = g["tokenizer"]

        encodings = tokenizer(
            text[:3000],
            return_tensors="pt",
            truncation=True,
            max_length=g["max_tokens"],
        ).to(g["device"])

        input_ids = encodings["input_ids"]

        with torch.inference_mode():
            logits = model(**encodings).logits

        pred_logits = logits[:, :-1]
        actual_tokens = input_ids[:, 1:]

        seq_len = actual_tokens.shape[1]
        if seq_len < 5:
            return None

        top_k = 10
        top_50 = 50

        _, top_k_idx = pred_logits.topk(top_k, dim=-1)
        _, top_50_idx = pred_logits.topk(top_50, dim=-1)

        actual_expanded = actual_tokens.unsqueeze(-1)
        in_top_k = (top_k_idx == actual_expanded).any(dim=-1).float()
        in_top_50 = (top_50_idx == actual_expanded).any(dim=-1).float()

        frac_top_k = in_top_k.mean().item()
        frac_top_50 = in_top_50.mean().item()

        gltr_score = 0.6 * frac_top_k + 0.4 * frac_top_50

        threshold = 0.65
        steepness = 15.0
        ai_prob = 1.0 / (1.0 + math.exp(-steepness * (gltr_score - threshold)))
        return max(0.0, min(1.0, ai_prob))
