#!/usr/bin/env python3
"""
AI Detector — Headless Playwright ensemble + API detectors.

Detectors: ZeroGPT + ContentDetector.ai + MyDetector.ai + Winston AI (API)
All run headless, no login required.

Usage:
    with AIDetector() as d:
        result = d.detect("some text here")
        print(result["ensemble"])  # 0.0-1.0
"""

import json
import os
import time
import re
import random
import requests
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout


class AIDetector:
    """Runs up to 4 AI detectors: 3 Playwright + 1 API (Winston)."""

    WEIGHTS = {
        "zerogpt": 0.30,
        "contentdetector": 0.20,
        "winston": 0.15,
        "mydetector": 0.35,
    }

    def __init__(self):
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._ctx = self._browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        self._last_known: dict = {}
        self._available_apis = {}
        print("  ✅ ZeroGPT (Playwright)")
        print("  ✅ ContentDetector (Playwright)")
        print("  ✅ MyDetector.ai (Playwright)")
        if os.getenv("WINSTON_API_KEY"):
            self._available_apis["winston"] = True
            print("  ✅ Winston AI (API)")
        else:
            print("  ❌ Winston AI — WINSTON_API_KEY not set")
        print("🌐 Browser ready (headless)")

    def _delay(self, lo=1.0, hi=2.5):
        time.sleep(random.uniform(lo, hi))

    # ── Public API ───────────────────────────────────────────────────────

    def detect(self, text: str) -> dict:
        """Run all detectors, return ensemble result."""
        results = {}

        for name, fn in [
            ("zerogpt", self._detect_zerogpt),
            ("contentdetector", self._detect_contentdetector),
            ("mydetector", self._detect_mydetector),
            ("winston", self._detect_winston),
        ]:
            if name == "winston" and "winston" not in self._available_apis:
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
                    print(f"  ⚠️  {name} timeout — using last: {self._last_known[name]:.0%}")
                else:
                    print(f"  ❌ {name} failed: {str(e)[:100]}")

            self._delay(1, 3)

        count = len(results)
        if count < 1:
            print("  ⚠️  Unreliable — no detectors responded")

        if results:
            tw = sum(self.WEIGHTS[k] for k in results)
            ensemble = sum(results[k] * self.WEIGHTS[k] for k in results) / tw
        else:
            ensemble = -1.0

        return {
            "ensemble": round(ensemble, 4) if ensemble >= 0 else -1.0,
            "zerogpt": results.get("zerogpt"),
            "contentdetector": results.get("contentdetector"),
            "mydetector": results.get("mydetector"),
            "winston": results.get("winston"),
            "detector_count": count,
        }

    def close(self):
        try:
            self._ctx.close()
            self._browser.close()
            self._pw.stop()
            print("🌐 Browser closed")
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ── Helper ───────────────────────────────────────────────────────────

    def _find_score(self, data, keys=None):
        """Recursively search JSON for an AI score field."""
        if keys is None:
            keys = [
                "fakePercentage", "fake_percentage", "ai_percentage",
                "is_gpt_generated_percentage", "isGPTGenerated",
                "aiPercentage", "ai_score", "ai_generated_percentage",
                "overall_score",
            ]
        if isinstance(data, dict):
            for k in keys:
                if k in data:
                    v = data[k]
                    if isinstance(v, (int, float)):
                        return v / 100.0 if v > 1.0 else v
            for v in data.values():
                r = self._find_score(v, keys)
                if r is not None:
                    return r
        elif isinstance(data, list):
            for item in data:
                r = self._find_score(item, keys)
                if r is not None:
                    return r
        return None

    # ── ZeroGPT ──────────────────────────────────────────────────────────

    def _detect_zerogpt(self, text: str):
        page = self._ctx.new_page()
        captured = []

        def on_resp(resp):
            try:
                ct = resp.headers.get("content-type", "")
                if resp.status == 200 and "json" in ct and "zerogpt" in resp.url:
                    captured.append(resp.json())
            except Exception:
                pass

        page.on("response", on_resp)

        try:
            page.goto("https://www.zerogpt.com", wait_until="domcontentloaded", timeout=20000)
            self._delay(1, 2)

            # Dismiss popups
            try:
                page.locator("button:has-text('Accept'), button:has-text('Got it')").first.click(timeout=3000)
            except Exception:
                pass

            # Fill textarea
            ta = page.locator("textarea").first
            ta.wait_for(timeout=10000)
            ta.fill(text[:5000])
            self._delay(0.5, 1.5)

            # Click detect
            btn = page.locator("button").filter(has_text=re.compile(r"detect|check|get result", re.I)).first
            btn.click()

            # Wait for API response
            checked = 0
            deadline = time.time() + 25
            while time.time() < deadline:
                while checked < len(captured):
                    score = self._find_score(captured[checked])
                    if score is not None:
                        return score
                    checked += 1
                page.wait_for_timeout(500)

            return None
        finally:
            page.close()

    # ── ContentDetector.ai ───────────────────────────────────────────────

    def _detect_contentdetector(self, text: str):
        page = self._ctx.new_page()
        captured = []

        def on_resp(resp):
            try:
                ct = resp.headers.get("content-type", "")
                if resp.status == 200 and "json" in ct and "contentdetector" in resp.url:
                    captured.append(resp.json())
            except Exception:
                pass

        page.on("response", on_resp)

        try:
            page.goto("https://contentdetector.ai", wait_until="domcontentloaded", timeout=20000)
            self._delay(1, 2)

            try:
                page.locator("button:has-text('Accept'), button:has-text('Got it')").first.click(timeout=3000)
            except Exception:
                pass

            # ContentDetector uses #editor-content textarea
            ta = page.locator("#editor-content, textarea").first
            ta.wait_for(timeout=10000)
            ta.fill(text[:5000])
            self._delay(0.5, 1.5)

            # Button says "Scan"
            btn = page.locator("button").filter(
                has_text=re.compile(r"scan|detect|check|analyze|submit", re.I)
            ).first
            btn.click()

            # Method 1: API interception
            checked = 0
            deadline = time.time() + 20
            while time.time() < deadline:
                while checked < len(captured):
                    score = self._find_score(captured[checked])
                    if score is not None:
                        return score
                    checked += 1
                page.wait_for_timeout(500)

            # Method 2: DOM scraping — look for probability score
            try:
                page.wait_for_timeout(3000)
                for sel in ["[class*='probability']", "[class*='score']", "[class*='response']"]:
                    try:
                        el = page.locator(sel).first
                        txt = el.inner_text(timeout=2000)
                        match = re.search(r"(\d+(?:\.\d+)?)\s*%", txt)
                        if match:
                            val = float(match.group(1))
                            return val / 100.0 if val > 1.0 else val
                    except Exception:
                        continue
            except Exception:
                pass

            return None
        finally:
            page.close()

    # ── MyDetector.ai ─────────────────────────────────────────────────

    def _detect_mydetector(self, text: str):
        """Detect via mydetector.ai — Playwright + API interception."""
        page = self._ctx.new_page()
        captured = []

        def on_resp(resp):
            try:
                ct = resp.headers.get("content-type", "")
                if resp.status == 200 and "json" in ct:
                    data = resp.json()
                    # Match: {"code": 100000, "result": {"output": {"sentences": [...]}}}
                    if isinstance(data, dict) and data.get("code") == 100000:
                        result = data.get("result", {})
                        output = result.get("output", {})
                        sentences = output.get("sentences", [])
                        if sentences:
                            captured.append(sentences)
            except Exception:
                pass

        page.on("response", on_resp)

        try:
            page.goto("https://mydetector.ai/", wait_until="domcontentloaded", timeout=20000)
            self._delay(1, 2)

            # Dismiss cookie/popup banners
            try:
                page.locator("button:has-text('Accept'), button:has-text('Got it')").first.click(timeout=3000)
            except Exception:
                pass

            # Fill contenteditable div via JS (no textarea on this site)
            ce = page.locator("[contenteditable='true']").first
            ce.wait_for(timeout=10000)
            ce.click()
            ce.evaluate(
                '(el, txt) => { el.innerText = txt; el.dispatchEvent(new Event("input", {bubbles: true})); }',
                text[:5000],
            )
            self._delay(0.5, 1.5)

            # "Detect Text" is button index 7 in the main content area
            # Use exact text match to avoid FAQ buttons lower on page
            btn = page.locator("button:has-text('Detect Text')").first
            btn.scroll_into_view_if_needed()
            btn.click()

            # Wait for API response with sentences
            deadline = time.time() + 30
            while time.time() < deadline:
                if captured:
                    sentences = captured[0]
                    scores = [s["score"] for s in sentences if "score" in s]
                    if scores:
                        return sum(scores) / len(scores)
                page.wait_for_timeout(500)

            return None
        finally:
            page.close()

    # ── Winston AI (API) ──────────────────────────────────────────────

    def _detect_winston(self, text: str):
        """Detect via Winston AI — pure HTTP API."""
        api_key = os.getenv("WINSTON_API_KEY", "")
        if not api_key:
            raise ValueError("WINSTON_API_KEY not set")
        resp = requests.post(
            "https://api.gowinston.ai/v2/ai-content-detection",
            json={"text": text[:5000], "sentences": False, "version": "latest"},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        human_score = data["score"]  # 0-100, high = human
        return 1.0 - (human_score / 100.0)  # invert to AI score
