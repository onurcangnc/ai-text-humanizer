# I Tested 5 AI Detectors on the Same Text — The Results Were Shocking

I ran the same university-level essay through five different AI detectors. One said it was 0% AI-generated. Another said 100%. Same exact text, same exact words, wildly different verdicts.

This isn't a hypothetical. It happened on a real optimized essay about Silk Road currency exchange systems — and it made me question everything I thought I knew about AI detection.

---

## The Experiment

I'm a cybersecurity researcher and computer science student. Over the past several weeks, I built a pipeline that transforms AI-generated academic essays into human-sounding text. Not to cheat on assignments — but to systematically test how reliable AI detectors actually are.

Here's what I did:

I used three AI models — GPT-4, Claude, and DeepSeek — to generate 46 academic essays on topics ranging from deep sea biodiversity to Weimar Republic hyperinflation. Each essay was 400-600 words, written in a standard university style.

Then I processed them through a hybrid humanization pipeline. The pipeline works in three phases: first, it rewrites the text using a *different* AI model (so GPT-4's text gets rewritten by Claude, Claude's by DeepSeek, and so on). The rewrite prompt forces contractions, informal language, varied sentence lengths, and the kind of small imperfections real students make. Second, it swaps out "AI-sounding" vocabulary — words like "Furthermore" become "On top of that," and "crucial" becomes "major." It also injects sentence-length variation and subtle hedging phrases. Third, a quality filter makes sure the meaning and facts are preserved.

I optimized 30 texts across three training batches, measuring detection scores before and after. On average, the pipeline reduced local detection scores by 11.6 percentage points.

Then I took 10 of these optimized texts and submitted them to five commercial AI detectors: GPTZero, ZeroGPT, Originality.ai, Winston AI, and QuillBot's AI detector.

I half-expected the detectors to agree. They didn't. Not even close.

---

## The Results

Here's what each detector said about the same 10 optimized texts:

**GPTZero:** 100% AI on every single text. Ten out of ten. No exceptions.

**Originality.ai:** Also 100% AI on every single text. Identical to GPTZero.

**Winston AI:** Average 80.8%. But wildly inconsistent — it scored one text (ocean acidification) at just 13% AI, while scoring another (refugee quota) at 100%.

**ZeroGPT:** Average 20.8%. It classified most texts as predominantly human. Four texts scored below 10%.

**QuillBot:** Average 24.9%. Similar to ZeroGPT — most texts classified as human or mixed.

Let that sink in. The same optimized texts that GPTZero called 100% AI, ZeroGPT called 79% human. The disagreement isn't small — it's total.

[Figure 1: External detector heatmap showing 10 texts scored by 5 detectors. Red = 100% AI, Green = 0% AI. The visual inconsistency is striking. See `graph7_external_detector_heatmap.png`]

---

## The Silk Road Problem

The most dramatic example was an essay about Silk Road currency exchange systems. Here's how each detector scored it:

- **ZeroGPT:** 0% AI (fully human)
- **QuillBot:** 0% AI (fully human)
- **Winston AI:** 16% AI (mostly human)
- **GPTZero:** 100% AI (fully artificial)
- **Originality.ai:** 100% AI (fully artificial)

Same text. Five detectors. Verdicts ranging from "definitely human" to "definitely AI." If a student submitted this essay and their professor used ZeroGPT, they'd be fine. If the professor used GPTZero, they'd face an academic integrity investigation.

This isn't an edge case. Across all 10 texts, the pairwise agreement rate between detector pairs ranged from 10% to 100%. The local detectors (which I built using open-source models) agreed with each other 80-100% of the time. But they agreed with ZeroGPT only 30% of the time, and with QuillBot only 10% of the time.

[Figure 2: Pairwise detector agreement matrix. Local detectors cluster tightly (80-100%), but cross-cluster agreement drops to 10-30%. See `graph8_detector_agreement.png`]

---

## Why Do They Disagree?

After running a correlation analysis between my local detectors and the five commercial ones, I found something that explains the disagreement: there are two fundamentally different types of AI detectors, and they measure completely different things.

**Type 1: Perplexity-based detectors.** These tools — including Binoculars (the strongest component of my local ensemble), GLTR, and Winston AI — analyze how *predictable* each word is. AI text tends to be very predictable: each word is the statistically most likely next word given the context. Human text is messier — we use unexpected words, unusual phrasings, and unpredictable structures. These detectors are mathematically grounded and hard to fool without degrading the text.

**Type 2: Classifier-based detectors.** ZeroGPT and QuillBot appear to use machine learning classifiers trained on features like: Does the text use contractions? How varied are the sentence lengths? Are there informal markers? These classifiers learned that AI text tends to be formal, uniform in structure, and devoid of contractions. When you add contractions and vary sentence lengths, these detectors get confused.

My correlation analysis proved this split. Binoculars (a perplexity detector) correlated with Winston AI at r=0.88 (p=0.001) — a strong, statistically significant relationship. But Binoculars correlated with ZeroGPT at r=-0.27. That's a *negative* correlation. When Binoculars says "more AI," ZeroGPT actually says "less AI." They're measuring opposite things.

The overall ensemble score showed essentially zero correlation with QuillBot (r=0.22, p=0.538). No relationship at all.

[Figure 3: Correlation heatmap between local and external detectors. Winston correlates strongly with Binoculars (r=0.88). ZeroGPT and QuillBot show near-zero or negative correlations with all local detectors. See `graph5_correlation_heatmap.png`]

---

## What This Means for You

**If you're a student:** Your genuinely human-written work might get flagged by one detector and cleared by another. A 2025 CDT survey found that 43% of US teachers use AI detectors. If your professor runs your essay through GPTZero, you might face scrutiny that wouldn't exist if they'd used ZeroGPT. This isn't about whether you used AI — it's about which detector your institution chose.

**If you're a teacher or professor:** Relying on a single AI detector is risky. The disagreement between tools isn't minor — it's fundamental. Two detectors can give diametrically opposite verdicts on the same text. Using AI detection results as the sole basis for academic integrity decisions is, at best, unreliable.

**If you're a publisher or employer:** There is no gold standard for AI detection. Even the most confident detectors (GPTZero and Originality.ai at 100%) disagree completely with other tools. The technology is useful as one signal among many, but it's not a definitive answer.

---

## The Bigger Question

My research started as a technical challenge: can you build a pipeline that reduces AI detection scores? The answer is yes, with significant caveats. My pipeline reduced local detection scores by an average of 11.6 percentage points across 30 texts. Against classifier-based detectors like ZeroGPT, the optimized texts averaged just 20.8% AI — well into "human" territory. But against perplexity-based detectors like GPTZero, every single text still scored 100%.

But the more interesting finding isn't about evasion — it's about detection itself. If five detectors can't agree on whether a text is AI-generated, what does "AI-generated" even mean in a detection context?

Maybe the answer isn't building better detectors. Maybe it's rethinking our approach entirely. Watermarking (embedding invisible signals in AI output), provenance tracking (recording the origin of text), or simply teaching people to use AI as a tool rather than a shortcut — these approaches don't depend on the impossible task of reliably distinguishing human from machine writing.

The AI detection arms race is accelerating, but based on my findings, it's built on a shaky foundation. The same text can be "definitely human" or "definitely AI" depending on which tool you ask.

And that should concern everyone.

---

*This research involved 46 AI-generated texts, 30 optimization runs, 10 external benchmark tests, 8 analysis graphs, and a correlation study across 7 detectors. For full technical details including methodology, statistical analysis, and the complete pipeline architecture, see the accompanying research paper.*

*Onurcan Genc is a cybersecurity researcher, CVE contributor, and Computer Technology and Information Systems student at Bilkent University, Ankara, Turkey.*

**Tags:** AI Detection, GPTZero, Machine Learning, Academic Integrity, Natural Language Processing
