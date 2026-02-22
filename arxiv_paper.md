# Cross-Detector Inconsistency in AI Text Detection: A Benchmark Study with Hybrid Humanization Techniques

**Onurcan Genc**
Department of Computer Technology and Information Systems, Bilkent University, Ankara, Turkey
onurcan.genc@bilkent.edu.tr

---

## Abstract

The proliferation of large language models (LLMs) has prompted the development of AI text detection tools, now widely deployed in educational and publishing contexts. However, the cross-detector consistency of these tools remains poorly studied. We present a systematic benchmark of five commercial AI detectors (GPTZero, ZeroGPT, Originality.ai, Winston AI, and QuillBot) and three open-source perplexity-based detectors (Binoculars, GPT-2 Perplexity, GLTR) evaluated on 46 AI-generated academic texts processed through a novel hybrid humanization pipeline. Our pipeline combines cross-model LLM rewriting, AI-telltale vocabulary replacement, perplexity injection, and classifier-targeted stylistic modification, achieving an average 11.6 percentage point reduction in local detection scores across 30 texts. External benchmarking reveals dramatic cross-detector inconsistency: the same optimized text scored 0% AI on ZeroGPT and 100% AI on GPTZero. Correlation analysis identifies two distinct detector clusters — perplexity-based (Binoculars, Winston; r=0.88, p<0.01) and classifier-based (ZeroGPT, QuillBot) — with cross-cluster agreement rates as low as 10%. Classifier-targeted optimization reduced ZeroGPT scores from 38.8% to 20.8% average while perplexity-based detectors remained unaffected. These findings demonstrate that AI text detection is fundamentally inconsistent across tools and that institutions relying on single detectors risk both false positives and false negatives at unacceptable rates.

**Keywords:** AI text detection, large language models, text humanization, cross-detector consistency, Binoculars, perplexity analysis, academic integrity

---

## 1. Introduction

The rapid advancement of large language models (LLMs) including GPT-4 (OpenAI, 2023), Claude (Anthropic, 2024), and DeepSeek (DeepSeek-AI, 2024) has made high-quality AI-generated text readily accessible. These models can produce fluent, coherent, and contextually appropriate text across virtually any domain, raising significant concerns in academic integrity, journalism, and professional communication.

In response, a growing ecosystem of AI text detection tools has emerged. A 2025 survey by the Center for Democracy and Technology found that 43% of US teachers report using AI detection tools in their classrooms. Commercial services such as GPTZero, ZeroGPT, Originality.ai, Winston AI, and QuillBot now serve millions of users. Despite this widespread adoption, the consistency and reliability of these tools — particularly when evaluated against each other on identical texts — has received insufficient systematic study.

This paper addresses three research questions:

**RQ1:** How consistent are commercial AI detectors when evaluating the same text? Specifically, what is the pairwise agreement rate across five major commercial detectors?

**RQ2:** Do open-source perplexity-based detectors correlate with commercial detection tools? If not, what explains the discrepancy?

**RQ3:** Can a hybrid humanization pipeline differentially affect detector types, and what does this reveal about underlying detection methodologies?

To answer these questions, we developed a hybrid humanization pipeline that combines cross-model LLM rewriting, vocabulary-level perturbation, and classifier-targeted stylistic modification. We generated 46 academic texts across 103 topics using three LLMs, optimized 30 of them through an iterative self-learning process, and benchmarked 10 optimized texts against eight detectors (three local, five commercial).

Our contributions are:

1. **A systematic cross-detector consistency study** revealing pairwise agreement rates ranging from 10% to 100% across eight AI detectors, with cross-paradigm agreement as low as 10%.

2. **Identification of two distinct detector clusters** — perplexity-based and classifier-based — with Pearson correlation r=0.88 within the perplexity cluster but r=-0.27 to r=0.44 between clusters.

3. **A novel hybrid humanization pipeline** combining cross-model LLM rewriting, 160+ vocabulary swaps, perplexity injection, and classifier-targeted features, with quality preservation through SBERT semantic guards.

4. **Empirical evidence of differential detector vulnerability:** classifier-based detectors (ZeroGPT: 38.8% to 20.8%) are substantially more affected by stylistic modification than perplexity-based detectors (GPTZero: constant 100%).

---

## 2. Related Work

### 2.1 AI Text Detection Methods

AI text detection approaches broadly fall into three categories: statistical methods, trained classifiers, and watermarking.

**Statistical and perplexity-based methods** analyze the probability distribution of tokens under a reference language model. Gehrmann et al. (2019) introduced GLTR (Giant Language Model Test Room), which visualizes the rank of each token in a language model's predicted distribution, observing that human text contains more low-rank (surprising) tokens than machine-generated text. Mitchell et al. (2023) proposed DetectGPT, which uses probability curvature — the observation that machine-generated text tends to occupy local maxima of a model's log-probability landscape. Hans et al. (2024) introduced Binoculars, a zero-shot detection method using two related language models (an "observer" and a "performer") to compute a cross-perplexity ratio, achieving state-of-the-art performance without requiring training data.

**Trained classifier methods** involve supervised learning on datasets of human and machine-generated text. GPTZero (Tian, 2023) uses perplexity and burstiness features in a classification framework. OpenAI released and subsequently discontinued their own classifier in 2023 due to low accuracy. Commercial tools including ZeroGPT, Originality.ai, Winston AI, and QuillBot operate as black boxes with undisclosed architectures, though their behavior patterns suggest varying combinations of statistical features and learned classifiers.

**Watermarking approaches** embed statistically detectable signals during text generation. Kirchenbauer et al. (2023) proposed partitioning the vocabulary into "green" and "red" lists at each generation step, biasing token selection toward green-list tokens. This approach offers provable detection guarantees but requires control over the generation process, making it inapplicable to existing deployed models.

### 2.2 Known Limitations of AI Detection

Several studies have identified fundamental limitations in AI text detection. Sadasivan et al. (2023) provided theoretical analysis showing that as language models improve, the overlap between human and machine text distributions increases, making reliable detection provably harder. They demonstrated that paraphrasing attacks can reduce detection accuracy to near-random levels.

Krishna et al. (2023) showed that even simple paraphrasing tools can defeat most AI detectors, though retrieval-based defenses offer some robustness. Liang et al. (2023) documented a concerning bias in AI detectors: non-native English speakers' writing was systematically flagged as AI-generated at higher rates than native speakers, raising fairness concerns for the millions of non-native English speakers in academic settings.

### 2.3 Research Gap

While individual detector evaluations exist, there is a notable absence of systematic cross-detector consistency studies using modern (2025-2026) commercial tools. Prior work has focused on detection accuracy against known AI-generated text, not on inter-detector agreement. Furthermore, the relationship between open-source perplexity-based detectors and commercial black-box tools has not been quantified through correlation analysis. This paper addresses both gaps.

---

## 3. Methodology

### 3.1 Text Generation

We generated 46 academic texts using three LLMs: GPT-4 (via OpenAI API, model `gpt-4o`), Claude (via Anthropic API, model `claude-sonnet-4-6`), and DeepSeek (via DeepSeek API, model `deepseek-chat`). Each model received a standardized prompt requesting a 400-600 word academic essay on a specified topic.

Topics were generated programmatically using the LLMs themselves in a rotating scheme, yielding 103 unique topics spanning marine biology, economics, philosophy, law, artificial intelligence, history, medicine, political science, and environmental science. This diversity ensures our findings are not domain-specific.

The 46 texts were distributed across three generation batches with model rotation: GPT-4 generated 21 texts (46%), Claude generated 12 texts (26%), and DeepSeek generated 13 texts (28%).

### 3.2 Humanization Pipeline

The humanization pipeline operates in three phases, applied iteratively with a self-learning optimization loop.

#### Phase 1: Cross-Model LLM Rewriting

Each text is rewritten by a different LLM than the one that generated it: GPT-4 text is rewritten by Claude, Claude text by DeepSeek, and DeepSeek text by GPT-4. This cross-model strategy forces a style transfer — the rewriting model's characteristic patterns overwrite the original model's patterns, producing text that doesn't cleanly match any single model's signature.

The rewrite prompt specifies detailed stylistic requirements targeting classifier-based detector weaknesses: natural contraction usage (minimum 5-7 per text), self-corrections and hedging phrases, rhetorical asides in parentheses, dramatic paragraph length variation, colloquial comparisons, lowercase conjunction sentence starters, dash-interrupted thoughts, mild qualifiers, and intentional minor writing imperfections. Temperature is set to 0.95 to maximize output diversity.

#### Phase 2: Statistical Perturbation

The rewritten text undergoes vocabulary-level and structural modifications:

**AI-telltale vocabulary replacement.** A curated dictionary of 130+ word-level and 30+ phrase-level substitutions replaces vocabulary patterns characteristic of AI output. Examples include "Furthermore" to "On top of that," "it is crucial to note" to "it matters a lot," and "comprehensive" to "thorough." Replacements are applied with case preservation and parenthetical content protection to prevent cross-iteration contamination.

**Rare synonym injection.** Using WordNet for synonym candidates and the `wordfreq` library for frequency scoring, we replace common words with rarer synonyms from the bottom 30-60% frequency percentile. A word-level SBERT semantic guard (cosine similarity threshold 0.70) ensures replacements preserve meaning. Spell-checking validates inflected forms.

**Perplexity injection.** Five methods increase token-level unpredictability: parenthetical asides with em-dashes, topic-matched rhetorical questions, discourse markers ("Granted," "Realistically,"), and sentence rhythm variation through short-sentence merging.

**Classifier-targeted features.** Three additional methods specifically target classifier-based detectors: contraction injection (26 formal-to-contraction patterns applied at 70% rate, skipping quoted and parenthetical content), burstiness injection (increasing sentence length standard deviation above 8 words through early splits and medium-sentence merges), and writing imperfection injection (hedging insertions, intensifier additions, informal summary prefixes).

#### Phase 3: Quality Preservation

Multiple filters ensure semantic fidelity:

- **SBERT sentence-level filter:** Each modified sentence is compared against the original using Sentence-BERT (all-MiniLM-L6-v2) cosine similarity. When sentence counts differ due to structural modifications, whole-text similarity is evaluated against a threshold of 0.70. When counts match, per-sentence threshold is 0.80 with reversion of failing sentences.
- **TF-IDF similarity guard:** Threshold 0.65 against original text.
- **Semantic similarity guard:** SBERT whole-text cosine threshold 0.70.
- **Domain term protection:** Technical vocabulary identified through domain detection is protected from modification.
- **Sentence-level check:** Injected fragments (rhetorical questions, very short splits) are filtered from quality evaluation. Up to 2 low-scoring sentences permitted when overall semantic similarity exceeds 0.85.

#### Self-Learning Optimization

The pipeline employs a two-level optimization strategy. Thompson Sampling with Beta priors selects among variant strategies based on accumulated success/failure statistics. Q-Learning with state discretization (domain, detection level, iteration) optimizes strategy ordering. An epsilon-greedy exploration policy (epsilon=0.05) balances exploitation of known effective strategies with exploration of potentially superior alternatives.

### 3.3 Detection Setup

#### Local Ensemble

Three open-source detectors run on a local NVIDIA RTX 4090 Laptop GPU (16GB VRAM):

**Binoculars** (Hans et al., 2024): Uses two Falcon-7B models (observer and performer) quantized to 4-bit NF4 precision. Computes cross-perplexity ratio with sigmoid normalization (threshold 0.8536). VRAM usage: approximately 9.2GB. Weight in ensemble: 0.35.

**GPT-2 Perplexity:** Computes document-level perplexity under GPT-2 (124M parameters, fp16). Texts with low perplexity (high predictability) score as more likely AI-generated. Sigmoid normalization with threshold 60.0 and steepness 0.08. VRAM usage: approximately 0.2GB. Weight in ensemble: 0.35.

**GLTR** (Gehrmann et al., 2019): Analyzes GPT-2 token rank distributions. Computes weighted score from top-10 and top-50 rank proportions (0.6 * top_k + 0.4 * top_50). VRAM usage: shared with GPT-2 Perplexity model. Weight in ensemble: 0.30.

Total VRAM usage: approximately 10.0GB including Sentence-BERT (0.08GB) and T5-base paraphraser (0.5GB, used in alternative pipeline branch).

#### Commercial Detectors

Five commercial detectors were evaluated through their web interfaces:

- **GPTZero** (gptzero.me): Reports per-document AI probability percentage.
- **ZeroGPT** (zerogpt.com): Reports percentage of AI-generated content.
- **Originality.ai** (originality.ai): Reports AI probability score.
- **Winston AI** (gowinston.ai): Reports AI detection confidence percentage.
- **QuillBot AI Detector** (quillbot.com): Reports percentage of AI-generated content.

All commercial tests were conducted manually through web UI submission in February 2026.

### 3.4 Evaluation Metrics

**Detection score:** AI detection confidence reported by each tool, normalized to 0-100% where 100% indicates full AI attribution.

**Cross-detector agreement:** For each detector pair, we compute binary agreement rate — the proportion of texts where both detectors agree on classification (both above 50% AI or both at or below 50% AI).

**Pearson correlation coefficient:** Computed between each pair of local and commercial detector scores across the 10 benchmark texts, with p-values for statistical significance testing. GPTZero and Originality.ai (constant 100%) are excluded from correlation analysis due to zero variance.

**Text quality metrics:** SBERT cosine similarity between original and optimized text, TF-IDF similarity, manual fact verification, contraction count, and sentence length standard deviation (burstiness).

---

## 4. Results

### 4.1 Training Performance

We optimized 30 texts across three training batches, each consisting of 10 texts processed through the full pipeline with 3 iterations of swap-based optimization following the initial LLM rewrite iteration.

[Table 1: Training results across three batches]

| Batch | Texts | Avg Initial | Avg Final | Avg Improvement | Best | Worst |
|-------|-------|-------------|-----------|-----------------|------|-------|
| 1 | 10 | 86.7% | 76.9% | -9.8 pp | -22.9 pp | 0.0 pp |
| 2 | 10 | 86.0% | 70.8% | -15.2 pp | -41.8 pp | -4.6 pp |
| 3 | 10 | 90.7% | 81.1% | -9.6 pp | -25.2 pp | -0.7 pp |
| **Overall** | **30** | **87.8%** | **76.3%** | **-11.6 pp** | **-41.8 pp** | **0.0 pp** |

The average improvement of 11.6 percentage points represents meaningful detection score reduction, though no text reached the target threshold of 30%. The best single improvement was 41.8 percentage points (Weimar Republic hyperinflation text: 80.4% to 38.5%). Three texts showed less than 1 percentage point improvement, indicating that the pipeline's effectiveness varies substantially with input characteristics.

[Figure 1: Training progress across three batches showing initial and final detection scores for each text. See `graph1_training_progress.png`]

#### Strategy Analysis

The self-learning system selected from 10 variant strategies across the T5 paraphrasing branch (light, balanced, structural, human-noise, kitchen-sink) and the perplexity injection branch (perplexity/classifier-light, moderate, structural, heavy, kitchen). Strategy selection frequencies and average per-selection improvements are shown in Figure 3.

Perplexity-light was selected most frequently (17 times) but achieved only 1.2 pp average improvement per selection. Kitchen-sink, selected only 4 times, achieved the highest average improvement at 6.8 pp. This suggests that conservative strategies are preferred by the self-learning system due to their reliability (consistent small improvements), while aggressive strategies offer higher potential gains but also higher rejection rates from quality filters.

[Figure 2: Strategy selection frequency and average improvement. See `graph3_strategy_effectiveness.png`]

#### Source Model Comparison

We analyzed optimization outcomes by the source AI model that generated the original text.

[Table 2: Optimization results by source model]

| Source Model | n | Avg Initial | Avg Final | Avg Improvement |
|-------------|---|-------------|-----------|-----------------|
| GPT-4 | 15 | 92.6% | 83.0% | -9.6 pp |
| Claude | 7 | 82.6% | 71.8% | -10.7 pp |
| DeepSeek | 8 | 83.4% | 67.5% | -15.9 pp |

DeepSeek-generated texts showed the largest average improvement (15.9 pp), suggesting they contain more exploitable AI patterns. GPT-4 texts were most resistant to optimization (9.6 pp), consistent with GPT-4's reputation for more diverse and human-like output. Claude texts showed intermediate behavior.

[Figure 3: Source model comparison showing initial and final detection scores. See `graph4_source_model_comparison.png`]

### 4.2 External Detector Benchmark

Ten optimized texts were submitted to five commercial AI detectors. Results are shown in Table 3.

[Table 3: External detector scores for 10 optimized texts (V1, pre-classifier optimization)]

| Text | Local Ens. | GPTZero | ZeroGPT | Originality | Winston | QuillBot |
|------|-----------|---------|---------|-------------|---------|----------|
| Blockchain supply chain | 91.3% | 100% | 36.8% | 100% | 99% | 33% |
| Epigenetic inheritance | 90.8% | 100% | 16.3% | 100% | 100% | 7% |
| Refugee quota | 79.0% | 100% | 100% | 100% | 100% | 74% |
| Ocean acidification | 75.2% | 100% | 63.3% | 100% | 13% | 20% |
| Silk Road currency | 54.7% | 100% | 0% | 100% | 16% | 0% |
| Soil microbiome | 79.0% | 100% | 66.4% | 100% | 100% | 22% |
| Longevity pension | 78.2% | 100% | 12.6% | 100% | 90% | 31% |
| Philosophy consciousness | 88.0% | 100% | 22.3% | 100% | 90% | 7% |
| Postcolonial literature | 90.6% | 100% | 39.1% | 100% | 100% | 32% |
| Robotics disaster | 83.8% | 100% | 32.2% | 100% | 100% | 41% |
| **Average** | **81.1%** | **100%** | **38.8%** | **100%** | **80.8%** | **26.7%** |

GPTZero and Originality.ai exhibited zero variance, classifying every text as 100% AI-generated regardless of the humanization applied. ZeroGPT showed the widest range (0% to 100%), with an average of 38.8%. QuillBot averaged 26.7%, classifying most texts as predominantly human. Winston AI averaged 80.8% but displayed extreme variance (13% to 100%).

The most striking finding is the Silk Road currency exchange text, which simultaneously scored 0% on ZeroGPT, 0% on QuillBot, 16% on Winston, 100% on GPTZero, and 100% on Originality.ai. The same text is classified as definitively human and definitively AI depending on which tool is used.

[Figure 4: External detector heatmap showing 10 texts across 5 detectors. See `graph7_external_detector_heatmap.png`]

### 4.3 Classifier Optimization Impact

Following the initial benchmark, we implemented classifier-targeted optimization (contraction injection, burstiness injection, writing imperfection injection) and re-optimized all 10 benchmark texts. Re-evaluation against ZeroGPT and QuillBot yielded the results in Table 4.

[Table 4: Before and after classifier-targeted optimization (ZeroGPT and QuillBot only)]

| Text | ZeroGPT V1 | ZeroGPT V2 | QuillBot V1 | QuillBot V2 |
|------|-----------|-----------|------------|------------|
| Blockchain supply chain | 36.8% | 36.5% | 33% | 54% |
| Epigenetic inheritance | 16.3% | 5.0% | 7% | 5% |
| Refugee quota | 100% | 65.3% | 74% | 69% |
| Ocean acidification | 63.3% | 28.9% | 20% | 5% |
| Silk Road currency | 0% | 4.8% | 0% | 62% |
| Soil microbiome | 66.4% | 6.5% | 22% | 0% |
| Longevity pension | 12.6% | 0% | 31% | 4% |
| Philosophy consciousness | 22.3% | 6.2% | 7% | 5% |
| Postcolonial literature | 39.1% | 16.8% | 32% | 37% |
| Robotics disaster | 32.2% | 37.9% | 41% | 8% |
| **Average** | **38.8%** | **20.8%** | **26.7%** | **24.9%** |

ZeroGPT scores decreased substantially from 38.8% to 20.8% average (-18.0 pp), confirming that classifier-based detectors respond to stylistic features. The refugee quota text — previously the worst performer at 100% — dropped to 65.3%. QuillBot showed marginal overall improvement (26.7% to 24.9%), though individual texts varied dramatically: ocean acidification dropped from 20% to 5%, while Silk Road increased from 0% to 62%.

The high variance between V1 and V2 QuillBot scores, particularly cases where scores increased (Silk Road: 0% to 62%, blockchain: 33% to 54%), highlights a critical methodological concern: the non-deterministic nature of both the humanization pipeline (different LLM rewrites per run) and potentially the detectors themselves produces substantial measurement noise. This limits the reproducibility of per-text comparisons while aggregate trends remain informative.

### 4.4 Correlation Analysis

We computed Pearson correlation coefficients between each local detector and each commercial detector with sufficient variance (excluding GPTZero and Originality.ai, which were constant at 100%).

[Table 5: Pearson correlation coefficients between local and external detectors]

| Local Detector | ZeroGPT (r/p) | Winston (r/p) | QuillBot (r/p) |
|---------------|---------------|---------------|----------------|
| Binoculars | -0.27 / 0.456 | **0.88 / 0.001** | 0.13 / 0.728 |
| GPT-2 PPL | 0.44 / 0.200 | 0.55 / 0.101 | 0.33 / 0.354 |
| GLTR | 0.17 / 0.635 | 0.43 / 0.218 | 0.06 / 0.879 |
| Ensemble | 0.13 / 0.710 | **0.77 / 0.010** | 0.22 / 0.538 |

Bold values indicate statistical significance at p<0.05.

The only statistically significant correlations are between our local detectors and Winston AI: Binoculars-Winston (r=0.88, p=0.001) and Ensemble-Winston (r=0.77, p=0.010). This suggests Winston AI employs a perplexity-based detection methodology similar to our open-source ensemble.

Critically, ZeroGPT shows near-zero or negative correlations with all local detectors. The Binoculars-ZeroGPT correlation is -0.27, meaning they respond in opposite directions to the same text modifications. This confirms that ZeroGPT measures fundamentally different textual properties than perplexity-based approaches.

QuillBot shows no statistically significant correlation with any local detector (all p>0.35), indicating it uses yet another distinct detection methodology.

[Figure 5: Correlation heatmap between local and external detectors. See `graph5_correlation_heatmap.png`]

[Figure 6: Scatter plots of local ensemble score vs. external detector scores with regression lines. See `graph6_scatter_correlations.png`]

### 4.5 Detector Agreement Analysis

We computed pairwise binary agreement rates across all seven detectors with variance (excluding GPTZero and Originality.ai).

[Figure 7: Pairwise detector agreement matrix. See `graph8_detector_agreement.png`]

Local detectors (Binoculars, GPT-2 PPL, GLTR, Ensemble) show internal agreement rates of 90-100%, forming a tightly coherent cluster. Winston AI agrees with this cluster at 80-90%, consistent with the strong Pearson correlations observed.

ZeroGPT agrees with local detectors at only 30-40% and with Winston AI at 30%. QuillBot shows the lowest cross-cluster agreement, agreeing with local detectors at only 10-20% and with the ensemble at just 10%.

ZeroGPT and QuillBot agree with each other at 80%, forming a second cluster. This two-cluster structure — perplexity-based detectors (Binoculars, GPT-2, GLTR, Winston) versus classifier-based detectors (ZeroGPT, QuillBot) — is the central structural finding of this study.

---

## 5. Discussion

### 5.1 The Reliability Problem

Our findings reveal a fundamental reliability problem in AI text detection: no consistent ground truth exists across detection tools. The same text can be classified as 0% AI by one tool and 100% AI by another, with the verdict determined not by the text's properties but by the detector's methodology.

This has immediate practical implications. In educational settings, a student's essay may be flagged or cleared depending on which tool their institution licenses. Given the CDT (2025) finding that 43% of US teachers use AI detectors, thousands of academic integrity decisions are potentially being made using tools that disagree with each other more than they agree.

The problem is not merely that detectors have different sensitivity thresholds — it is that they measure different properties of text. Adjusting thresholds cannot reconcile detectors that respond to opposite signals.

### 5.2 Two Detection Paradigms

Our correlation and agreement analyses identify two distinct detection paradigms:

**Perplexity-based detection** (Binoculars, GPT-2 PPL, GLTR, Winston AI) measures the statistical predictability of token sequences under a reference language model. This approach has strong theoretical foundations — machine-generated text tends to occupy high-probability regions of the model's output space — and is robust against superficial textual modifications. Our optimization pipeline reduced ensemble scores by only 11.6 pp on average despite extensive vocabulary and structural changes, and GPTZero/Originality.ai remained at 100% throughout.

**Classifier-based detection** (ZeroGPT, QuillBot) appears to rely on learned stylistic features: contraction frequency, sentence length variance (burstiness), informal marker presence, and structural regularity. These features correlate with human vs. AI writing in training data but are easily manipulated. Our classifier-targeted optimization — adding contractions, increasing burstiness, and injecting writing imperfections — reduced ZeroGPT scores from 38.8% to 20.8% with minimal impact on text quality.

Neither paradigm alone provides reliable detection. Perplexity-based methods are robust but may produce false positives on formulaic human writing. Classifier-based methods capture genuine stylistic differences but are trivially bypassed by prompt engineering.

### 5.3 The Arms Race and Sustainable Alternatives

Our results demonstrate a clear detection-evasion arms race. Each optimization technique we developed exploits a specific detector weakness: cross-model rewriting disrupts per-model signatures, vocabulary replacement targets telltale word patterns, and classifier-targeted features exploit stylistic heuristics.

This arms race is unlikely to converge. As Sadasivan et al. (2023) showed theoretically, the fundamental overlap between human and machine text distributions grows as models improve. Sustainable alternatives to detection may include:

- **Watermarking** (Kirchenbauer et al., 2023): Embedding statistical signals during generation provides provable detection guarantees but requires model-provider cooperation.
- **Provenance tracking:** Cryptographic attestation of text origin (e.g., C2PA content credentials) sidesteps the detection problem entirely.
- **Pedagogical approaches:** Teaching responsible AI use rather than attempting to detect and punish AI usage.

### 5.4 Ethical Considerations

This work presents a humanization pipeline that could be misused to circumvent AI detection for academic dishonesty. We frame this research explicitly as adversarial evaluation — a standard practice in security research where exposing vulnerabilities improves defensive capabilities. Our findings serve the detection research community by identifying specific weaknesses in current tools and quantifying cross-detector inconsistency.

Moreover, our results argue against over-reliance on AI detection rather than for evasion. The primary implication is that institutions should not treat AI detector outputs as authoritative, regardless of whether humanization techniques exist. The fundamental inconsistency across detectors is a property of the tools themselves, not of our pipeline.

### 5.5 Limitations

Several limitations constrain the generalizability of our findings:

**Sample size.** Our external benchmark involves only 10 texts, limited by the manual effort required for web UI testing. While aggregate trends are informative, per-text and per-detector correlation estimates have wide confidence intervals. A larger benchmark with API access would strengthen statistical conclusions.

**Web UI testing.** Commercial detectors were tested through web interfaces, which may differ from API behavior. Score presentation, preprocessing, or model versions may vary between interface modalities.

**Language scope.** All texts are in English. AI detection tools may behave differently on other languages, and the humanization techniques (contractions, discourse markers) are English-specific.

**Temporal snapshot.** Commercial detectors update their models regularly. Results from February 2026 may not reflect future detector capabilities.

**Non-deterministic pipeline.** The LLM rewrite phase produces different outputs per execution. While this introduces measurement noise, it also mirrors real-world conditions where users would obtain different results on repeated attempts.

**English writing conventions.** Our classifier-targeted optimization (contractions, informal markers) assumes English-language academic conventions. These techniques may not transfer to other linguistic or cultural contexts.

---

## 6. Conclusion

We present the first systematic cross-detector consistency study evaluating eight AI text detectors — three open-source perplexity-based tools and five commercial services — on identical texts processed through a hybrid humanization pipeline. Our findings reveal three key insights:

First, **AI text detection is fundamentally inconsistent across tools.** The same optimized text can simultaneously score 0% AI on one detector and 100% on another. Pairwise agreement rates between detector clusters drop as low as 10%, and the overall cross-paradigm agreement rate is 10-30%.

Second, **two distinct detection paradigms exist with near-zero cross-correlation.** Perplexity-based detectors (Binoculars, GPT-2, GLTR, Winston AI) form one coherent cluster (internal agreement 80-100%, Pearson r up to 0.88), while classifier-based detectors (ZeroGPT, QuillBot) form another (internal agreement 80%). These clusters measure fundamentally different textual properties and respond in opposite directions to the same modifications.

Third, **classifier-based detectors are substantially more vulnerable to targeted optimization** than perplexity-based detectors. Stylistic modifications (contractions, burstiness, imperfections) reduced ZeroGPT scores from 38.8% to 20.8% average while GPTZero and Originality.ai remained at 100%.

These findings have direct implications for policy: institutions should not rely on a single AI detection tool for consequential decisions. The disagreement between detectors is not a matter of sensitivity calibration — it reflects fundamentally different measurement approaches. Until these tools achieve cross-detector consistency, their use in academic integrity enforcement carries an unacceptable risk of both false positives (flagging human text) and false negatives (clearing AI text), depending on which tool is chosen.

Future work should expand the benchmark to include larger text samples, API-level testing, temporal analysis across detector updates, multilingual evaluation, and systematic comparison of detection consistency across text domains and difficulty levels.

---

## References

Center for Democracy and Technology. (2025). *AI in Education: Teacher Perspectives on AI Detection Tools.* CDT Research Report.

Gehrmann, S., Strobelt, H., & Rush, A. M. (2019). GLTR: Statistical Detection and Visualization of Generated Text. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations*, pp. 111-116.

Hans, A., Schwarzschild, A., Cheber, V., Lakkaraju, H., & Barak, B. (2024). Spotting LLMs with Binoculars: Zero-Shot Detection of Machine-Generated Text. In *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*.

Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., & Goldstein, T. (2023). A Watermark for Large Language Models. In *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*.

Krishna, K., Song, Y., Karpinska, M., Wieting, J., & Iyyer, M. (2023). Paraphrasing Evades Detectors of AI-Generated Text, but Retrieval is an Effective Defense. In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*.

Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., & Zou, J. (2023). GPT detectors are biased against non-native English writers. *Patterns*, 4(7), 100779.

Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., & Finn, C. (2023). DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature. In *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*.

OpenAI. (2023). GPT-4 Technical Report. *arXiv preprint arXiv:2303.08774*.

Sadasivan, V. S., Kumar, A., Balasubramanian, S., Wang, W., & Feizi, S. (2023). Can AI-Generated Text be Reliably Detected? *arXiv preprint arXiv:2303.11156*.

Tian, E. (2023). GPTZero: Towards Detection of AI-Generated Text. *Undergraduate thesis, Princeton University*.

---

**Appendix A: Humanization Pipeline Parameters**

| Parameter | Value |
|-----------|-------|
| LLM rewrite temperature | 0.95 |
| LLM rewrite max_tokens | 1500 |
| SBERT model | all-MiniLM-L6-v2 |
| SBERT word-level threshold | 0.70 |
| BERT sentence-level threshold (same count) | 0.80 |
| BERT sentence-level threshold (different count) | 0.70 |
| TF-IDF similarity threshold | 0.65 |
| Semantic similarity threshold | 0.70 |
| Sentence-level semantic floor | 0.25 |
| Max bad sentences (sem >= 0.85) | 2 |
| Contraction application rate | 70% |
| Burstiness stdev target | > 8 words |
| Rare synonym frequency floor | 1e-8 |
| AI-telltale word swaps | 130+ |
| AI-telltale phrase swaps | 30+ |
| Binoculars ensemble weight | 0.35 |
| GPT-2 PPL ensemble weight | 0.35 |
| GLTR ensemble weight | 0.30 |
| Binoculars quantization | NF4 (4-bit) |
| Total VRAM usage | ~10.0 GB |
| Hardware | NVIDIA RTX 4090 Laptop (16GB) |

**Appendix B: Classifier-Targeted Variant Configurations**

| Config | Contractions | Burstiness | Rare Synonyms | Parentheticals | Rhetorical | Rhythm | Discourse | Imperfections |
|--------|-------------|-----------|---------------|----------------|-----------|--------|-----------|--------------|
| classifier-light | Yes | Yes | 0.10 | 0 | 0 | No | 2 | 0 |
| classifier-moderate | Yes | Yes | 0.10 | 1 | 0 | No | 0 | 0 |
| classifier-structural | Yes | Yes | 0 | 0 | 1 | Yes | 0 | 0 |
| classifier-heavy | Yes | Yes | 0.15 | 2 | 0 | No | 0 | 2 |
| classifier-kitchen | Yes | Yes | 0.10 | 1 | 1 | Yes | 1 | 1 |
