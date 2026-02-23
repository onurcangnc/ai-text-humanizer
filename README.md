# AI Text Humanizer

AI-generated text optimizer using a 3-detector perplexity ensemble (Binoculars + GPT-2 PPL + GLTR), Q-learning strategy selection, and Thompson sampling to iteratively reduce AI detection scores. Includes cross-detector consistency benchmark across 8 detectors.

> **Research paper:** [ArXiv paper](arxiv_paper.md) | **Blog post:** [Medium article](medium_article.md)

## Key Findings

- **Cross-detector inconsistency:** The same optimized text scored 0% AI on ZeroGPT and 100% AI on GPTZero simultaneously
- **Detectors oppose each other:** Binoculars vs ZeroGPT correlation is r=-0.65 (p=0.041) — statistically significant *negative*
- **Two detection paradigms:** Perplexity-based (GPTZero, Originality.ai, Binoculars) vs classifier-based (ZeroGPT, QuillBot) measure fundamentally different signals
- **Classifier vulnerability:** ZeroGPT scores dropped from 38.8% to 20.8% with stylistic modifications; GPTZero stayed at 100% throughout
- **Cross-cluster agreement as low as 10%** between perplexity and classifier detector groups

## Architecture

```
generate_training_data.py    — Training pipeline: generate AI text → optimize → verify
         ├── ml.py           — Synonym engine, sentence restructuring, Q-learning
         ├── detector.py     — 3-detector local ensemble (Binoculars, GPT-2 PPL, GLTR)
         └── self_learn.py   — v6.0 orchestrator: feedback loops, domain profiling

benchmark_analysis.py        — Benchmark analysis: 9 graphs + research report
```

## How It Works

1. **Generate** academic text via GPT-4 / Claude / DeepSeek (46 texts, 103 topics)
2. **Analyze** with 3-detector local perplexity ensemble (weighted score)
3. **Optimize** using 10 strategies across two branches:
   - **Perplexity injection:** rare synonyms, discourse markers, rhetorical questions, rhythm variation
   - **Classifier-targeted:** contraction injection, burstiness injection, writing imperfections
4. **Learn** from detector feedback via Thompson Sampling + Q-Learning
5. **Benchmark** against 5 commercial detectors (GPTZero, ZeroGPT, Originality.ai, Winston AI, QuillBot)

## Local Detector Ensemble

| Detector | Method | Weight | VRAM |
|----------|--------|--------|------|
| Binoculars [1] | Falcon-7B x2, cross-perplexity ratio, 4-bit NF4 | 0.35 | ~9.2 GB |
| GPT-2 Perplexity [13] | Document-level perplexity, sigmoid normalization | 0.35 | ~0.2 GB |
| GLTR [2] | Token rank distribution (top-10/top-50) | 0.30 | shared |

**Total VRAM:** ~10.0 GB (NVIDIA RTX 4090 Laptop 16GB)

## Benchmark Results

### V1: LLM Rewrite + Perplexity Injection

| Detector | Average Score | Type |
|----------|--------------|------|
| GPTZero | 100% | Perplexity-based |
| Originality.ai | 100% | Perplexity-based |
| Winston AI | 80.8% | Hybrid |
| ZeroGPT | 38.8% | Classifier-based |
| QuillBot | 26.7% | Classifier-based |

### V2: + Classifier-Targeted Optimization

| Phase | ZeroGPT | QuillBot | GPTZero | Originality |
|-------|---------|----------|---------|-------------|
| Original AI text | ~95% | ~95% | ~95% | ~95% |
| V1 (LLM rewrite + perplexity) | 38.8% | 26.7% | 100% | 100% |
| V2 (+ classifier optimization) | **20.8%** | **24.9%** | 100% | 100% |

### Correlation Analysis (V2)

| Local Detector | ZeroGPT (r/p) | Winston (r/p) | QuillBot (r/p) |
|---------------|---------------|---------------|----------------|
| Binoculars | **-0.65 / 0.041** | 0.24 / 0.507 | -0.40 / 0.254 |
| GPT-2 PPL | 0.34 / 0.344 | -0.02 / 0.964 | 0.08 / 0.829 |
| GLTR | -0.00 / 0.994 | -0.03 / 0.936 | -0.31 / 0.383 |
| Ensemble | -0.08 / 0.820 | 0.08 / 0.829 | -0.27 / 0.444 |

## Analysis Graphs

| Graph | Description |
|-------|-------------|
| `graph1_training_progress.png` | Training progress across 3 batches (30 texts) |
| `graph2_improvement_distribution.png` | Per-text improvement distribution |
| `graph3_strategy_effectiveness.png` | Strategy selection frequency & avg improvement |
| `graph4_source_model_comparison.png` | GPT-4 vs Claude vs DeepSeek performance |
| `graph5_correlation_heatmap.png` | Local vs external detector correlation (V2) |
| `graph6_scatter_correlations.png` | Scatter plots with regression lines (V2) |
| `graph7_external_detector_heatmap.png` | 10 texts x 5 detectors heatmap |
| `graph8_detector_agreement.png` | Pairwise detector agreement matrix |
| `graph9_v1_vs_v2_comparison.png` | V1 vs V2 ZeroGPT/QuillBot comparison |

## Optimized Benchmark Texts

11 optimized texts used for external detector evaluation are included in the repository:

- 3x Claude-generated (blockchain, epigenetic, refugee)
- 3x DeepSeek-generated (ocean acidification, silk road, soil microbiome)
- 5x GPT-4-generated (deep sea, longevity, philosophy, postcolonial, robotics)

## Setup

### Windows (with CUDA)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

### Mac / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

### Environment Variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required:
- `ANTHROPIC_API_KEY` — Claude API
- `OPENAI_API_KEY` — GPT-4 API
- `DEEPSEEK_API_KEY` — DeepSeek API

## Usage

### Training Pipeline
```bash
# Generate and optimize 20 texts
python generate_training_data.py --max-texts 20 --train

# Generate only (no training)
python generate_training_data.py --max-texts 10
```

### Benchmark Analysis
```bash
# Generate all 9 graphs + research report
python benchmark_analysis.py
```

## Project Structure

```
.
├── detector.py                  # 3-detector local ensemble (Binoculars, GPT-2, GLTR)
├── ml.py                        # Synonym engine + Q-learning
├── self_learn.py                # Training orchestrator v6.0
├── generate_training_data.py    # Training pipeline
├── benchmark_analysis.py        # Benchmark analysis (9 graphs + report)
├── medium_article.md            # Published Medium article
├── arxiv_paper.md               # ArXiv research paper
├── research_report.txt          # Generated benchmark report
├── optimized_*.txt              # 11 benchmark texts
├── graph*.png                   # 9 analysis graphs
├── requirements.txt
├── .env.example
├── training_data/               # Generated AI texts (gitignored)
└── learning_data/               # Q-tables, synonym success rates (gitignored)
```

## References

[1] Hans et al. (2024). Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text. ICML 2024. https://arxiv.org/abs/2401.12070

[2] Gehrmann et al. (2019). GLTR: Statistical Detection and Visualization of Generated Text. ACL 2019. https://arxiv.org/abs/1906.04043

[3] CDT (2025). Hand in Hand: Schools' Embrace of AI Connected to Increased Risks to Students. https://cdt.org/wp-content/uploads/2025/10/FINAL-CDT-2025-Hand-in-Hand-Polling-100225-accessible.pdf

[13] Radford et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI.

[14] Liang et al. (2023). GPT detectors are biased against non-native English writers. Patterns, 4(7). https://arxiv.org/abs/2304.02819

[18] Kirchenbauer et al. (2023). A Watermark for Large Language Models. ICML 2023. https://arxiv.org/abs/2301.10226
