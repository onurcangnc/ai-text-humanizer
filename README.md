# AI Text Humanizer

AI-generated text optimizer that uses BERT MLM synonym replacement, Q-learning strategy selection, and Thompson sampling feedback to iteratively reduce AI detection scores.

## Architecture

```
generate_training_data.py   — Training pipeline: generate AI text → optimize → verify
         ├── ml.py          — BERT MLM synonym engine, sentence restructuring, Q-learning
         ├── detector.py    — 4-detector ensemble (ZeroGPT, ContentDetector, MyDetector.ai, Winston AI)
         └── self_learn.py  — v6.0 orchestrator: feedback loops, domain profiling
```

## How It Works

1. **Generate** academic text via GPT-4 / Claude / DeepSeek
2. **Analyze** with 4 AI detectors (ensemble weighted score)
3. **Optimize** using 5 strategies: conservative, moderate, aggressive, restructure, paraphrase
4. **Learn** from detector feedback via Thompson sampling + Q-learning
5. **Repeat** until ensemble score drops below target (default: 30%)

## Setup

### Mac / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

### Windows (with CUDA)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"

# Optional: Binoculars AI detector (requires NVIDIA GPU)
pip install git+https://github.com/ahans30/Binoculars.git
```

### Environment Variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required keys:
- `ANTHROPIC_API_KEY` — Claude API (for text generation)
- `OPENAI_API_KEY` — GPT-4 API (for text generation)
- `DEEPSEEK_API_KEY` — DeepSeek API (for text generation)

Optional:
- `WINSTON_API_KEY` — Winston AI detector (adds 4th detector)
- `USE_PARROT=true` — Enable Parrot T5 paraphraser

## Usage

### Training Pipeline
```bash
# Generate and optimize 20 texts
python generate_training_data.py --max-texts 20 --train

# Generate only (no training)
python generate_training_data.py --max-texts 10
```

### Standalone Detection
```python
from detector import AIDetector

with AIDetector() as d:
    result = d.detect("Your text here...")
    print(f"AI probability: {result['ensemble']:.1%}")
    # Individual scores: result['zerogpt'], result['contentdetector'],
    #                    result['mydetector'], result['winston']
```

### Standalone Text Optimization
```python
from ml import TextQualityAnalyzer

analyzer = TextQualityAnalyzer(enable_learning=True)
variants = analyzer.generate_variants("Your AI-generated text here...")

for text, metrics, strategy, changes in variants:
    print(f"{strategy}: {len(changes)} changes")
```

## Detectors

| Detector | Method | Weight | Notes |
|----------|--------|--------|-------|
| ZeroGPT | Playwright | 0.30 | Free, no login |
| ContentDetector.ai | Playwright | 0.20 | Free, no login |
| MyDetector.ai | Playwright | 0.35 | Free, per-sentence scores |
| Winston AI | HTTP API | 0.15 | Requires API key |

## Optimization Strategies

| Strategy | Description |
|----------|-------------|
| Conservative (0.15) | Light synonym replacement, minimal changes |
| Moderate (0.30) | Balanced synonym + sentence splitting |
| Aggressive (0.50) | Heavy replacement + filler phrases |
| Restructure (0.35) | Sentence reordering + restructuring |
| Paraphrase (0.40) | Parrot T5 paraphrasing (if enabled) |

## Project Structure

```
.
├── detector.py              # AI detector ensemble
├── ml.py                    # BERT MLM engine + Q-learning
├── self_learn.py            # Training orchestrator v6.0
├── generate_training_data.py # Training pipeline
├── requirements.txt
├── .env.example
├── training_data/           # Generated AI texts (gitignored)
└── learning_data/           # Q-tables, synonym success rates (gitignored)
```
