"""
TEXT QUALITY ANALYZER v53.0 - SELF-LEARNING EDITION
From "Evasion" to "Analysis": Real Metrics, Real Quality, Real Transparency
CHAOTIC INVERTED: The tool now serves truth, not deception
NOW WITH: Dynamic Self-Learning, Domain Adaptation, and Intelligent Reinforcement

# Normal mod
python analyzer.py document.txt

# Öğrenme modu ile (geri bildirim istenir)
python analyzer.py document.txt --feedback

# Öğrenmeyi devre dışı bırak
python analyzer.py document.txt --no-learn
"""

import random
import re
import os
import sys
import json
import hashlib
from typing import List, Dict, Any, Tuple, Set
from datetime import datetime
from collections import Counter, defaultdict
import math
import pickle
from pathlib import Path

# WordNet + LemmInflect for massive synonym coverage + conjugation
import nltk
from lemminflect import getLemma, getInflection
from spellchecker import SpellChecker

# Shared spell checker instance (English)
_spell = SpellChecker()

# Ensure NLTK data is available
for _pkg in ('averaged_perceptron_tagger_eng', 'punkt_tab', 'wordnet'):
    try:
        if _pkg == 'punkt_tab':
            nltk.data.find(f'tokenizers/{_pkg}')
        elif _pkg == 'wordnet':
            nltk.data.find(f'corpora/{_pkg}')
        else:
            nltk.data.find(f'taggers/{_pkg}')
    except LookupError:
        nltk.download(_pkg, quiet=True)

# ── POS tag helpers ─────────────────────────────────────────────────────────
# Stopwords to never replace (determiners, prepositions, conjunctions, etc.)
_STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'must', 'ought',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'ours', 'theirs',
    'this', 'that', 'these', 'those', 'who', 'whom', 'which', 'what', 'whose',
    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    'not', 'no', 'if', 'then', 'than', 'when', 'while', 'as', 'because', 'since',
    'also', 'just', 'only', 'very', 'too', 'quite', 'rather', 'almost',
    'there', 'here', 'where', 'how', 'why', 'all', 'each', 'every', 'some',
    'any', 'few', 'more', 'most', 'other', 'such', 'own', 'same',
    'one', 'two', 'three', 'first', 'second', 'third', 'last', 'next',
})

# Words too polysemous (many meanings) for safe automatic substitution
_AMBIGUOUS_WORDS = frozenset({
    'time', 'way', 'set', 'run', 'mark', 'marks', 'plant', 'right', 'left',
    'point', 'case', 'part', 'place', 'state', 'head', 'line', 'hand',
    'long', 'high', 'well', 'back', 'still', 'even', 'old', 'new',
    'like', 'take', 'give', 'turn', 'keep', 'start', 'end', 'side',
    'kind', 'mean', 'form', 'hold', 'close', 'power', 'light', 'play',
    'class', 'field', 'force', 'order', 'sense', 'level', 'figure',
    'century', 'decade', 'decades', 'age', 'period', 'energy', 'term',
    'address', 'issue', 'spring', 'fall', 'leaves', 'base', 'post',
    'current', 'present', 'bear', 'sound', 'fair', 'match', 'ring',
    'area', 'often', 'describe', 'project', 'video', 'discourse',
    'role', 'goal', 'goals', 'history', 'story',
    'legacy', 'testament', 'event', 'events', 'case', 'cases',
    'explosion', 'transformation', 'culture', 'cultural',
    'become', 'create', 'creating', 'dimension', 'dimensions',
    'offer', 'used', 'style',
    # Noun/verb crossover — replacing these in compound phrases corrupts meaning
    'change', 'changes', 'work', 'works', 'lead', 'leads',
    'report', 'reports', 'study', 'studies', 'value', 'values',
    'process', 'processes', 'rate', 'rates', 'range', 'ranges',
    'approach', 'model', 'models', 'system', 'systems', 'growth',
    'research', 'impact', 'impacts', 'effect', 'effects',
    'species', 'threat', 'threats', 'damage', 'conditions',
    'balance', 'structure', 'function', 'functions', 'focus',
})

def _penn_to_upos(tag: str):
    """Map Penn Treebank POS tag to Universal POS for LemmInflect."""
    if tag.startswith('J'): return 'ADJ'
    if tag.startswith('V'): return 'VERB'
    if tag.startswith('N'): return 'NOUN'
    if tag.startswith('R'): return 'ADV'
    return None

def _pos_compatible(target_penn_tag: str, candidate_word: str) -> bool:
    """Check broad POS match (N/V/J/R) between target and candidate."""
    if not target_penn_tag:
        return True
    candidate_tags = nltk.pos_tag([candidate_word])
    if not candidate_tags:
        return True
    return target_penn_tag[0] == candidate_tags[0][1][0]

def _fix_articles(text: str) -> str:
    """Fix a/an mismatches after word substitution (e.g., 'a admonition' → 'an admonition')."""
    # "a" before vowel sound → "an"
    text = re.sub(r'\ba\s+([aeiouAEIOU]\w)', r'an \1', text)
    # "an" before consonant sound → "a"  (but keep "an hour", "an honest" etc.)
    _silent_h = {'hour', 'honest', 'honor', 'honour', 'heir', 'herb'}
    def _fix_an(m):
        word = m.group(1)
        if word.lower() in _silent_h:
            return m.group(0)  # keep "an"
        return f'a {word}'
    text = re.sub(r'\ban\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\w+)', _fix_an, text)
    return text


def _reconstruct_text(original: str, orig_tokens: List[str], new_tokens: List[str]) -> str:
    """Reconstruct text by replacing tokens in the original string, preserving whitespace."""
    result = original
    # Walk through tokens and replace changed ones in-order
    offset = 0
    for orig_tok, new_tok in zip(orig_tokens, new_tokens):
        if orig_tok == new_tok:
            # Find and skip past this token
            idx = result.find(orig_tok, offset)
            if idx >= 0:
                offset = idx + len(orig_tok)
            continue
        # Find the original token and replace it
        idx = result.find(orig_tok, offset)
        if idx >= 0:
            result = result[:idx] + new_tok + result[idx + len(orig_tok):]
            offset = idx + len(new_tok)
    return result


class LLMRewriter:
    """Cross-model LLM rewriting for high-quality humanization.
    GPT-4 text → Claude rewrites, Claude → DeepSeek, DeepSeek → GPT-4.
    Falls back to any available model if preferred one unavailable."""

    _CROSS_MODEL = {
        'gpt4': 'anthropic',
        'claude': 'deepseek',
        'deepseek': 'openai',
        'gemini': 'openai',
        'unknown': None,  # pick first available
    }

    _REWRITE_PROMPT = """Rewrite this academic text so it sounds like a real university student wrote it — not a polished AI. Follow these rules strictly:

STRUCTURE & RHYTHM:
- Dramatically vary sentence length. Mix short punchy sentences (5-8 words) with longer explanatory ones (25-30 words). Not every sentence should be medium-length.
- Some paragraphs should be shorter than others. Don't make them all the same length.
- Break one or two long sentences into fragments for emphasis. Like this.

TONE & WORD CHOICE:
- Use contractions naturally throughout: "don't", "isn't", "it's", "they're", "wouldn't", "can't"
- Start some sentences with "And", "But", "So", "Also" — real students do this
- Use first person sparingly (1-2 times total): "I think", "in my view", "as I see it"
- Include hedging: "probably", "it seems like", "arguably", "to some extent", "more or less"
- Use slightly informal phrasing occasionally: "a big deal", "kind of", "a lot of", "pretty much"
- NEVER use formal transitions: no "Furthermore", "Moreover", "Consequently", "Nevertheless", "In conclusion", "Additionally". Use "Also", "Plus", "But", "So", "Still", "That said", "On top of that" instead.

CLASSIFIER-BYPASS SIGNALS (these fool AI-detection classifiers):
- Use at least 5-7 contractions throughout (don't, isn't, it's, they're, won't, can't, wouldn't, hasn't, there's)
- Include 2-3 self-corrections or hedging: "well, actually", "or rather", "I mean", "to put it differently"
- Add 1-2 rhetorical asides in parentheses: (which is pretty fascinating when you think about it), (not that this is surprising), (at least from what we know so far)
- Vary paragraph length dramatically: one paragraph should be 2-3 sentences, another should be 5-6 sentences
- Include exactly ONE slightly colloquial comparison or analogy that a student might use
- Start at least 2 sentences with lowercase conjunctions: "and", "but", "so", "or"
- Use at least 1 dash-interrupted thought: "The main concern — and this is where it gets interesting — is that..."
- End one sentence with a mild qualifier: "...at least for now.", "...or so it seems.", "...though that's debatable."
- Do NOT make every sentence perfect — one sentence can be slightly wordy or have a minor structural quirk (like starting with "There is" or using passive voice unnecessarily) — this mimics real student writing imperfections

ABSOLUTE CONSTRAINTS:
- ALL facts, numbers, technical terms, and proper nouns must remain EXACTLY the same — do not alter any factual content
- Do NOT add or remove information
- The overall quality should still be good academic writing — just human, not AI
- Output ONLY the rewritten text, nothing else

Text to rewrite:
"""

    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        self._keys = {
            'openai': os.getenv('OPENAI_API_KEY', ''),
            'anthropic': os.getenv('ANTHROPIC_API_KEY', ''),
            'deepseek': os.getenv('DEEPSEEK_API_KEY', ''),
        }
        self.available = [k for k, v in self._keys.items() if v]

    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self._keys['openai'])
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.95,
        )
        return resp.choices[0].message.content.strip()

    def _call_anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self._keys['anthropic'])
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            temperature=0.95,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    def _call_deepseek(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self._keys['deepseek'], base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.95,
        )
        return resp.choices[0].message.content.strip()

    def _pick_provider(self, source_model: str) -> str:
        """Pick rewrite provider via cross-model mapping. Falls back to any available."""
        preferred = self._CROSS_MODEL.get(source_model)
        if preferred and preferred in self.available:
            return preferred
        # Fallback: pick any available provider that isn't the source
        source_providers = {'gpt4': 'openai', 'claude': 'anthropic', 'deepseek': 'deepseek'}
        source_prov = source_providers.get(source_model, '')
        for p in self.available:
            if p != source_prov:
                return p
        # Last resort: use any available
        return self.available[0] if self.available else None

    def rewrite(self, text: str, source_model: str = 'unknown') -> str:
        """Single-pass LLM rewrite. Returns rewritten text or None on failure."""
        provider = self._pick_provider(source_model)
        if not provider:
            return None

        prompt = self._REWRITE_PROMPT + text
        callers = {
            'openai': self._call_openai,
            'anthropic': self._call_anthropic,
            'deepseek': self._call_deepseek,
        }

        try:
            print(f"   🤖 LLM rewrite via {provider}...")
            result = callers[provider](prompt)
            if result and len(result) > 50:
                print(f"   ✅ LLM rewrite done ({len(result)} chars)")
                return result
            return None
        except Exception as e:
            print(f"   ❌ LLM rewrite failed: {str(e)[:120]}")
            return None


class T5SentenceParaphraser:
    """Sentence-level paraphrasing using humarin/chatgpt_paraphraser_on_T5_base.
    Lazy-loads ~500MB model (fp16) on first use. Uses Sentence-BERT for quality filtering."""

    STRATEGY_PARAMS = {
        # (sentence_ratio, num_beams, temperature)
        "light":        (0.20, 5, 0.7),
        "balanced":     (0.30, 5, 0.7),
        "structural":   (0.25, 5, 0.7),
        "human-noise":  (0.20, 5, 0.7),
        "kitchen-sink": (0.40, 5, 0.8),
    }

    def __init__(self, model_name: str = "humarin/chatgpt_paraphraser_on_T5_base"):
        self._model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = None
        self._sbert = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self._device == "cuda" else torch.float32

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self._model_name, torch_dtype=dtype
            ).to(self._device)
            self._model.eval()
            print(f"   T5 paraphraser loaded ({self._device})")
        return self._model, self._tokenizer

    def _load_sbert(self):
        """Lazy-load Sentence-BERT for quality filtering."""
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer("all-MiniLM-L6-v2")
        return self._sbert

    def _sbert_filter(self, original: str, candidates: List[str],
                      threshold: float = 0.75) -> List[str]:
        """Filter candidates by Sentence-BERT cosine similarity."""
        if not candidates:
            return []
        sbert = self._load_sbert()
        texts = [original] + candidates
        embeddings = sbert.encode(texts)
        from numpy import dot
        from numpy.linalg import norm
        orig_emb = embeddings[0]
        filtered = []
        for i, c in enumerate(candidates):
            cand_emb = embeddings[i + 1]
            sim = float(dot(orig_emb, cand_emb) / (norm(orig_emb) * norm(cand_emb)))
            if sim >= threshold:
                filtered.append(c)
        return filtered

    def paraphrase_sentence(self, sentence: str, num_beams: int = 5,
                            temperature: float = 0.7,
                            num_return: int = 5) -> List[str]:
        """Generate paraphrase candidates for a single sentence."""
        model, tokenizer = self._load()
        import torch

        input_text = f"paraphrase: {sentence}"
        encoding = tokenizer(
            input_text, max_length=256, padding="max_length",
            truncation=True, return_tensors="pt"
        ).to(self._device)

        with torch.inference_mode():
            outputs = model.generate(
                **encoding,
                max_length=256,
                num_beams=num_beams,
                num_return_sequences=min(num_return, num_beams),
                temperature=temperature,
                repetition_penalty=1.5,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        candidates = []
        for output in outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=True).strip()
            if decoded and decoded.lower() != sentence.lower():
                candidates.append(decoded)
        return candidates

    def paraphrase(self, text: str, strategy: str = "light",
                   learning_engine=None) -> Tuple[str, List[Tuple[str, str]]]:
        """Paraphrase text at sentence level per strategy params.
        Returns (paraphrased_text, [(orig_sent, new_sent), ...])."""
        params = self.STRATEGY_PARAMS.get(strategy, self.STRATEGY_PARAMS["balanced"])
        ratio, num_beams, temperature = params

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        result_sents = []
        changes = []

        for sent in sentences:
            words = sent.split()
            if len(words) < 5 or random.random() > ratio:
                result_sents.append(sent)
                continue

            candidates = self.paraphrase_sentence(
                sent, num_beams=num_beams,
                temperature=temperature, num_return=5
            )

            # Sentence-BERT quality filter
            filtered = self._sbert_filter(sent, candidates, threshold=0.80)

            if not filtered:
                result_sents.append(sent)
                continue

            # Thompson Sampling selection if learning engine available
            if learning_engine and hasattr(learning_engine, 'thompson_sample_sentence') and len(filtered) > 1:
                best = learning_engine.thompson_sample_sentence(sent, filtered)
            else:
                best = filtered[0]

            # Normalize punctuation and capitalization
            if not best.endswith(('.', '!', '?')):
                best = best.rstrip() + '.'
            if best and sent and best[0].islower() and sent[0].isupper():
                best = best[0].upper() + best[1:]

            result_sents.append(best)
            changes.append((sent.strip(), best.strip()))

        return ' '.join(result_sents), changes


class SelfLearningEngine:
    """
    Öğrenen ve gelişen dinamik motor
    Kullanıcı onaylarından öğrenir, domain-spesifik optimize eder
    """
    
    def __init__(self, data_dir: str = "learning_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.user_approvals_file = self.data_dir / "user_approvals.json"
        self.domain_profiles_file = self.data_dir / "domain_profiles.json"
        self.synonym_success_file = self.data_dir / "synonym_success.json"
        self.pattern_memory_file = self.data_dir / "pattern_memory.pkl"
        
        self.user_approvals = self._load_json(self.user_approvals_file, default=[])
        self.domain_profiles = self._load_json(self.domain_profiles_file, default={})
        self.synonym_success = self._load_json(self.synonym_success_file, default={})
        self.pattern_memory = self._load_pickle(self.pattern_memory_file, default={})
        
        # Domain detection için keyword bazlı sınıflandırıcı
        self.domain_keywords = {
            'nuclear_physics': ['reactor', 'fission', 'fusion', 'isotope', 'radioactive', 'uranium', 'plutonium', 'neutron', 'gamma', 'half-life'],
            'biology': ['cell', 'protein', 'enzyme', 'dna', 'rna', 'mitochondria', 'photosynthesis', 'organism', 'membrane', 'genome'],
            'history': ['empire', 'dynasty', 'revolution', 'war', 'treaty', 'civilization', 'ancient', 'medieval', 'colonial', 'monarchy'],
            'computer_science': ['algorithm', 'database', 'network', 'encryption', 'compiler', 'recursion', 'bandwidth', 'latency', 'throughput'],
            'medicine': ['diagnosis', 'symptom', 'pathogen', 'vaccine', 'antibody', 'surgery', 'pharmaceutical', 'clinical', 'epidemiology'],
            'economics': ['inflation', 'gdp', 'monetary', 'fiscal', 'supply', 'demand', 'market', 'investment', 'recession', 'trade'],
            'law': ['jurisdiction', 'precedent', 'statute', 'liability', 'plaintiff', 'defendant', 'tort', 'contract', 'constitutional'],
            'psychology': ['cognitive', 'behavioral', 'neurotransmitter', 'subconscious', 'conditioning', 'perception', 'personality', 'therapy']
        }
        
        self.learning_stats = {
            'total_interactions': len(self.user_approvals),
            'successful_transforms': sum(1 for a in self.user_approvals if a.get('approved', False)),
            'domains_learned': list(self.domain_profiles.keys()),
            'synonym_effectiveness': self._calculate_synonym_effectiveness()
        }
    
    def _load_json(self, filepath: Path, default: Any) -> Any:
        """JSON dosyasını yükle veya varsayılan değer döndür"""
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return default
        return default
    
    def _load_pickle(self, filepath: Path, default: Any) -> Any:
        """Pickle dosyasını yükle veya varsayılan değer döndür"""
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            except:
                return default
        return default
    
    def _save_json(self, data: Any, filepath: Path):
        """Veriyi JSON olarak kaydet"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_pickle(self, data: Any, filepath: Path):
        """Veriyi pickle olarak kaydet"""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def _calculate_synonym_effectiveness(self) -> Dict[str, float]:
        """Eş anlamlı kelimelerin başarı oranını hesapla"""
        effectiveness = {}
        for entry in self.user_approvals:
            if not entry.get('approved'):
                continue
            changes = entry.get('changes_made', [])
            for orig, new in changes:
                key = f"{orig}->{new}"
                if key not in effectiveness:
                    effectiveness[key] = {'success': 0, 'total': 0}
                effectiveness[key]['success'] += 1
                effectiveness[key]['total'] += 1
        
        # Normalize
        for key in effectiveness:
            effectiveness[key] = effectiveness[key]['success'] / effectiveness[key]['total']
        
        return effectiveness
    
    def detect_domain(self, text: str) -> Tuple[str, float]:
        """Metnin domain'ini tespit et ve güven skoru döndür"""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            matches = len(words & set(keywords))
            score = matches / len(keywords) if keywords else 0
            domain_scores[domain] = score
        
        if not domain_scores:
            return 'general', 0.0
        
        best_domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[best_domain]
        
        # Eşik değeri: 0.1 (en az %10 keyword eşleşmesi)
        if confidence < 0.1:
            return 'general', confidence
        
        return best_domain, confidence
    
    def get_domain_synonyms(self, domain: str) -> Dict[str, List[str]]:
        """Domain-spesifik eş anlamlıları getir veya genel listeyi genişlet"""
        base_synonyms = self._get_base_synonyms()
        
        if domain in self.domain_profiles:
            domain_specific = self.domain_profiles[domain].get('custom_synonyms', {})
            # Domain-specific olanları öncelikli ekle
            merged = base_synonyms.copy()
            for word, syns in domain_specific.items():
                if word in merged:
                    # Başarılı olanları öne koy
                    successful = [s for s in syns if self._get_success_rate(f"{word}->{s}") > 0.5]
                    unsuccessful = [s for s in syns if self._get_success_rate(f"{word}->{s}") <= 0.5]
                    merged[word] = successful + unsuccessful + merged[word]
                    # Tekrarları kaldır
                    seen = set()
                    merged[word] = [x for x in merged[word] if not (x in seen or seen.add(x))]
                else:
                    merged[word] = syns
            return merged
        
        return base_synonyms
    
    def record_user_feedback(self, original: str, transformed: str, approved: bool, 
                           changes_made: List[Tuple[str, str]], domain: str):
        """Kullanıcı geri bildirimini kaydet ve öğren"""
        timestamp = datetime.now().isoformat()
        text_hash = hashlib.md5(original.encode()).hexdigest()[:8]
        
        approval_record = {
            'timestamp': timestamp,
            'text_hash': text_hash,
            'approved': approved,
            'changes_made': changes_made,
            'domain': domain,
            'transform_length_ratio': len(transformed) / len(original) if original else 1.0
        }
        
        self.user_approvals.append(approval_record)
        self._save_json(self.user_approvals, self.user_approvals_file)
        
        # Thompson Sampling: update alpha/beta for each change pair (word or sentence level)
        for orig_item, new_item in changes_made:
            # Sentence-level changes are longer; use hash-based keys
            if len(orig_item) > 50:
                orig_hash = hashlib.md5(orig_item.strip().lower().encode()).hexdigest()[:12]
                cand_hash = hashlib.md5(new_item.strip().lower().encode()).hexdigest()[:12]
                key = f"s:{orig_hash}->{cand_hash}"
            else:
                key = f"{orig_item}->{new_item}"
            if key not in self.synonym_success or not isinstance(self.synonym_success[key], dict):
                self.synonym_success[key] = {"alpha": 1.0, "beta": 1.0}
            if approved:
                self.synonym_success[key]["alpha"] += 1.0
            else:
                self.synonym_success[key]["beta"] += 1.0
        
        # Domain profili güncelle
        if domain not in self.domain_profiles:
            self.domain_profiles[domain] = {
                'custom_synonyms': {},
                'successful_patterns': [],
                'avg_change_ratio': [],
                'interaction_count': 0
            }
        
        profile = self.domain_profiles[domain]
        profile['interaction_count'] += 1
        profile['avg_change_ratio'].append(approval_record['transform_length_ratio'])
        
        # Başarılı değişimleri domain-specific synonym listesine ekle
        if approved:
            for orig_word, new_word in changes_made:
                if orig_word not in profile['custom_synonyms']:
                    profile['custom_synonyms'][orig_word] = []
                if new_word not in profile['custom_synonyms'][orig_word]:
                    profile['custom_synonyms'][orig_word].insert(0, new_word)  # Başarılı olanları başa ekle
        
        self._save_json(self.domain_profiles, self.domain_profiles_file)
        self._save_json(self.synonym_success, self.synonym_success_file)
        
        # İstatistikleri güncelle
        self.learning_stats['total_interactions'] += 1
        if approved:
            self.learning_stats['successful_transforms'] += 1
    
    def _get_success_rate(self, key: str) -> float:
        """Get success rate from synonym_success, handling both old (float) and new (dict) formats."""
        data = self.synonym_success.get(key)
        if data is None:
            return 0.5  # uninformed prior
        if isinstance(data, dict):
            alpha = data.get("alpha", 1.0)
            beta_val = data.get("beta", 1.0)
            return alpha / (alpha + beta_val)
        # Legacy float format
        return float(data)

    def thompson_sample(self, word: str, candidates: List[str]) -> str:
        """Pick best candidate using Thompson Sampling (Beta distribution).
        Each synonym pair has alpha (success+1) and beta (failure+1) parameters.
        We sample from Beta(alpha, beta) for each candidate and pick the highest."""
        best_sample = -1.0
        best_candidate = candidates[0]
        for c in candidates:
            key = f"{word}->{c}"
            data = self.synonym_success.get(key, {})
            if isinstance(data, dict):
                alpha = data.get("alpha", 1.0)
                beta_val = data.get("beta", 1.0)
            else:
                # Legacy scalar → approximate alpha/beta
                rate = float(data) if data else 0.5
                alpha = max(1.0, rate * 5)
                beta_val = max(1.0, (1.0 - rate) * 5)
            sample = random.betavariate(alpha, beta_val)
            if sample > best_sample:
                best_sample = sample
                best_candidate = c
        return best_candidate

    def thompson_sample_sentence(self, original_sent: str, candidates: List[str]) -> str:
        """Pick best paraphrase candidate using Thompson Sampling.
        Keyed on sentence hash pairs for sentence-level tracking."""
        orig_hash = hashlib.md5(original_sent.strip().lower().encode()).hexdigest()[:12]
        best_sample = -1.0
        best_candidate = candidates[0]
        for c in candidates:
            cand_hash = hashlib.md5(c.strip().lower().encode()).hexdigest()[:12]
            key = f"s:{orig_hash}->{cand_hash}"
            data = self.synonym_success.get(key, {"alpha": 1.0, "beta": 1.0})
            if isinstance(data, dict):
                alpha = data.get("alpha", 1.0)
                beta_val = data.get("beta", 1.0)
            else:
                alpha, beta_val = 1.0, 1.0
            sample = random.betavariate(alpha, beta_val)
            if sample > best_sample:
                best_sample = sample
                best_candidate = c
        return best_candidate

    def get_intelligent_replacement_candidates(self, word: str, domain: str) -> List[str]:
        """Öğrenilmiş verilere göre en iyi adayları getir"""
        candidates = []

        # 1. Domain-specific başarılı eş anlamlılar
        if domain in self.domain_profiles:
            domain_syns = self.domain_profiles[domain].get('custom_synonyms', {}).get(word, [])
            candidates.extend(domain_syns)

        # 2. Genel başarılı eş anlamlılar (success rate > 0.5)
        base_syns = self._get_base_synonyms().get(word, [])
        successful_general = [
            syn for syn in base_syns
            if self._get_success_rate(f"{word}->{syn}") > 0.5
        ]
        candidates.extend(successful_general)

        # 3. Diğer eş anlamlılar
        candidates.extend([s for s in base_syns if s not in candidates])

        # Tekrarları kaldır, sıralı tut
        seen = set()
        return [x for x in candidates if not (x in seen or seen.add(x))]
    
    def _get_base_synonyms(self) -> Dict[str, List[str]]:
        """Temel eş anlamlı sözlüğü"""
        return {
            'important': ['crucial', 'essential', 'significant', 'vital', 'critical', 'key'],
            'show': ['demonstrate', 'display', 'reveal', 'indicate', 'exhibit', 'manifest'],
            'use': ['utilize', 'employ', 'apply', 'deploy', 'leverage', 'exploit'],
            'make': ['create', 'produce', 'generate', 'construct', 'formulate', 'craft'],
            'good': ['excellent', 'superior', 'fine', 'outstanding', 'exceptional', 'satisfactory'],
            'big': ['large', 'substantial', 'considerable', 'significant', 'extensive', 'major'],
            'think': ['consider', 'believe', 'judge', 'deem', 'regard', 'suppose'],
            'get': ['obtain', 'acquire', 'receive', 'secure', 'gain', 'attain'],
            'know': ['understand', 'comprehend', 'recognize', 'grasp', 'perceive', 'apprehend'],
            'go': ['proceed', 'move', 'advance', 'progress', 'continue', 'journey'],
            'see': ['observe', 'notice', 'perceive', 'discern', 'detect', 'witness'],
            'look': ['examine', 'inspect', 'view', 'scrutinize', 'survey', 'observe'],
            'want': ['desire', 'wish', 'seek', 'aim', 'intend', 'aspire'],
            'need': ['require', 'necessitate', 'demand', 'call for', 'warrant', 'entail'],
            'help': ['assist', 'aid', 'support', 'facilitate', 'enable', 'bolster'],
            'try': ['attempt', 'endeavor', 'strive', 'undertake', 'essay', 'assay'],
            'find': ['discover', 'locate', 'detect', 'identify', 'pinpoint', 'uncover'],
            'tell': ['inform', 'relate', 'narrate', 'convey', 'communicate', 'impart'],
            'ask': ['inquire', 'question', 'query', 'interrogate', 'consult', 'solicit'],
            'seem': ['appear', 'look', 'sound', 'strike', 'come across', 'give the impression'],
            'feel': ['sense', 'experience', 'perceive', 'undergo', 'endure', 'savor'],
            'leave': ['depart', 'exit', 'go', 'withdraw', 'retire', 'vacate'],
            'call': ['name', 'term', 'designate', 'denominate', 'label', 'classify'],
            'come': ['arrive', 'approach', 'reach', 'attain', 'enter', 'materialize'],
            'work': ['function', 'operate', 'perform', 'act', 'run', 'serve'],
            'live': ['reside', 'dwell', 'inhabit', 'occupy', 'populate', 'settle'],
            'believe': ['think', 'trust', 'accept', 'hold', 'maintain', 'presume'],
            'bring': ['fetch', 'carry', 'bear', 'convey', 'transport', 'deliver'],
            'happen': ['occur', 'transpire', 'take place', 'come about', 'ensue', 'result'],
            'write': ['compose', 'draft', 'author', 'pen', 'inscribe', 'record'],
            'provide': ['supply', 'furnish', 'give', 'render', 'afford', 'bestow'],
            'sit': ['rest', 'be seated', 'repose', 'perch', 'settle'],
            'stand': ['rise', 'be standing', 'arise', 'stand up', 'be erect'],
            'lose': ['misplace', 'forfeit', 'surrender', 'relinquish', 'cede', 'sacrifice'],
            'pay': ['compensate', 'remunerate', 'reimburse', 'remit', 'discharge', 'settle'],
            'meet': ['encounter', 'confront', 'face', 'run into', 'come across', 'greet'],
            'include': ['contain', 'comprise', 'include', 'incorporate', 'embrace', 'encompass'],
            'continue': ['persist', 'maintain', 'keep', 'carry on', 'proceed', 'persevere'],
            'set': ['establish', 'place', 'put', 'position', 'situate', 'install'],
            'learn': ['acquire', 'master', 'study', 'assimilate', 'absorb', 'grasp'],
            'change': ['alter', 'modify', 'adjust', 'transform', 'convert', 'vary'],
            'lead': ['guide', 'direct', 'conduct', 'steer', 'pilot', 'usher'],
            'understand': ['comprehend', 'grasp', 'apprehend', 'fathom', 'digest', 'assimilate'],
            'watch': ['observe', 'monitor', 'view', 'surveil', 'track', 'keep an eye on'],
            'follow': ['pursue', 'trail', 'track', 'shadow', 'chase', 'tail'],
            'stop': ['cease', 'halt', 'discontinue', 'desist', 'terminate', 'conclude'],
            'create': ['make', 'produce', 'form', 'fashion', 'forge', 'fabricate'],
            'speak': ['talk', 'say', 'utter', 'articulate', 'enunciate', 'voice'],
            'read': ['peruse', 'study', 'scan', 'scrutinize', 'examine', 'review'],
            'allow': ['permit', 'let', 'authorize', 'sanction', 'empower', 'enable'],
            'add': ['append', 'attach', 'include', 'affix', 'annex', 'tag on'],
            'spend': ['expend', 'disburse', 'use', 'consume', 'exhaust', 'deplete'],
            'grow': ['expand', 'increase', 'develop', 'mushroom', 'burgeon', 'proliferate'],
            'open': ['unlock', 'uncover', 'reveal', 'expose', 'unveil', 'disclose'],
            'walk': ['stroll', 'move', 'go', 'amble', 'saunter', 'stride'],
            'win': ['triumph', 'succeed', 'prevail', 'conquer', 'vanquish', 'overcome'],
            'offer': ['propose', 'suggest', 'present', 'extend', 'tender', 'proffer'],
            'remember': ['recall', 'recollect', 'remember', 'reminisce', 'retain', 'bear in mind'],
            'love': ['adore', 'cherish', 'treasure', 'idolize', 'worship', 'revere'],
            'consider': ['contemplate', 'evaluate', 'assess', 'weigh', 'ponder', 'deliberate'],
            'appear': ['seem', 'look', 'materialize', 'emerge', 'surface', 'arise'],
            'buy': ['purchase', 'acquire', 'procure', 'obtain', 'secure', 'buy up'],
            'wait': ['stay', 'remain', 'tarry', 'linger', 'bide', 'hold on'],
            'serve': ['attend', 'assist', 'help', 'minister to', 'accommodate', 'satisfy'],
            'die': ['perish', 'expire', 'decease', 'succumb', 'pass away', 'kick the bucket'],
            'send': ['dispatch', 'transmit', 'convey', 'forward', 'ship', 'route'],
            'expect': ['anticipate', 'await', 'look for', 'envisage', 'foresee', 'predict'],
            'build': ['construct', 'erect', 'make', 'fabricate', 'assemble', 'put up'],
            'stay': ['remain', 'abide', 'linger', 'tarry', 'sojourn', 'reside'],
            'fall': ['drop', 'descend', 'decrease', 'plummet', 'plunge', 'tumble'],
            'cut': ['sever', 'divide', 'separate', 'slice', 'carve', 'cleave'],
            'reach': ['attain', 'achieve', 'arrive at', 'gain', 'accomplish', 'fulfill'],
            'kill': ['slay', 'destroy', 'eliminate', 'annihilate', 'extinguish', 'eradicate'],
            'remain': ['stay', 'continue', 'persist', 'endure', 'last', 'abide'],
            'suggest': ['propose', 'recommend', 'advise', 'advocate', 'counsel', 'urge'],
            'raise': ['lift', 'elevate', 'increase', 'boost', 'heighten', 'uplift'],
            'pass': ['proceed', 'go', 'move', 'elapse', 'lapse', 'flow'],
            'sell': ['vend', 'market', 'trade', 'peddle', 'hawk', 'unload'],
            'require': ['need', 'demand', 'call for', 'necessitate', 'warrant', 'entail'],
            'report': ['announce', 'declare', 'state', 'communicate', 'notify', 'inform'],
            'decide': ['determine', 'resolve', 'settle', 'conclude', 'judge', 'adjudicate'],
            'pull': ['draw', 'drag', 'haul', 'tug', 'yank', 'heave'],
            'return': ['go back', 'come back', 'revert', 'retrogress', 'regress', 'retreat']
        }
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Öğrenme istatistiklerini raporla"""
        return {
            'total_interactions': self.learning_stats['total_interactions'],
            'success_rate': self.learning_stats['successful_transforms'] / max(1, self.learning_stats['total_interactions']),
            'domains_learned': self.learning_stats['domains_learned'],
            'domain_details': {
                domain: {
                    'interactions': data['interaction_count'],
                    'custom_synonyms_count': len(data.get('custom_synonyms', {})),
                    'avg_length_change': sum(data.get('avg_change_ratio', [1.0])) / max(1, len(data.get('avg_change_ratio', [1.0])))
                }
                for domain, data in self.domain_profiles.items()
            },
            'top_successful_synonyms': sorted(
                self.synonym_success.items(),
                key=lambda x: x[1]["alpha"] / (x[1]["alpha"] + x[1]["beta"]) if isinstance(x[1], dict) else x[1],
                reverse=True
            )[:10]
        }


class StrategyQLearner:
    """
    Q-Learning agent for strategy selection.
    Learns which text transformation strategy works best for a given state.

    State:   (domain, score_bucket, iteration)
    Action:  strategy name (light, balanced, structural, human-noise, kitchen-sink)
    Reward:  score_reduction * 10 (positive = AI score went down = good)

    Uses epsilon-greedy exploration with decay.
    Q-table persists to learning_data/q_table.json across runs.
    """

    STRATEGIES = ['light', 'balanced', 'structural', 'human-noise', 'kitchen-sink']

    def __init__(self, data_dir: str = "learning_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.q_file = self.data_dir / "q_table.json"
        self.q_table: Dict[str, Dict[str, float]] = self._load()
        self.lr = 0.15           # learning rate (alpha)
        self.gamma = 0.9         # discount factor
        self.epsilon = 0.4       # initial exploration rate
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.95  # per-episode decay
        # Load epsilon from disk if available
        eps_file = self.data_dir / "epsilon.json"
        if eps_file.exists():
            try:
                with open(eps_file) as f:
                    self.epsilon = json.load(f).get("epsilon", self.epsilon)
            except Exception:
                pass

    def _state_key(self, domain: str, score: float, iteration: int,
                    source_model: str = "unknown") -> str:
        """Discretize state into a hashable key."""
        score_bucket = round(score * 10) / 10  # 0.0, 0.1, ..., 1.0
        iter_bucket = min(iteration, 5)
        return f"{domain}|{score_bucket:.1f}|{iter_bucket}|{source_model}"

    def select_strategy_order(self, domain: str, score: float, iteration: int,
                              source_model: str = "unknown") -> List[str]:
        """Return strategies ordered by Q-value (epsilon-greedy).
        With probability epsilon, returns random order (exploration).
        Otherwise, returns sorted by Q-value descending (exploitation)."""
        state = self._state_key(domain, score, iteration, source_model)

        if random.random() < self.epsilon:
            # Explore: random order
            order = list(self.STRATEGIES)
            random.shuffle(order)
            return order

        # Exploit: sort by Q-value (highest first)
        q_values = self.q_table.get(state, {})
        scored = [(s, q_values.get(s, 0.0)) for s in self.STRATEGIES]
        scored.sort(key=lambda x: -x[1])
        return [s for s, _ in scored]

    def update(self, domain: str, score_before: float, iteration: int,
               strategy: str, score_after: float, source_model: str = "unknown"):
        """Update Q(state, action) after observing the result of a strategy."""
        state = self._state_key(domain, score_before, iteration, source_model)
        next_state = self._state_key(domain, score_after, iteration + 1, source_model)

        # Reward: how much did the AI score drop? (higher = better)
        reward = (score_before - score_after) * 10.0

        # Initialize Q-values if missing
        if state not in self.q_table:
            self.q_table[state] = {}
        if strategy not in self.q_table[state]:
            self.q_table[state][strategy] = 0.0

        # Max Q(s', a') for the next state
        next_q = self.q_table.get(next_state, {})
        max_next_q = max(next_q.values()) if next_q else 0.0

        # Temporal Difference update: Q(s,a) += lr * (R + gamma * max_Q(s') - Q(s,a))
        old_q = self.q_table[state][strategy]
        self.q_table[state][strategy] = old_q + self.lr * (
            reward + self.gamma * max_next_q - old_q
        )

        # Decay exploration rate
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        self._save()

    def get_q_report(self) -> Dict[str, Any]:
        """Summary of learned Q-values."""
        total_states = len(self.q_table)
        if total_states == 0:
            return {"states": 0, "epsilon": self.epsilon, "top_actions": []}

        # Find best strategy per state
        best_per_state = {}
        for state, actions in self.q_table.items():
            if actions:
                best = max(actions, key=actions.get)
                best_per_state[state] = (best, actions[best])

        # Strategy win count
        from collections import Counter
        wins = Counter(v[0] for v in best_per_state.values())

        return {
            "states": total_states,
            "epsilon": round(self.epsilon, 3),
            "strategy_wins": dict(wins.most_common()),
            "top_states": sorted(
                best_per_state.items(),
                key=lambda x: -x[1][1]
            )[:5],
        }

    def _load(self) -> Dict:
        if self.q_file.exists():
            try:
                with open(self.q_file) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        with open(self.q_file, 'w') as f:
            json.dump(self.q_table, f, indent=2)
        with open(self.data_dir / "epsilon.json", 'w') as f:
            json.dump({"epsilon": self.epsilon}, f)


class TextQualityAnalyzer:
    """
    GERÇEKÇİ metin analiz ve varyasyon aracı - Şimdi Self-Learning ile!
    Artık "evasion" yok - sadece dürüst metrikler, kalite analizi ve akıllı adaptasyon
    """
    
    def __init__(self, enable_learning: bool = True):
        self.enable_learning = enable_learning
        self.learning_engine = SelfLearningEngine() if enable_learning else None
        self._sbert_model = None  # Lazy-loaded Sentence-BERT
        self._t5 = T5SentenceParaphraser()  # T5 sentence-level paraphraser (fallback)
        self._llm_rewriter = LLMRewriter()  # Cross-model LLM rewriter

        self.connectors = [
            "Furthermore", "Moreover", "However", "Consequently", 
            "Nevertheless", "Additionally", "In contrast", "Therefore",
            "Subsequently", "Conversely", "Similarly", "Alternatively"
        ]
        
        self.transformation_history = []
        self.current_domain = 'general'
        
    def detect_content_domain(self, text: str) -> Tuple[str, float]:
        """Metnin domain'ini tespit et"""
        if self.learning_engine:
            return self.learning_engine.detect_domain(text)
        return 'general', 0.0
    
    def _load_sbert(self):
        """Lazy-load Sentence-BERT model (downloads ~80MB on first run)."""
        if self._sbert_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("   📦 Sentence-BERT loaded (all-MiniLM-L6-v2)")
            except ImportError:
                print("   ⚠️  sentence-transformers not installed, BERT filtering disabled")
                self._sbert_model = False  # Sentinel: don't retry
        return self._sbert_model if self._sbert_model is not False else None

    def bert_filter_sentences(self, original: str, modified: str,
                              threshold: float = 0.80) -> Tuple[str, int]:
        """
        Sentence-level BERT filter: revert any sentence where semantic similarity
        drops below threshold. Returns (filtered_text, num_reverted).
        Uses batch encoding for efficiency (~50ms for 7 sentences).
        When sentence counts differ (from burstiness splits/merges), uses
        whole-text similarity with a lower threshold (0.70) to avoid reverting
        all burstiness work.
        """
        model = self._load_sbert()
        if model is None:
            return modified, 0

        orig_sents = re.split(r'(?<=[.!?])\s+', original.strip())
        mod_sents = re.split(r'(?<=[.!?])\s+', modified.strip())

        # If sentence count changed (splits/merges from burstiness), use whole-text
        # similarity with a lower bar — we expect structural changes here
        if len(orig_sents) != len(mod_sents):
            embeddings = model.encode([original, modified])
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            sim = cos_sim([embeddings[0]], [embeddings[1]])[0][0]
            # Lower threshold (0.70) when sentence count differs — burstiness
            # splits/merges are intentional structural changes, not drift
            if sim >= 0.70:
                return modified, 0
            return original, len(mod_sents)

        # Batch encode all sentences at once
        all_sents = orig_sents + mod_sents
        all_embeddings = model.encode(all_sents)
        orig_embeds = all_embeddings[:len(orig_sents)]
        mod_embeds = all_embeddings[len(orig_sents):]

        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        result_sents = []
        reverted = 0
        for i in range(len(orig_sents)):
            if orig_sents[i].strip() == mod_sents[i].strip():
                result_sents.append(mod_sents[i])
                continue

            sim = cos_sim([orig_embeds[i]], [mod_embeds[i]])[0][0]
            if sim >= threshold:
                result_sents.append(mod_sents[i])
            else:
                result_sents.append(orig_sents[i])  # Revert
                reverted += 1

        return ' '.join(result_sents), reverted

    def _wordnet_synonyms(self, word: str, penn_tag: str) -> List[str]:
        """Get WordNet synonyms for a word — first synset only (most common sense)."""
        from nltk.corpus import wordnet as wn
        upos = _penn_to_upos(penn_tag)
        wn_pos_map = {'ADJ': wn.ADJ, 'VERB': wn.VERB, 'NOUN': wn.NOUN, 'ADV': wn.ADV}
        wn_pos = wn_pos_map.get(upos)

        synonyms = set()
        synsets = wn.synsets(word, pos=wn_pos) if wn_pos else wn.synsets(word)
        if synsets:
            # Only use first synset (most common sense) to avoid wrong-sense synonyms
            for lemma in synsets[0].lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() != word.lower() and ' ' not in name and len(name) > 2:
                    synonyms.add(name.lower())
        return list(synonyms)[:4]

    # Curated AI-telltale replacements: words AI overuses → human alternatives
    _AI_TELLTALE_SWAPS = {
        # ── Formal verbs → simpler human alternatives ─────────────────────
        'utilize': ['use', 'employ'], 'utilizes': ['uses', 'employs'],
        'utilized': ['used', 'employed'], 'utilizing': ['using', 'employing'],
        'demonstrate': ['show', 'reveal'], 'demonstrates': ['shows', 'reveals'],
        'demonstrated': ['showed', 'revealed'], 'demonstrating': ['showing'],
        'facilitate': ['help', 'enable', 'ease'], 'facilitates': ['helps', 'enables'],
        'facilitated': ['helped', 'enabled'], 'facilitating': ['helping', 'enabling'],
        'necessitate': ['need', 'require'], 'necessitates': ['needs', 'requires', 'calls for'],
        'necessitated': ['needed', 'required'], 'necessitating': ['needing', 'requiring'],
        'encompass': ['cover', 'include'], 'encompasses': ['covers', 'includes'],
        'encompassed': ['covered', 'included'], 'encompassing': ['covering', 'including'],
        'constitute': ['make up', 'form'], 'constitutes': ['makes up', 'forms'],
        'possess': ['have', 'hold', 'carry'], 'possesses': ['has', 'holds'],
        'underscore': ['highlight', 'stress'], 'underscores': ['highlights', 'stresses'],
        'delineate': ['outline', 'describe'], 'elucidate': ['clarify', 'explain'],
        'enhance': ['improve', 'boost', 'strengthen'], 'enhances': ['improves', 'boosts'],
        'enhancing': ['improving', 'boosting'], 'enhanced': ['improved', 'boosted'],
        'optimize': ['improve', 'tune', 'refine'], 'optimizing': ['improving', 'tuning'],
        'leverage': ['use', 'draw on', 'tap into'], 'leveraging': ['using', 'drawing on'],
        'leveraged': ['used', 'drew on'], 'leverages': ['uses', 'draws on'],
        'mitigate': ['reduce', 'lessen', 'ease'], 'mitigating': ['reducing', 'easing'],
        'mitigated': ['reduced', 'lessened'], 'mitigates': ['reduces', 'lessens'],
        'exacerbate': ['worsen', 'intensify'], 'exacerbating': ['worsening'],
        'exacerbated': ['worsened', 'intensified'],
        'exhibit': ['show', 'display'], 'exhibits': ['shows', 'displays'],
        'commence': ['start', 'begin'], 'commences': ['starts', 'begins'],
        'commenced': ['started', 'began'],
        'endeavor': ['try', 'attempt', 'effort'], 'endeavors': ['tries', 'attempts'],
        'ascertain': ['find out', 'figure out'], 'ascertained': ['found out'],
        'substantiate': ['back up', 'support', 'confirm'],
        'proliferate': ['spread', 'grow', 'multiply'],
        'proliferating': ['spreading', 'growing'],
        'perpetuate': ['continue', 'keep going'], 'perpetuated': ['continued', 'kept going'],
        'juxtapose': ['compare', 'contrast'], 'juxtaposed': ['compared', 'placed side by side'],
        'augment': ['add to', 'increase', 'boost'], 'augmented': ['increased', 'boosted'],
        'denote': ['mean', 'refer to', 'stand for'], 'denotes': ['means', 'refers to'],
        'foster': ['encourage', 'support', 'promote'], 'fostering': ['encouraging', 'supporting'],
        'fosters': ['encourages', 'promotes'],
        'garner': ['gain', 'earn', 'get'], 'garnered': ['gained', 'earned'],
        'illuminate': ['shed light on', 'clarify'], 'illuminates': ['sheds light on'],

        # ── Formal adjectives/adverbs → simpler ──────────────────────────
        'significant': ['major', 'big', 'notable'], 'significantly': ['greatly', 'a lot'],
        'comprehensive': ['thorough', 'broad', 'wide'],
        'crucial': ['key', 'vital', 'major'], 'essential': ['needed', 'core', 'key'],
        'numerous': ['many', 'several', 'plenty of'],
        'primarily': ['mainly', 'mostly'], 'predominantly': ['mostly', 'largely'],
        'approximately': ['about', 'roughly', 'around'],
        'particularly': ['especially', 'notably'],
        'specifically': ['namely', 'in particular'],
        'inherently': ['naturally', 'by nature'],
        'increasingly': ['more and more', 'ever more'],
        'imperative': ['vital', 'urgent', 'pressing'],
        'multifaceted': ['complex', 'varied'], 'multidisciplinary': ['cross-field', 'varied'],
        'paradigm': ['model', 'framework'], 'robust': ['strong', 'solid', 'sturdy'],
        'pivotal': ['key', 'central', 'critical'], 'intricate': ['complex', 'detailed'],
        'enigmatic': ['mysterious', 'puzzling'], 'invaluable': ['priceless', 'precious'],
        'inadequate': ['lacking', 'poor', 'weak'],
        'remarkable': ['striking', 'notable'], 'notable': ['important', 'major'],
        'contingent': ['dependent', 'based'], 'requisite': ['needed', 'required'],
        'characterized': ['marked', 'defined', 'known for'],
        'comprising': ['including', 'made up of'],
        'paramount': ['top', 'most important', 'key'],
        'pertinent': ['relevant', 'related'], 'prevalent': ['common', 'widespread'],
        'profound': ['deep', 'strong', 'serious'], 'profoundly': ['deeply', 'greatly'],
        'substantial': ['large', 'big', 'major'], 'substantially': ['a lot', 'greatly'],
        'detrimental': ['harmful', 'damaging', 'bad'],
        'conducive': ['helpful', 'good for'], 'ubiquitous': ['everywhere', 'common'],
        'indispensable': ['vital', 'must-have'], 'overarching': ['main', 'broad'],
        'unprecedented': ['never before seen', 'first-ever'],
        'burgeoning': ['growing', 'booming'], 'seminal': ['landmark', 'groundbreaking'],
        'holistic': ['whole', 'complete', 'overall'],
        'myriad': ['many', 'countless'], 'plethora': ['lots of', 'a ton of', 'many'],
        'aforementioned': ['mentioned', 'noted above'],
        'concomitant': ['accompanying', 'related'], 'efficacious': ['effective', 'useful'],

        # ── Formal connectors → casual ───────────────────────────────────
        'subsequently': ['then', 'later', 'after that'],
        'therefore': ['so', 'thus'], 'thereby': ['by doing so', 'this way'],
        'however': ['but', 'still', 'yet'], 'moreover': ['also', 'plus', 'and'],
        'furthermore': ['also', 'besides', 'what is more'],
        'nevertheless': ['still', 'even so', 'yet'],
        'consequently': ['so', 'as a result'],
        'additionally': ['also', 'on top of that'],
        'notwithstanding': ['despite', 'even though'],
        'henceforth': ['from now on', 'going forward'],
        'wherein': ['where', 'in which'],
        'whereby': ['by which', 'through which'],
        'whilst': ['while', 'as'],
        'regarding': ['about', 'on'],

        # ── Formal nouns → simpler ───────────────────────────────────────
        'establishment': ['creation', 'setup'], 'preservation': ['protection', 'upkeep'],
        'integration': ['combining', 'blending'], 'implementation': ['carrying out', 'rollout'],
        'collaboration': ['teamwork', 'cooperation'], 'formulation': ['creation', 'design'],
        'advancement': ['progress', 'growth'], 'dynamics': ['workings', 'interactions'],
        'vulnerability': ['weakness', 'exposure'], 'disturbances': ['disruptions', 'upsets'],
        'constraints': ['limits', 'restrictions'],
        'informing': ['guiding', 'shaping'], 'ensuring': ['making sure', 'guaranteeing'],
        'addressing': ['tackling', 'dealing with'], 'harnessing': ['tapping', 'using'],
        'ramifications': ['effects', 'consequences'], 'implications': ['effects', 'results'],
        'methodology': ['method', 'approach'], 'methodologies': ['methods', 'approaches'],
        'mechanisms': ['ways', 'processes'], 'mechanism': ['way', 'process'],
        'phenomenon': ['event', 'thing'], 'phenomena': ['events', 'things'],
        'manifestation': ['sign', 'expression'], 'manifestations': ['signs', 'expressions'],
        'trajectory': ['path', 'direction', 'trend'],
        'trajectories': ['paths', 'directions', 'trends'],
        'discourse': ['discussion', 'debate', 'talk'],
        'dichotomy': ['split', 'divide', 'contrast'],
        'heterogeneity': ['variety', 'diversity', 'mix'],
        'juxtaposition': ['contrast', 'comparison'],
        'underpinning': ['basis', 'foundation'], 'underpinnings': ['foundations', 'bases'],
        'stakeholders': ['people involved', 'parties'],
        'synergies': ['combined effects', 'benefits'],
        'framework': ['structure', 'setup', 'system'],
        'frameworks': ['structures', 'systems'],
    }

    # Phrase-level AI-telltale replacements (applied before word-level)
    _AI_TELLTALE_PHRASES = {
        'in order to': 'to',
        'a wide range of': 'many different',
        'a vast array of': 'many different',
        'a myriad of': 'many',
        'plays a crucial role': 'matters a lot',
        'plays a vital role': 'is really important',
        'plays a significant role': 'matters a lot',
        'plays an important role': 'matters',
        'it is worth noting that': 'notably,',
        'it is important to note that': 'keep in mind,',
        'it should be noted that': 'note that',
        'in the context of': 'when it comes to',
        'with respect to': 'about',
        'in light of': 'given',
        'on the other hand': 'then again',
        'as a consequence': 'because of this',
        'in conjunction with': 'along with',
        'for the purpose of': 'to',
        'in the realm of': 'in',
        'serves as a': 'is a',
        'serves as an': 'is an',
        'has the potential to': 'could',
        'it is essential to': 'we need to',
        'it is crucial to': 'we must',
        'a growing body of': 'more and more',
        'the overarching goal': 'the main goal',
        'from a broader perspective': 'looking at the big picture',
        'of paramount importance': 'very important',
        'is characterized by': 'is known for',
        'in contemporary society': 'today',
        'prior to': 'before',
    }

    def _word_semantic_ok(self, original_word: str, candidate: str,
                          threshold: float = 0.70) -> bool:
        """Word-level semantic check using Sentence-BERT.
        Embeds both words in a short context and compares cosine similarity.
        Catches 'future' → 'economic' type errors that sentence-level filter misses."""
        model = self._load_sbert()
        if model is None:
            return True  # can't check, allow
        ctx_orig = f"The {original_word} is important"
        ctx_cand = f"The {candidate} is important"
        embs = model.encode([ctx_orig, ctx_cand])
        from numpy import dot
        from numpy.linalg import norm
        sim = float(dot(embs[0], embs[1]) / (norm(embs[0]) * norm(embs[1])))
        return sim >= threshold

    def safe_synonym_replace(self, text: str, max_ratio: float = 0.25,
                            domain: str = 'general',
                            original_text: str = None,
                            use_learned_synonyms: bool = False) -> Tuple[str, List[Tuple[str, str]]]:
        """Targeted word replacement focusing on AI-telltale vocabulary.
        Uses curated safe swaps + static base synonyms only (no learned data by default).
        Protects domain-critical terms. Word-level SBERT guard rejects bad synonyms."""

        result = text
        changes_made = []

        # Phase A: Multi-word phrase replacements (highest impact, always safe)
        for phrase, replacement in self._AI_TELLTALE_PHRASES.items():
            if phrase in result.lower():
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                def _phrase_replace(m):
                    orig = m.group(0)
                    if orig[0].isupper():
                        return replacement[0].upper() + replacement[1:]
                    return replacement
                new_result = pattern.sub(_phrase_replace, result, count=1)
                if new_result != result:
                    result = new_result
                    changes_made.append((phrase, replacement))

        # Phase B: Single-word AI-telltale replacements (curated, always safe)
        # Skip matches inside parentheses or em-dash asides (injected content)
        def _outside_paren(m):
            """Return True if match position is NOT inside (...) or — ... —."""
            s = m.start()
            prefix = result[:s]
            # Check parentheses
            if prefix.count('(') > prefix.count(')'):
                return False
            # Check em-dashes: odd count of — before match means inside aside
            if prefix.count('—') % 2 == 1:
                return False
            return True

        for ai_word, human_alts in self._AI_TELLTALE_SWAPS.items():
            if ai_word in result.lower():
                replacement = random.choice(human_alts)
                pattern = re.compile(re.escape(ai_word), re.IGNORECASE)
                matches = [m for m in pattern.finditer(result) if _outside_paren(m)]
                if matches:
                    m = matches[0]  # replace first outside-paren match only
                    orig = m.group(0)
                    if orig[0].isupper():
                        repl = replacement[0].upper() + replacement[1:]
                    else:
                        repl = replacement
                    result = result[:m.start()] + repl + result[m.end():]
                    changes_made.append((ai_word, replacement))

        # Static base synonyms only — no learned/domain synonyms unless explicitly enabled
        if use_learned_synonyms and self.learning_engine:
            static_syns = self.learning_engine.get_domain_synonyms(domain)
        else:
            static_syns = self.learning_engine._get_base_synonyms() if self.learning_engine else self._load_static_synonyms()

        ref = original_text if original_text else text
        ref_words = [re.sub(r'[^\w]', '', w.lower()) for w in ref.split() if len(w) > 3]
        word_freq = Counter(ref_words)
        protected = frozenset(w for w, c in word_freq.items() if c >= 2)
        if self.learning_engine and domain in self.learning_engine.domain_keywords:
            protected = protected | frozenset(self.learning_engine.domain_keywords[domain])
        # Also protect words already in AI telltale dict (already handled above)
        protected = protected | frozenset(self._AI_TELLTALE_SWAPS.keys())

        tokens = nltk.word_tokenize(result)
        tagged = nltk.pos_tag(tokens)
        max_changes = max(2, int(len(tokens) * max_ratio * 0.5))
        result_tokens = list(tokens)

        eligible = []
        for i, (word, penn_tag) in enumerate(tagged):
            clean = re.sub(r'[^\w]', '', word.lower())
            if (len(clean) < 5
                    or clean in _STOPWORDS
                    or clean in _AMBIGUOUS_WORDS
                    or clean in protected
                    or penn_tag in ('NNP', 'NNPS')
                    or not clean.isalpha()):
                continue
            eligible.append(i)

        random.shuffle(eligible)
        changed = 0
        for i in eligible[:max_changes * 2]:
            if changed >= max_changes:
                break
            word, penn_tag = tagged[i]
            clean = re.sub(r'[^\w]', '', word.lower())
            upos = _penn_to_upos(penn_tag)
            synonym = None

            # Static dictionary only (no WordNet for remaining words)
            if clean in static_syns:
                fallback = static_syns[clean]
                if fallback:
                    candidate = random.choice(fallback)
                    if upos:
                        inflected = getInflection(candidate, tag=penn_tag)
                        if inflected and inflected[0].lower() != word.lower():
                            candidate = inflected[0]
                    if candidate.lower() != word.lower():
                        if candidate.lower() not in _spell.unknown([candidate.lower()]):
                            # Word-level semantic guard
                            if self._word_semantic_ok(clean, candidate.lower()):
                                synonym = candidate

            if synonym:
                if word[0].isupper() and synonym[0].islower():
                    synonym = synonym[0].upper() + synonym[1:]
                trailing = ''
                for ch in reversed(word):
                    if not ch.isalpha():
                        trailing = ch + trailing
                    else:
                        break
                result_tokens[i] = synonym + trailing
                changed += 1
                changes_made.append((clean, synonym.lower()))

        reconstructed = _reconstruct_text(result, tokens, result_tokens)
        reconstructed = _fix_articles(reconstructed)
        return reconstructed, changes_made

    def advanced_restructure(self, text: str, aggressiveness: float = 0.3) -> str:
        """
        Advanced sentence structure variation with multiple transforms:
        1. Connector swap/removal
        2. Short sentence merging
        3. Passive ↔ Active voice (improved)
        4. Nominalization ("X analyzed Y" → "The analysis of Y by X")
        5. Clause fronting ("Z happened because X" → "Because X, Z happened")
        6. Appositive insertion
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []

        i = 0
        while i < len(sentences):
            sent = sentences[i]

            # Skip very short sentences
            if len(sent.split()) < 5:
                result.append(sent)
                i += 1
                continue

            # ── 1. Connector swap/removal ──
            for conn in self.connectors:
                if sent.startswith(conn):
                    if random.random() < aggressiveness:
                        if random.random() < 0.5:
                            sent = sent[len(conn):].strip()
                            if sent:
                                sent = sent[0].upper() + sent[1:]
                        else:
                            new_conn = random.choice([c for c in self.connectors if c != conn])
                            sent = new_conn + sent[len(conn):]
                    break

            # ── 2. Merge two short sentences ──
            if (i < len(sentences) - 1
                    and len(sent.split()) < 10
                    and len(sentences[i + 1].split()) < 10
                    and random.random() < aggressiveness * 0.5):
                next_sent = sentences[i + 1]
                connector = random.choice(['; moreover,', '; however,', ', and', ', while'])
                combined = sent.rstrip('.') + connector + ' ' + next_sent.lstrip()
                if combined and combined[0].islower():
                    combined = combined[0].upper() + combined[1:]
                result.append(combined)
                i += 2
                continue

            # ── 3. Passive ↔ Active voice (improved) ──
            if random.random() < aggressiveness * 0.4:
                # Active → Passive: "X verb-ed Y" → "Y was verb-ed by X"
                m = re.match(
                    r'^((?:The |Our |This |A )\w+(?:\s+\w+)?)\s+'
                    r'(examines|investigates|analyzes|explores|addresses|assesses|considers)\s+'
                    r'(.+)',
                    sent
                )
                if m:
                    subj, verb, obj = m.group(1), m.group(2), m.group(3)
                    # Map active verb to passive
                    passive_map = {
                        'examines': 'is examined by', 'investigates': 'is investigated by',
                        'analyzes': 'is analyzed by', 'explores': 'is explored by',
                        'addresses': 'is addressed by', 'assesses': 'is assessed by',
                        'considers': 'is considered by',
                    }
                    passive_v = passive_map.get(verb)
                    if passive_v:
                        obj_cap = obj[0].upper() + obj[1:] if obj else obj
                        sent = f"{obj_cap.rstrip('.')} {passive_v} {subj.lower()}."

            # ── 4. Nominalization ──
            if random.random() < aggressiveness * 0.3:
                # "We will analyze X" → "Our analysis of X"
                nominalizations = {
                    r'\b[Ww]e\s+will\s+analyze\b': 'Our analysis covers',
                    r'\b[Ww]e\s+will\s+explore\b': 'Our exploration encompasses',
                    r'\b[Ww]e\s+will\s+examine\b': 'Our examination addresses',
                    r'\b[Ww]e\s+address\b': 'Our discussion covers',
                    r'\binvestigates\s+how\b': 'looks into how',
                    r'\bexamines\s+the\b': 'provides an examination of the',
                }
                for pat, repl in nominalizations.items():
                    new_sent = re.sub(pat, repl, sent, count=1)
                    if new_sent != sent:
                        sent = new_sent
                        break

            # ── 5. Clause fronting ──
            if random.random() < aggressiveness * 0.3:
                # "X because Y" → "Because Y, X"
                m = re.match(r'^(.+?)\s+(because|since|as|although|while)\s+(.+)$', sent, re.IGNORECASE)
                if m and len(m.group(1).split()) > 3 and len(m.group(3).split()) > 3:
                    main, conj, sub = m.group(1), m.group(2), m.group(3)
                    main = main.rstrip('.,')
                    if main and main[0].isupper():
                        main = main[0].lower() + main[1:]
                    sub_cap = sub[0].upper() + sub[1:] if sub else sub
                    sent = f"{conj.capitalize()} {sub.rstrip('.')}, {main}."

            # ── 6. Appositive/parenthetical insertion ──
            if random.random() < aggressiveness * 0.2:
                appositives = {
                    'Chernobyl Exclusion Zone': 'the Chernobyl Exclusion Zone, a 2,600 km² restricted area,',
                    'dark tourism': 'dark tourism (visiting sites associated with death or tragedy)',
                    'Przewalski\'s horses': 'Przewalski\'s horses, a rare species reintroduced to the area,',
                }
                for term, replacement in appositives.items():
                    if term in sent and replacement not in sent:
                        sent = sent.replace(term, replacement, 1)
                        break

            result.append(sent)
            i += 1

        return ' '.join(result)
    
    def typo_inject(self, text: str, count: int = 2) -> str:
        """
        Inject subtle, human-like typos to break AI text patterns.
        Types: adjacent char swap, double letter, dropped letter.
        Only targets common words (4+ chars), never proper nouns or numbers.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return text

        injected = 0
        result_sents = []

        for sent in sentences:
            if injected >= count:
                result_sents.append(sent)
                continue

            words = sent.split()
            eligible = [
                (i, w) for i, w in enumerate(words)
                if len(w) >= 4 and w.isalpha() and w[0].islower()
                and w.lower() not in _STOPWORDS
            ]

            if eligible and random.random() < 0.4:
                idx, word = random.choice(eligible)
                typo_type = random.choice(['swap', 'double', 'drop'])

                original_word = word
                if typo_type == 'swap' and len(word) >= 4:
                    # Swap two adjacent interior characters
                    pos = random.randint(1, len(word) - 3)
                    word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
                elif typo_type == 'double' and len(word) >= 4:
                    # Double an interior letter
                    pos = random.randint(1, len(word) - 2)
                    word = word[:pos] + word[pos] + word[pos:]
                elif typo_type == 'drop' and len(word) >= 5:
                    # Drop an interior letter
                    pos = random.randint(1, len(word) - 2)
                    word = word[:pos] + word[pos + 1:]

                # Spell-check: revert if typo created a non-word
                if word.lower() in _spell.unknown([word.lower()]):
                    word = original_word  # revert
                else:
                    injected += 1

                words[idx] = word

            result_sents.append(' '.join(words))

        return ' '.join(result_sents)

    def add_filler_phrases(self, text: str, count: int = 2) -> str:
        """
        Insert natural filler/hedging phrases at sentence boundaries.
        These are typical in human academic writing but absent in AI text.
        """
        fillers_start = [
            "It is worth noting that", "Interestingly,", "As one might expect,",
            "To some extent,", "In a sense,", "Broadly speaking,",
            "It appears that", "Evidence suggests that",
            "In practical terms,", "Looking at this more closely,",
        ]
        fillers_mid = [
            "— at least in part —", ", so to speak,", ", in a way,",
            ", arguably,", ", to be fair,", ", in this context,",
        ]

        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 3:
            return text

        added = 0
        result = []
        # Don't modify first or last sentence
        for i, sent in enumerate(sentences):
            if added >= count or i == 0 or i == len(sentences) - 1:
                result.append(sent)
                continue

            # Skip if sentence already starts with a connector/filler
            _starts_with_connector = any(
                sent.lower().startswith(c.lower())
                for c in ['also', 'but', 'however', 'on top of', 'that said',
                           'plus', 'still', 'yet', 'in practical', 'interestingly',
                           'broadly', 'to some extent', 'this', 'these', 'those']
            )
            if random.random() < 0.3 and not _starts_with_connector:
                choice = random.random()
                if choice < 0.6:
                    # Prepend a filler phrase
                    filler = random.choice(fillers_start)
                    # Lowercase the sentence start if filler is a full clause
                    if filler.endswith("that") or filler.endswith("that,"):
                        sent = sent[0].lower() + sent[1:]
                    result.append(f"{filler} {sent}")
                else:
                    # Insert mid-sentence filler after first clause
                    comma_pos = sent.find(',')
                    if comma_pos > 10 and comma_pos < len(sent) - 20:
                        filler = random.choice(fillers_mid)
                        sent = sent[:comma_pos] + filler + sent[comma_pos + 1:]
                    result.append(sent)
                added += 1
            else:
                result.append(sent)

        return ' '.join(result)

    def split_long_sentences(self, text: str) -> str:
        """
        Split long sentences at natural break points (conjunctions, semicolons).
        Only targets sentences > 25 words. Preserves meaning.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []

        for sent in sentences:
            words = sent.split()
            if len(words) <= 25:
                result.append(sent)
                continue

            # Try to split at natural points
            split_patterns = [
                r',\s+and\s+', r',\s+but\s+', r',\s+while\s+',
                r',\s+whereas\s+', r';\s+', r',\s+which\s+',
            ]

            split_done = False
            for pattern in split_patterns:
                m = re.search(pattern, sent)
                if m and m.start() > 20 and (len(sent) - m.end()) > 20:
                    first = sent[:m.start()].rstrip(',;') + '.'
                    second = sent[m.end():].strip()
                    # Capitalize second part
                    if second and second[0].islower():
                        # For "which" splits, add a subject
                        if 'which' in pattern:
                            second = 'This ' + second
                        else:
                            second = second[0].upper() + second[1:]
                    result.append(first)
                    result.append(second)
                    split_done = True
                    break

            if not split_done:
                result.append(sent)

        return ' '.join(result)

    def reorder_sentences(self, text: str) -> str:
        """
        Reorder sentences within paragraphs. Keeps first and last sentences
        in place (topic sentence + concluding sentence). Only swaps middle sentences.
        """
        paragraphs = text.split('\n\n') if '\n\n' in text else [text]
        result_paragraphs = []

        for para in paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', para.strip())
            if len(sentences) <= 3:
                result_paragraphs.append(para)
                continue

            # Keep first and last, shuffle middle
            first = sentences[0]
            last = sentences[-1]
            middle = sentences[1:-1]

            # Only swap 1-2 adjacent pairs (subtle reordering)
            swaps = min(2, len(middle) - 1)
            for _ in range(swaps):
                if len(middle) >= 2 and random.random() < 0.5:
                    idx = random.randint(0, len(middle) - 2)
                    middle[idx], middle[idx + 1] = middle[idx + 1], middle[idx]

            reordered = [first] + middle + [last]
            result_paragraphs.append(' '.join(reordered))

        return '\n\n'.join(result_paragraphs)

    def calculate_real_metrics(self, original: str, transformed: str) -> Dict[str, Any]:
        """
        GERÇEK, HESAPLANMIŞ metrikler - uydurma YOK
        """
        orig_words = original.split()
        trans_words = transformed.split()
        
        # 1. GERÇEK kelime değişim oranı
        orig_set = set(w.lower() for w in orig_words)
        trans_set = set(w.lower() for w in trans_words)
        
        common_words = len(orig_set & trans_set)
        total_unique = len(orig_set)
        
        word_change_ratio = (total_unique - common_words) / total_unique if total_unique > 0 else 0
        
        # 2. GERÇEK n-gram benzerliği (4-gram)
        def ngram_similarity(n=4):
            if len(orig_words) < n or len(trans_words) < n:
                return 0.0
            
            orig_ngrams = set()
            for i in range(len(orig_words) - n + 1):
                orig_ngrams.add(' '.join(orig_words[i:i+n]).lower())
            
            matches = 0
            for i in range(len(trans_words) - n + 1):
                ngram = ' '.join(trans_words[i:i+n]).lower()
                if ngram in orig_ngrams:
                    matches += 1
            
            total_possible = len(trans_words) - n + 1
            return matches / total_possible if total_possible > 0 else 0
        
        ngram_sim = ngram_similarity(4)
        
        # 3. Okunabilirlik skoru (geliştirilmiş Flesch benzeri)
        def readability(text):
            sentences = len(re.split(r'[.!?]+', text))
            words = len(text.split())
            syllables = len(re.findall(r'[aeiouAEIOU]+', text))
            
            if sentences == 0 or words == 0:
                return 0
            
            avg_words_per_sentence = words / sentences
            avg_syllables_per_word = syllables / words
            
            # Basit Flesch Reading Ease formülü
            score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
            # Normalize to 0-1
            normalized = max(0, min(1, score / 100))
            return normalized
        
        readability_score = readability(transformed)
        
        # 4. Anlam kaybı tahmini (geliştirilmiş)
        def meaning_preservation():
            # Önemli içerik kelimeleri (4+ harf) ve domain-specific terimler korunmuş mu?
            orig_content = [w.lower() for w in orig_words if len(w) > 4]
            trans_content = [w.lower() for w in trans_words if len(w) > 4]
            
            if not orig_content:
                return 1.0
            
            # Ağırlıklı eşleşme (kelime sıklığı dikkate alınarak)
            orig_counter = Counter(orig_content)
            preserved_score = 0
            total_weight = 0
            
            for word, count in orig_counter.items():
                weight = math.log(count + 1)  # Logaritmik ağırlık
                total_weight += weight
                if word in trans_content:
                    preserved_score += weight
            
            return preserved_score / total_weight if total_weight > 0 else 0
        
        meaning_score = meaning_preservation()
        
        # 5. UZUNLUK değişimi
        length_ratio = len(transformed) / len(original) if original else 1.0
        
        # 6. Semantic drift tahmini (basit)
        # Çok kısa kelimelerin (1-2 harf) değişim oranı (bu genellikle anlamı bozar)
        short_orig = [w for w in orig_words if len(w) <= 2]
        short_trans = [w for w in trans_words if len(w) <= 2]
        short_change = abs(len(short_orig) - len(short_trans)) / max(len(short_orig), 1)
        
        # 7. GERÇEKÇİ değerlendirme
        issues = []
        if word_change_ratio < 0.1:
            issues.append("MINIMAL CHANGE")
        elif word_change_ratio > 0.6:
            issues.append("EXCESSIVE CHANGE")
        
        if ngram_sim > 0.7:
            issues.append("HIGH N-GRAM OVERLAP")
        
        if readability_score < 0.3:
            issues.append("POOR READABILITY")
        
        if meaning_score < 0.6:
            issues.append("MEANING DRIFT")
        
        if length_ratio > 1.5 or length_ratio < 0.7:
            issues.append("LENGTH DISTORTION")
        
        if short_change > 0.3:
            issues.append("STRUCTURAL DAMAGE")
        
        if not issues:
            assessment = "✅ BALANCED - Adequate variation with preserved meaning"
        else:
            assessment = "⚠️ " + " | ".join(issues)
        
        return {
            'word_change_ratio': word_change_ratio,
            'ngram_similarity': ngram_sim,
            'readability_score': readability_score,
            'meaning_preservation': meaning_score,
            'length_ratio': length_ratio,
            'original_word_count': len(orig_words),
            'transformed_word_count': len(trans_words),
            'assessment': assessment,
            'warning': "These are REAL metrics, not guarantees of detection evasion",
            'learning_enabled': self.enable_learning,
            'domain': self.current_domain
        }
    
    def generate_llm_variant(self, text: str, source_model: str = 'unknown') -> tuple:
        """Single-pass LLM rewrite via cross-model API.
        Returns (rewritten_text, changes_list) or None if unavailable/failed."""
        if not self._llm_rewriter.available:
            return None

        rewritten = self._llm_rewriter.rewrite(text, source_model=source_model)
        if not rewritten:
            return None

        # Sanity check: rewritten text should be reasonable length
        ratio = len(rewritten) / len(text) if text else 0
        if ratio < 0.5 or ratio > 2.0:
            print(f"   ⚠️  LLM rewrite length ratio {ratio:.2f} — rejected")
            return None

        # Build a changes list (whole-text level)
        changes = [("llm_rewrite", f"via {self._llm_rewriter._pick_provider(source_model)}")]
        return rewritten, changes

    # ── Perplexity injection methods ─────────────────────────────────────
    def rare_synonym_replace(self, text: str, max_ratio: float = 0.15,
                             original_text: str = None) -> Tuple[str, List[Tuple[str, str]]]:
        """Replace content words with less common (higher perplexity) synonyms.
        Picks from the bottom 30% of WordNet synonyms by word frequency."""
        from wordfreq import word_frequency
        from nltk.corpus import wordnet as wn

        ref = original_text if original_text else text
        ref_words = [re.sub(r'[^\w]', '', w.lower()) for w in ref.split() if len(w) > 3]
        word_freq = Counter(ref_words)
        protected = frozenset(w for w, c in word_freq.items() if c >= 2)
        if self.learning_engine and self.current_domain in self.learning_engine.domain_keywords:
            protected = protected | frozenset(
                self.learning_engine.domain_keywords[self.current_domain])

        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        max_changes = max(2, int(len(tokens) * max_ratio))
        result_tokens = list(tokens)
        changes = []

        # Mark tokens inside parentheses or em-dash asides as off-limits
        in_paren = False
        in_dash = False
        shielded = set()
        for i, (word, _) in enumerate(tagged):
            if word == '(' or word == '(':
                in_paren = True
            if word == '—' and not in_dash:
                in_dash = True
                shielded.add(i)
                continue
            elif word == '—' and in_dash:
                in_dash = False
                shielded.add(i)
                continue
            if in_paren or in_dash:
                shielded.add(i)
            if word == ')' or word == ')':
                in_paren = False

        eligible = []
        for i, (word, penn_tag) in enumerate(tagged):
            clean = re.sub(r'[^\w]', '', word.lower())
            if (len(clean) < 4
                    or clean in _STOPWORDS or clean in _AMBIGUOUS_WORDS
                    or clean in protected
                    or penn_tag in ('NNP', 'NNPS')
                    or not clean.isalpha()
                    or not penn_tag[0] in ('N', 'V', 'J', 'R')
                    or i in shielded):
                continue
            eligible.append(i)

        random.shuffle(eligible)
        changed = 0
        for i in eligible:
            if changed >= max_changes:
                break
            word, penn_tag = tagged[i]
            clean = re.sub(r'[^\w]', '', word.lower())
            upos = _penn_to_upos(penn_tag)

            # Get WordNet synonyms
            wn_pos_map = {'ADJ': wn.ADJ, 'VERB': wn.VERB, 'NOUN': wn.NOUN, 'ADV': wn.ADV}
            wn_pos = wn_pos_map.get(upos)
            synsets = wn.synsets(clean, pos=wn_pos) if wn_pos else wn.synsets(clean)
            candidates = set()
            for ss in synsets[:3]:  # first 3 synsets (common senses)
                for lemma in ss.lemmas():
                    name = lemma.name().replace('_', ' ')
                    if (name.lower() != clean
                            and ' ' not in name
                            and len(name) > 2
                            and name.lower() not in _STOPWORDS
                            and name.lower() not in _AMBIGUOUS_WORDS):
                        candidates.add(name.lower())

            if not candidates:
                continue

            # Score by frequency — pick from bottom 30% (less common = higher perplexity)
            scored = [(c, word_frequency(c, 'en')) for c in candidates]
            scored.sort(key=lambda x: x[1])
            cutoff = max(1, int(len(scored) * 0.3))
            rare_pool = scored[:cutoff]

            # Try each rare candidate (least common first)
            synonym = None
            for cand, freq in rare_pool:
                # Reject extremely rare words (freq < 1e-8) — they sound unnatural
                if freq < 1e-8:
                    continue
                # Spell check base form
                if cand in _spell.unknown([cand]):
                    continue
                # POS inflect
                inflected = cand
                if upos:
                    infl = getInflection(cand, tag=penn_tag)
                    if infl and infl[0].lower() != word.lower():
                        inflected = infl[0]
                if inflected.lower() == word.lower():
                    continue
                # Spell check inflected form too
                if inflected.lower() in _spell.unknown([inflected.lower()]):
                    continue
                # Word-level semantic guard
                if self._word_semantic_ok(clean, inflected.lower()):
                    synonym = inflected
                    break

            if synonym:
                if word[0].isupper() and synonym[0].islower():
                    synonym = synonym[0].upper() + synonym[1:]
                trailing = ''
                for ch in reversed(word):
                    if not ch.isalpha():
                        trailing = ch + trailing
                    else:
                        break
                result_tokens[i] = synonym + trailing
                changed += 1
                changes.append((clean, synonym.lower()))

        reconstructed = _reconstruct_text(text, tokens, result_tokens)
        reconstructed = _fix_articles(reconstructed)
        return reconstructed, changes

    def inject_parentheticals(self, text: str, count: int = 2) -> str:
        """Insert parenthetical asides at existing comma positions to break
        token predictability without corrupting grammar."""
        asides = [
            "which is often overlooked",
            "though this varies by region",
            "as several studies suggest",
            "arguably the most critical factor",
            "a point often debated",
            "despite some disagreement",
            "though evidence varies",
            "which complicates things further",
            "not without controversy",
            "though not everyone agrees",
            "still poorly understood",
            "a growing concern",
        ]

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) < 4:
            return text

        # Track asides already in text to prevent duplicates across iterations
        text_lower = text.lower()
        available_asides = [a for a in asides if a.lower() not in text_lower]
        if not available_asides:
            return text

        added = 0
        result = []
        for i, sent in enumerate(sentences):
            if (added >= count
                    or i == 0 or i >= len(sentences) - 1
                    or len(sent.split()) < 15
                    or '—' in sent or '(' in sent):
                result.append(sent)
                continue

            # Only insert at an existing comma position (safe grammatical boundary)
            commas = [m.start() for m in re.finditer(r',', sent)]
            valid_commas = [p for p in commas if 15 < p < len(sent) - 20]

            if valid_commas and random.random() < 0.4 and available_asides:
                pos = random.choice(valid_commas)
                aside = available_asides.pop(random.randrange(len(available_asides)))
                style = random.choice(['dash', 'paren', 'comma'])
                if style == 'dash':
                    new_sent = sent[:pos] + f" — {aside} —" + sent[pos:]
                elif style == 'paren':
                    new_sent = sent[:pos] + f" ({aside})" + sent[pos:]
                else:
                    new_sent = sent[:pos] + f", {aside}," + sent[pos + 1:]
                result.append(new_sent)
                added += 1
            else:
                result.append(sent)

        return ' '.join(result)

    def inject_rhetorical_questions(self, text: str, count: int = 1) -> str:
        """Insert a standalone rhetorical question BEFORE a declarative sentence."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) < 5:
            return text

        # Pre-built rhetorical questions keyed by topic cue words
        topic_questions = [
            (r'\bthreat|damage|harm|destroy|disrupt', [
                "But why should we care?",
                "So what's really at stake here?",
            ]),
            (r'\bconserv|protect|preserv|sustain', [
                "But can we actually fix this?",
                "So what do we do about it?",
            ]),
            (r'\bresearch|study|science|understand', [
                "And how do we even begin to tackle this?",
                "So where does that leave us?",
            ]),
            (r'\bchalleng|difficult|problem|issue|crisis', [
                "But is it really that simple?",
                "So what's the catch?",
            ]),
        ]
        # Fallback generic questions
        generic_questions = [
            "But why does this matter?",
            "So what's the takeaway?",
            "And here's the thing.",
        ]

        inserted = 0
        result = []
        for i, sent in enumerate(sentences):
            # Only target middle sentences, skip first 2 and last 2
            if inserted < count and 2 <= i < len(sentences) - 2:
                words = sent.split()
                if len(words) >= 10 and '?' not in sent and random.random() < 0.3:
                    # Pick a topic-matched question or generic
                    question = None
                    for pattern, qs in topic_questions:
                        if re.search(pattern, sent, re.IGNORECASE):
                            question = random.choice(qs)
                            break
                    if question is None:
                        question = random.choice(generic_questions)
                    result.append(question)
                    inserted += 1
            result.append(sent)

        return ' '.join(result)

    def vary_sentence_rhythm(self, text: str) -> str:
        """Break uniform sentence length pattern — merge short sentences only."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) < 5:
            return text

        result = []
        merged_next = False
        for i, sent in enumerate(sentences):
            if merged_next:
                merged_next = False
                continue

            words = sent.split()

            # Merge two consecutive short sentences (< 10 words each)
            if (len(words) < 10
                    and i < len(sentences) - 1
                    and len(sentences[i + 1].split()) < 10
                    and random.random() < 0.35):
                next_sent = sentences[i + 1]
                connector = random.choice([' — ', '; '])
                combined = sent.rstrip('.') + connector + next_sent[0].lower() + next_sent[1:]
                result.append(combined)
                merged_next = True
                continue

            result.append(sent)

        return ' '.join(result)

    def inject_discourse_markers(self, text: str, count: int = 3) -> str:
        """Add informal discourse markers that increase perplexity."""
        starters = [
            "Granted,", "Admittedly,", "To be fair,", "In a way,",
            "Realistically,", "Interestingly enough,", "The thing is,",
            "Look,", "Honestly,", "That said,",
        ]
        mid_markers = [
            "in a sense", "if you think about it", "strictly speaking",
            "for better or worse", "at least in theory",
        ]

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) < 4:
            return text

        added = 0
        result = []
        # Track starters/markers already in text to prevent duplicates across iterations
        text_lower = text.lower()
        used_starters = {s for s in starters if s.lower().rstrip(',') in text_lower}
        used_mid = {m for m in mid_markers if m.lower() in text_lower}

        for i, sent in enumerate(sentences):
            if added >= count or i == 0 or i >= len(sentences) - 1:
                result.append(sent)
                continue

            # Skip if sentence already starts with a discourse marker or connector
            first_word = sent.split()[0].lower().rstrip(',') if sent.split() else ''
            if first_word in ('but', 'so', 'also', 'and', 'still', 'yet', 'granted',
                              'admittedly', 'honestly', 'look', 'realistically',
                              'to', 'in', 'the', 'interestingly', 'that'):
                result.append(sent)
                continue

            if random.random() < 0.35:
                if random.random() < 0.6:
                    # Sentence starter
                    available = [s for s in starters if s not in used_starters]
                    if available:
                        marker = random.choice(available)
                        used_starters.add(marker)
                        sent = f"{marker} {sent[0].lower()}{sent[1:]}"
                        added += 1
                else:
                    # Mid-sentence marker after first clause
                    available_mid = [m for m in mid_markers if m not in used_mid]
                    if not available_mid:
                        result.append(sent)
                        continue
                    comma_pos = sent.find(',')
                    if 8 < comma_pos < len(sent) - 15:
                        marker = random.choice(available_mid)
                        used_mid.add(marker)
                        sent = sent[:comma_pos + 1] + f" {marker}," + sent[comma_pos + 1:]
                        added += 1
                result.append(sent)
            else:
                result.append(sent)

        return ' '.join(result)

    # ── Classifier-bypass methods ─────────────────────────────────────────

    _CONTRACTION_MAP = [
        # Ordered longest-first to prevent partial matches
        ("would not", "wouldn't"), ("should not", "shouldn't"),
        ("could not", "couldn't"), ("does not", "doesn't"),
        ("did not", "didn't"), ("has not", "hasn't"),
        ("have not", "haven't"), ("had not", "hadn't"),
        ("will not", "won't"), ("do not", "don't"),
        ("is not", "isn't"), ("are not", "aren't"),
        ("was not", "wasn't"), ("were not", "weren't"),
        ("cannot", "can't"), ("can not", "can't"),
        ("they are", "they're"), ("they have", "they've"),
        ("there is", "there's"), ("that is", "that's"),
        ("it is", "it's"), ("we are", "we're"),
        ("we have", "we've"), ("who is", "who's"),
        ("what is", "what's"), ("let us", "let's"),
    ]

    def inject_contractions(self, text: str) -> str:
        """Replace formal constructions with contractions.
        Classifier-based detectors flag absence of contractions as AI signal.
        Applies to ~70% of matches; skips content inside quotes or parentheses."""
        result = text
        applied = 0

        for formal, contraction in self._CONTRACTION_MAP:
            # Case-insensitive search, but skip inside quotes or parenthetical asides
            pattern = re.compile(re.escape(formal), re.IGNORECASE)
            matches = list(pattern.finditer(result))
            if not matches:
                continue

            # Process matches in reverse order to preserve positions
            for m in reversed(matches):
                # Skip ~30% of matches to keep some formal constructions
                if random.random() < 0.30:
                    continue

                # Skip if inside quotes
                prefix = result[:m.start()]
                if prefix.count('"') % 2 == 1:
                    continue
                # Skip if inside parentheses or em-dash asides
                if prefix.count('(') > prefix.count(')'):
                    continue
                if prefix.count('\u2014') % 2 == 1:  # em-dash
                    continue

                orig = m.group(0)
                # Preserve case of first character
                if orig[0].isupper():
                    repl = contraction[0].upper() + contraction[1:]
                else:
                    repl = contraction
                result = result[:m.start()] + repl + result[m.end():]
                applied += 1

        return result

    def inject_burstiness(self, text: str) -> str:
        """Increase sentence length variance (burstiness) to fool classifier-based detectors.
        Human text has stdev > 8 words; AI text typically has stdev < 4.
        Strategy: create very short fragments (3-6 words) and long merged sentences."""
        import numpy as np

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) < 5:
            return text

        lengths = [len(s.split()) for s in sentences]
        current_stdev = float(np.std(lengths))
        if current_stdev >= 8:
            return text  # already bursty enough

        result = list(sentences)
        splits_done = 0

        # Pass 1: Split long sentences EARLY to create one short fragment (3-6 words)
        # and one long remainder — this maximizes variance
        new_result = []
        for i, sent in enumerate(result):
            words = sent.split()
            wlen = len(words)

            if splits_done < 3 and wlen >= 14:
                # Look for comma or conjunction in first 3-7 words
                split_pos = None
                for j in range(2, min(7, wlen - 6)):
                    if words[j].endswith(','):
                        split_pos = j + 1
                        break
                    if words[j].lower().rstrip(',') in ('and', 'but', 'which', 'because',
                                                         'although', 'while', 'since', 'where'):
                        split_pos = j
                        break

                if split_pos and 3 <= split_pos <= 7 and wlen - split_pos >= 6:
                    part1 = ' '.join(words[:split_pos]).rstrip(',')
                    part2 = ' '.join(words[split_pos:])

                    if not part1.endswith(('.', '!', '?')):
                        part1 = part1 + '.'
                    if part2 and part2[0].islower():
                        part2 = part2[0].upper() + part2[1:]

                    new_result.append(part1)
                    new_result.append(part2)
                    splits_done += 1
                    continue

            new_result.append(sent)

        result = new_result

        # Pass 2: Merge 2-3 consecutive medium sentences into one very long sentence
        merged = []
        merges_done = 0
        skip_next = False
        for i in range(len(result)):
            if skip_next:
                skip_next = False
                continue

            sent = result[i]
            words = sent.split()

            # Merge two medium sentences (8-18 words each) into one long one (16-36)
            if (merges_done < 2
                    and 8 <= len(words) <= 18
                    and i + 1 < len(result)
                    and 8 <= len(result[i + 1].split()) <= 18):
                next_sent = result[i + 1]
                connector = random.choice([' \u2014 ', '; ', ', and in fact '])
                combined = sent.rstrip('.!?') + connector + next_sent[0].lower() + next_sent[1:]
                merged.append(combined)
                merges_done += 1
                skip_next = True
            else:
                merged.append(sent)

        return ' '.join(merged)

    def inject_imperfections(self, text: str, count: int = 2) -> str:
        """Add subtle writing imperfections that classifiers associate with human writing.
        Not errors — stylistic choices that AI rarely makes."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) < 5:
            return text

        applied = 0
        result = list(sentences)

        # Pool of imperfection transforms
        transforms = []

        # Type 1: Hedging insertion — "shows that" → "seems to show that"
        hedges = [
            (r'\b(shows? that)\b', 'seems to show that'),
            (r'\b(proves? that)\b', 'seems to suggest that'),
            (r'\b(causes?)\b(?= [a-z])', 'is one of the causes of'),
            (r'\b(demonstrates? that)\b', 'appears to demonstrate that'),
        ]
        for i, sent in enumerate(result):
            if applied >= count:
                break
            for pattern, repl in hedges:
                if re.search(pattern, sent, re.IGNORECASE):
                    transforms.append(('hedge', i, pattern, repl))
                    break

        # Type 2: Intensifier addition — "important" → "really important"
        intensifiers = [
            (r'\b(important)\b', 'really important'),
            (r'\b(significant)\b', 'pretty significant'),
            (r'\b(complex)\b', 'quite complex'),
            (r'\b(critical)\b', 'absolutely critical'),
        ]
        for i, sent in enumerate(result):
            if len(transforms) >= count * 2:
                break
            for pattern, repl in intensifiers:
                if re.search(pattern, sent, re.IGNORECASE):
                    transforms.append(('intensify', i, pattern, repl))
                    break

        # Type 3: "So basically" or "In any case" before summary-like sentences
        summary_starters = ["So basically, ", "In any case, ", "At the end of the day, "]
        for i, sent in enumerate(result):
            if i > len(result) * 0.6 and len(transforms) < count * 3:
                first_word = sent.split()[0].lower() if sent.split() else ''
                if first_word in ('this', 'these', 'the', 'overall', 'in'):
                    transforms.append(('summary_prefix', i, None, random.choice(summary_starters)))
                    break

        # Shuffle and apply up to 'count' transforms
        random.shuffle(transforms)
        used_indices = set()

        for kind, idx, pattern, repl in transforms:
            if applied >= count or idx in used_indices:
                continue

            sent = result[idx]
            if kind == 'hedge' or kind == 'intensify':
                new_sent = re.sub(pattern, repl, sent, count=1, flags=re.IGNORECASE)
                if new_sent != sent:
                    result[idx] = new_sent
                    applied += 1
                    used_indices.add(idx)
            elif kind == 'summary_prefix':
                result[idx] = repl + sent[0].lower() + sent[1:]
                applied += 1
                used_indices.add(idx)

        return ' '.join(result)

    # ── Variant generation ────────────────────────────────────────────────
    def generate_swap_variants(self, text: str, original_text: str = None) -> List[Tuple[str, Dict, str, List[Tuple[str, str]]]]:
        """5 classifier-bypass variants combining contractions, burstiness,
        imperfections, AI-telltale swaps, rare synonyms, parentheticals,
        rhetorical questions, rhythm variation, and discourse markers.
        NO T5. BERT-filtered against original."""
        self.current_domain, confidence = self.detect_content_domain(text)
        ref = original_text if original_text else text
        raw = []

        # Phase A: apply AI-telltale swaps as a base (safe, curated)
        base, base_changes = self.safe_synonym_replace(
            text, max_ratio=0.20, domain=self.current_domain, original_text=ref)

        # Phase B: apply contractions + burstiness to base (classifier signals)
        base = self.inject_contractions(base)
        base = self.inject_burstiness(base)

        # (name, rare_ratio, parens, rhetorical, rhythm, discourse, imperfections)
        configs = [
            ("classifier-light",      0.10, 0, 0, False, 2, 0),
            ("classifier-moderate",   0.10, 1, 0, False, 0, 0),
            ("classifier-structural", 0.0,  0, 1, True,  0, 0),
            ("classifier-heavy",      0.15, 2, 0, False, 0, 2),
            ("classifier-kitchen",    0.10, 1, 1, True,  1, 1),
        ]

        for name, rare_ratio, parens, rhetorical, rhythm, discourse, imperf in configs:
            v = base
            changes = list(base_changes)

            # Rare synonym replacement (higher perplexity words)
            if rare_ratio > 0:
                v, rare_changes = self.rare_synonym_replace(
                    v, max_ratio=rare_ratio, original_text=ref)
                changes.extend(rare_changes)

            # Writing imperfections (hedging, intensifiers)
            if imperf > 0:
                v = self.inject_imperfections(v, count=imperf)

            # Structural perplexity injections
            if rhythm:
                v = self.vary_sentence_rhythm(v)
            if rhetorical > 0:
                v = self.inject_rhetorical_questions(v, count=rhetorical)
            if parens > 0:
                v = self.inject_parentheticals(v, count=parens)
            if discourse > 0:
                v = self.inject_discourse_markers(v, count=discourse)

            raw.append((v, name, changes))

        # BERT filter against original
        ref_text = original_text if original_text else text
        variants = []
        for var_text, strategy, changes in raw:
            filtered, reverted = self.bert_filter_sentences(ref_text, var_text)
            if reverted > 0:
                print(f"   🧠 BERT filter: reverted {reverted} sentence(s) in {strategy}")
            metrics = self.calculate_real_metrics(ref_text, filtered)
            variants.append((filtered, metrics, strategy, changes))

        return variants

    def generate_variants(self, text: str, original_text: str = None) -> List[Tuple[str, Dict, str, List[Tuple[str, str]]]]:
        """
        5 strategies using T5 sentence-level paraphrasing + post-processing.
        Sentence-BERT filtering applied for quality control.
        original_text: if provided, BERT filter compares against this (prevents cumulative drift).
        """
        self.current_domain, confidence = self.detect_content_domain(text)
        print(f"🔍 Detected domain: {self.current_domain} (confidence: {confidence:.2f})")

        raw_variants = []

        # Word-level replacement ratios per strategy (main detection reducer)
        synonym_ratios = {
            "light": 0.25, "balanced": 0.30, "structural": 0.25,
            "human-noise": 0.30, "kitchen-sink": 0.40,
        }

        for strategy in ["light", "balanced", "structural", "human-noise", "kitchen-sink"]:
            v, changes = self._t5.paraphrase(
                text, strategy=strategy,
                learning_engine=self.learning_engine
            )
            # Word-level synonym replacement on top of T5 paraphrasing
            syn_ratio = synonym_ratios.get(strategy, 0.20)
            ref = original_text if original_text else text
            v, word_changes = self.safe_synonym_replace(v, max_ratio=syn_ratio,
                                                         domain=self.current_domain,
                                                         original_text=ref)
            changes = changes + word_changes

            # Strategy-specific post-processing
            if strategy == "balanced":
                v = self.advanced_restructure(v, aggressiveness=0.3)
            elif strategy == "structural":
                v = self.advanced_restructure(v, aggressiveness=0.5)
                v = self.split_long_sentences(v)
                v = self.reorder_sentences(v)
            elif strategy == "human-noise":
                v = self.typo_inject(v, count=2)
                v = self.add_filler_phrases(v, count=2)
            elif strategy == "kitchen-sink":
                v = self.advanced_restructure(v, aggressiveness=0.4)
                v = self.split_long_sentences(v)
                v = self.add_filler_phrases(v, count=1)
                v = self.typo_inject(v, count=1)
            raw_variants.append((v, strategy, changes))

        # Apply Sentence-BERT filter to each variant (compare against original to prevent drift)
        ref_text = original_text if original_text else text
        variants = []
        for var_text, strategy, changes in raw_variants:
            filtered, reverted = self.bert_filter_sentences(ref_text, var_text)
            if reverted > 0:
                print(f"   🧠 BERT filter: reverted {reverted} sentence(s) in {strategy}")
            metrics = self.calculate_real_metrics(text, filtered)
            variants.append((filtered, metrics, strategy, changes))

        return variants
    
    def interactive_feedback(self, original: str, variants: List):
        """Kullanıcıdan geri bildirim al ve öğren"""
        if not self.learning_engine:
            return
        
        print("\n" + "="*60)
        print("LEARNING MODE: Please provide feedback to improve future transformations")
        print("="*60)
        
        for i, (text, metrics, name, changes) in enumerate(variants, 1):
            print(f"\nVariant {i} ({name}):")
            print(f"Changes made: {len(changes)} word substitutions")
            if changes:
                print(f"Examples: {', '.join([f'{o}->{n}' for o, n in changes[:3]])}")
            
            while True:
                feedback = input(f"Approve this variant? (y/n/skip): ").lower().strip()
                if feedback in ['y', 'n', 'skip']:
                    break
                print("Please enter 'y', 'n', or 'skip'")
            
            if feedback == 'y':
                self.learning_engine.record_user_feedback(
                    original, text, True, changes, self.current_domain
                )
                print("✅ Recorded as successful transformation")
            elif feedback == 'n':
                self.learning_engine.record_user_feedback(
                    original, text, False, changes, self.current_domain
                )
                print("❌ Recorded as unsuccessful transformation")
            else:
                print("⏭️ Skipped")
    
    def create_report(self, original: str, variants: List, filename: str) -> str:
        """ŞEFFAF ve GERÇEKÇİ rapor - öğrenme istatistikleri ile"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("analysis_outputs", exist_ok=True)
        path = f"analysis_outputs/QUALITY_REPORT_{ts}.txt"
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TEXT QUALITY ANALYZER v53.0 - SELF-LEARNING EDITION\n")
            f.write("=" * 80 + "\n")
            f.write(f"Source: {filename}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Domain Detected: {self.current_domain}\n")
            f.write(f"Learning Enabled: {self.enable_learning}\n")
            if self.learning_engine:
                stats = self.learning_engine.get_learning_report()
                f.write(f"Total Past Interactions: {stats['total_interactions']}\n")
                f.write(f"Learning Success Rate: {stats['success_rate']:.1%}\n")
            f.write("⚠️  IMPORTANT: These are REAL calculated metrics\n")
            f.write("⚠️  NO 'evasion probability' - such metrics are heuristic at best\n")
            f.write("⚠️  Modern detection uses semantic analysis, not just string matching\n")
            f.write("=" * 80 + "\n\n")
            
            # Öğrenme istatistikleri
            if self.learning_engine and self.learning_engine.learning_stats['total_interactions'] > 0:
                f.write("LEARNING STATISTICS\n")
                f.write("-" * 80 + "\n")
                stats = self.learning_engine.get_learning_report()
                f.write(f"Domains Learned: {', '.join(stats['domains_learned'])}\n")
                f.write(f"Total Interactions: {stats['total_interactions']}\n")
                f.write(f"Overall Success Rate: {stats['success_rate']:.1%}\n")
                
                if stats['top_successful_synonyms']:
                    f.write("\nTop Successful Synonym Pairs (learned):\n")
                    for pair, rate in stats['top_successful_synonyms'][:5]:
                        f.write(f"  {pair}: {rate:.1%} success\n")
                f.write("-" * 80 + "\n\n")
            
            f.write("REALISTIC EXPECTATIONS:\n")
            f.write("-" * 80 + "\n")
            f.write("This tool demonstrates:\n")
            f.write("  • Lexical substitution effects on text similarity\n")
            f.write("  • Structural variation impact on n-gram matching\n")
            f.write("  • Readability and meaning preservation trade-offs\n")
            f.write("  • Self-learning adaptation to domains and user preferences\n\n")
            f.write("It does NOT guarantee:\n")
            f.write("  • Detection evasion (modern systems are sophisticated)\n")
            f.write("  • Academic integrity (only original work ensures this)\n")
            f.write("  • Semantic equivalence (automated paraphrasing has limits)\n")
            f.write("-" * 80 + "\n\n")
            
            # Orijinal
            f.write("1. ORIGINAL TEXT\n")
            f.write("=" * 80 + "\n")
            f.write(original)
            f.write("\n\n")
            
            # Varyantlar
            for i, (text, metrics, name, changes) in enumerate(variants, 2):
                f.write(f"{i}. VARIANT: {name.upper()}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("TRANSFORMATION DETAILS:\n")
                f.write(f"  Word Substitutions: {len(changes)}\n")
                if changes:
                    f.write(f"  Changes: {', '.join([f'{o}->{n}' for o, n in changes[:10]])}")
                    if len(changes) > 10:
                        f.write(f" ... and {len(changes)-10} more")
                    f.write("\n")
                f.write(f"  Domain Context: {metrics['domain']}\n\n")
                
                f.write("METRICS (Calculated, not estimated):\n")
                f.write(f"  Word Change Ratio:      {metrics['word_change_ratio']:.1%}\n")
                f.write(f"  4-gram Similarity:      {metrics['ngram_similarity']:.1%}\n")
                f.write(f"  Readability Score:      {metrics['readability_score']:.2f}\n")
                f.write(f"  Meaning Preservation:   {metrics['meaning_preservation']:.1%}\n")
                f.write(f"  Length Ratio:           {metrics['length_ratio']:.2f}x\n")
                f.write(f"  Assessment:             {metrics['assessment']}\n")
                f.write(f"  Word Count:             {metrics['original_word_count']} → {metrics['transformed_word_count']}\n\n")
                
                f.write("CONTENT:\n")
                f.write(text)
                f.write("\n\n")
            
            # Karşılaştırma
            f.write(f"{len(variants)+2}. COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"{'Strategy':<15} {'Word Change':<12} {'N-gram Sim':<12} {'Readability':<12} {'Meaning':<12} {'Status'}\n")
            f.write("-" * 80 + "\n")
            
            for _, m, name, _ in variants:
                status = "OK" if "BALANCED" in m['assessment'] else "ISSUE"
                f.write(f"{name:<15} {m['word_change_ratio']:<12.1%} {m['ngram_similarity']:<12.1%} "
                       f"{m['readability_score']:<12.2f} {m['meaning_preservation']:<12.1%} {status}\n")
            
            # Öğrenme önerileri
            if self.enable_learning:
                f.write("\n" + "=" * 80 + "\n")
                f.write("LEARNING RECOMMENDATIONS\n")
                f.write("-" * 80 + "\n")
                f.write("To improve future transformations:\n")
                f.write("1. Run with --feedback flag to provide approval/disapproval\n")
                f.write("2. The system learns which synonyms work best in which domains\n")
                f.write("3. Domain-specific terminology is preserved and learned\n")
                f.write("4. Check learning_data/ directory for accumulated knowledge\n")
                f.write("-" * 80 + "\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("Remember: Original intellectual work is the only true academic integrity.\n")
            f.write("=" * 80 + "\n")
        
        return path
    
    def _load_static_synonyms(self) -> Dict[str, List[str]]:
        """Statik eş anlamlı sözlük (fallback)"""
        if self.learning_engine:
            return self.learning_engine._get_base_synonyms()
        return {}


def main():
    print("TEXT QUALITY ANALYZER v53.0 - SELF-LEARNING EDITION")
    print("Real Metrics | Real Analysis | Real Transparency | Real Adaptation")
    print("=" * 70)
    
    # Argüman parsing
    feedback_mode = '--feedback' in sys.argv
    no_learning = '--no-learn' in sys.argv
    
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    target = args[0] if args else input("📄 File (.txt): ").strip()
    
    if not target.endswith('.txt') or not os.path.exists(target):
        print("❌ Error: Provide valid .txt file")
        return
    
    with open(target, 'r', encoding='utf-8') as f:
        original = f.read()
    
    print(f"Analyzing: {len(original)} characters")
    print(f"Learning Mode: {'OFF' if no_learning else 'ON'}")
    print(f"Feedback Mode: {'ON' if feedback_mode else 'OFF'}")
    
    analyzer = TextQualityAnalyzer(enable_learning=not no_learning)
    variants = analyzer.generate_variants(original)
    
    # En dengeli olanı seç
    best = min(variants, key=lambda x: abs(x[1]['word_change_ratio'] - 0.25))
    
    report = analyzer.create_report(original, variants, target)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    for _, m, name, changes in variants:
        print(f"{name:12} | Change: {m['word_change_ratio']:>5.1%} | N-gram: {m['ngram_similarity']:>5.1%} | Changes: {len(changes):>3} | {m['assessment'][:30]}")
    print(f"{'='*70}")
    print(f"📄 Report: {report}")
    print(f"🏆 Most balanced: {best[2]} ({best[1]['word_change_ratio']:.1%} change)")
    
    # İnteraktif geri bildirim
    if feedback_mode:
        analyzer.interactive_feedback(original, variants)
        
        # Öğrenme raporunu göster
        if analyzer.learning_engine:
            print("\n" + "="*70)
            print("UPDATED LEARNING STATISTICS")
            print("="*70)
            stats = analyzer.learning_engine.get_learning_report()
            print(f"Total Interactions: {stats['total_interactions']}")
            print(f"Success Rate: {stats['success_rate']:.1%}")
            print(f"Domains: {', '.join(stats['domains_learned'])}")


if __name__ == "__main__":
    main()