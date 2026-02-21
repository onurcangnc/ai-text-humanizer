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
for _pkg in ('averaged_perceptron_tagger_eng', 'punkt_tab'):
    try:
        nltk.data.find(f'tokenizers/{_pkg}' if _pkg == 'punkt_tab' else f'taggers/{_pkg}')
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


class BERTSynonymSwapper:
    """BERT Masked Language Model synonym replacement.
    Lazy-loads bert-base-uncased (~440MB) on first use."""

    def __init__(self, model_name: str = "bert-base-uncased", top_k: int = 10):
        self._model_name = model_name
        self._top_k = top_k
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load the fill-mask pipeline."""
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "fill-mask", model=self._model_name, top_k=self._top_k
            )
            print(f"   🤖 BERT MLM loaded ({self._model_name})")
        return self._pipeline

    def get_replacements(self, context: str, target_word: str,
                         target_pos: str) -> List[str]:
        """Mask target_word in context, return filtered BERT predictions."""
        pipe = self._load_pipeline()
        try:
            results = pipe(context)
        except Exception:
            return []

        candidates = []
        for r in results:
            token = r["token_str"].strip()
            if (token.startswith("##")
                    or token.lower() == target_word.lower()
                    or token.lower() in _STOPWORDS
                    or token.lower() in _AMBIGUOUS_WORDS
                    or len(token) < 4
                    or not token.isalpha()
                    or token.lower() in _spell.unknown([token.lower()])
                    or not _pos_compatible(target_pos, token)):
                continue
            candidates.append(token)
        return candidates

    @staticmethod
    def _restore_capitalization(original: str, replacement: str) -> str:
        """Transfer capitalization pattern from original to replacement."""
        if original.isupper():
            return replacement.upper()
        if original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement.lower()


class ParrotParaphraser:
    """Sentence-level paraphrasing using Parrot (T5-based).
    Lazy-loads prithivida/parrot_paraphraser_on_T5 on first use."""

    STRATEGY_RATIOS = {
        "light": 0.3,
        "balanced": 0.5,
        "structural": 0.4,
        "human-noise": 0.35,
        "kitchen-sink": 0.6,
    }

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            from parrot import Parrot
            self._model = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)
            print("   🦜 Parrot paraphraser loaded")
        return self._model

    def paraphrase(self, text: str, strategy: str = "light") -> Tuple[str, List[Tuple[str, str]]]:
        """Paraphrase text at the sentence level.
        Returns (paraphrased_text, list_of_(original_sent, new_sent) pairs)."""
        model = self._load()
        ratio = self.STRATEGY_RATIOS.get(strategy, 0.4)

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        result_sents = []
        changes = []

        for sent in sentences:
            words = sent.split()
            # Skip very short sentences or apply probabilistically
            if len(words) < 5 or random.random() > ratio:
                result_sents.append(sent)
                continue

            try:
                # max_length must exceed input token count; estimate ~1.5 tokens/word
                max_len = max(64, int(len(words) * 1.5))
                paraphrases = model.augment(
                    input_phrase=sent,
                    max_length=max_len,
                    adequacy_threshold=0.55,
                    fluency_threshold=0.55,
                    do_diverse=False,
                    max_return_phrases=10,
                )
                if paraphrases:
                    # paraphrases is a list of (phrase, score) tuples
                    best = paraphrases[0][0]
                    # Ensure it ends with proper punctuation
                    if not best.endswith(('.', '!', '?')):
                        best = best.rstrip() + '.'
                    # Capitalize first letter
                    if best and best[0].islower():
                        best = best[0].upper() + best[1:]
                    result_sents.append(best)
                    changes.append((sent.strip(), best.strip()))
                else:
                    result_sents.append(sent)
            except Exception:
                result_sents.append(sent)

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
        
        # Thompson Sampling: update alpha/beta for each synonym pair
        for orig_word, new_word in changes_made:
            key = f"{orig_word}->{new_word}"
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
        self._bert_swapper = BERTSynonymSwapper()  # BERT MLM for synonym swap
        self._use_parrot = os.getenv("USE_PARROT", "").lower() in ("true", "1", "yes")
        self._parrot = ParrotParaphraser() if self._use_parrot else None

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
                              threshold: float = 0.82) -> Tuple[str, int]:
        """
        Sentence-level BERT filter: revert any sentence where semantic similarity
        drops below threshold. Returns (filtered_text, num_reverted).
        Uses batch encoding for efficiency (~50ms for 7 sentences).
        """
        model = self._load_sbert()
        if model is None:
            return modified, 0

        orig_sents = re.split(r'(?<=[.!?])\s+', original.strip())
        mod_sents = re.split(r'(?<=[.!?])\s+', modified.strip())

        # If sentence count changed (splits/merges), check overall similarity only
        if len(orig_sents) != len(mod_sents):
            embeddings = model.encode([original, modified])
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            sim = cos_sim([embeddings[0]], [embeddings[1]])[0][0]
            if sim >= threshold:
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

    def _build_context_sentence(self, tokens: List[str], target_index: int,
                                window: int = 40) -> str:
        """Build windowed context with [MASK] at target position for BERT."""
        start = max(0, target_index - window)
        end = min(len(tokens), target_index + window)
        ctx = list(tokens[start:end])
        ctx[target_index - start] = "[MASK]"
        return ' '.join(ctx)

    def intelligent_synonym_replace(self, text: str, max_ratio: float = 0.3,
                                   domain: str = 'general') -> Tuple[str, List[Tuple[str, str]]]:
        """
        BERT MLM-powered synonym replacement.
        Priority: learned synonyms (Thompson) → BERT MLM → static dictionary.
        max_ratio: maximum fraction of words to change
        Returns: (transformed_text, list_of_changes)
        """
        # Static synonyms as last-resort fallback
        if self.learning_engine:
            static_syns = self.learning_engine.get_domain_synonyms(domain)
        else:
            static_syns = self._load_static_synonyms()

        # POS-tag the entire text for context-aware replacement
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)

        max_changes = int(len(tokens) * max_ratio)
        result_tokens = list(tokens)

        # Identify eligible word positions
        eligible = []
        for i, (word, penn_tag) in enumerate(tagged):
            clean = re.sub(r'[^\w]', '', word.lower())
            if (len(clean) < 4
                    or clean in _STOPWORDS
                    or clean in _AMBIGUOUS_WORDS
                    or penn_tag in ('NNP', 'NNPS')
                    or not clean.isalpha()):
                continue
            eligible.append(i)

        # Probabilistic selection: shuffle and take up to 2x max_changes
        random.shuffle(eligible)
        candidates_to_try = eligible[:max_changes * 2]

        changed = 0
        changes_made = []

        for i in candidates_to_try:
            if changed >= max_changes:
                break

            word, penn_tag = tagged[i]
            clean = re.sub(r'[^\w]', '', word.lower())
            upos = _penn_to_upos(penn_tag)
            synonym = None

            # ── Try 1: Learned synonyms (Thompson sampling) ───────────
            if self.learning_engine and upos:
                lemma_result = getLemma(word, upos=upos)
                lemma = lemma_result[0] if lemma_result else clean
                learned = self.learning_engine.get_intelligent_replacement_candidates(lemma, domain)
                if learned:
                    candidate = self.learning_engine.thompson_sample(lemma, learned)
                    inflected = getInflection(candidate, tag=penn_tag)
                    if inflected and inflected[0].lower() != word.lower():
                        if inflected[0].lower() not in _spell.unknown([inflected[0].lower()]):
                            synonym = inflected[0]

            # ── Try 2: BERT MLM prediction ────────────────────────────
            if not synonym:
                context = self._build_context_sentence(result_tokens, i)
                bert_candidates = self._bert_swapper.get_replacements(
                    context, clean, penn_tag
                )
                if bert_candidates:
                    if self.learning_engine:
                        lemma_result = getLemma(word, upos=upos) if upos else (clean,)
                        lemma = lemma_result[0] if lemma_result else clean
                        synonym = self.learning_engine.thompson_sample(lemma, bert_candidates)
                    else:
                        synonym = bert_candidates[0]

            # ── Try 3: Static dictionary fallback ─────────────────────
            if not synonym and clean in static_syns:
                fallback = static_syns[clean]
                if fallback:
                    synonym = random.choice(fallback)

            # ── Spell-check validation ────────────────────────────────
            if synonym and synonym.lower() in _spell.unknown([synonym.lower()]):
                synonym = None

            # ── Apply replacement ─────────────────────────────────────
            if synonym:
                synonym = BERTSynonymSwapper._restore_capitalization(word, synonym)

                # Preserve trailing punctuation
                trailing = ''
                for ch in reversed(word):
                    if not ch.isalpha():
                        trailing = ch + trailing
                    else:
                        break

                result_tokens[i] = synonym + trailing
                changed += 1
                changes_made.append((clean, synonym.lower()))

        # Reconstruct text preserving original whitespace pattern
        reconstructed = _reconstruct_text(text, tokens, result_tokens)
        # Post-processing: fix a/an article mismatches
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

            if random.random() < 0.3:
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
    
    def generate_variants(self, text: str) -> List[Tuple[str, Dict, str, List[Tuple[str, str]]]]:
        """
        5 strateji - hepsi dengeli, öğrenme ile optimize edilmiş
        BERT filtering applied to each variant for quality control.
        """
        # Domain tespiti
        self.current_domain, confidence = self.detect_content_domain(text)
        print(f"🔍 Detected domain: {self.current_domain} (confidence: {confidence:.2f})")

        raw_variants = []

        if self._use_parrot and self._parrot:
            print("   🦜 Using Parrot paraphraser engine")
            for strategy in ["light", "balanced", "structural", "human-noise", "kitchen-sink"]:
                v, changes = self._parrot.paraphrase(text, strategy=strategy)
                # Apply additional transforms per strategy
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
        else:
            # Default: BERT MLM synonym engine
            # Strateji 1: Hafif (sadece eş anlamlı, öğrenilmiş)
            v1, changes1 = self.intelligent_synonym_replace(text, max_ratio=0.2, domain=self.current_domain)
            raw_variants.append((v1, "light", changes1))

            # Strateji 2: Orta (eş anlamlı + yapısal)
            v2, changes2 = self.intelligent_synonym_replace(text, max_ratio=0.35, domain=self.current_domain)
            v2 = self.advanced_restructure(v2, aggressiveness=0.3)
            raw_variants.append((v2, "balanced", changes2))

            # Strateji 3: Yapısal odaklı (daha az eş anlamlı, daha çok yapı)
            v3, changes3 = self.intelligent_synonym_replace(text, max_ratio=0.15, domain=self.current_domain)
            v3 = self.advanced_restructure(v3, aggressiveness=0.5)
            v3 = self.split_long_sentences(v3)
            v3 = self.reorder_sentences(v3)
            raw_variants.append((v3, "structural", changes3))

            # Strateji 4: Human-noise (typo + filler + synonym)
            v4, changes4 = self.intelligent_synonym_replace(text, max_ratio=0.20, domain=self.current_domain)
            v4 = self.typo_inject(v4, count=2)
            v4 = self.add_filler_phrases(v4, count=2)
            raw_variants.append((v4, "human-noise", changes4))

            # Strateji 5: Kitchen-sink (all techniques combined, aggressive)
            v5, changes5 = self.intelligent_synonym_replace(text, max_ratio=0.30, domain=self.current_domain)
            v5 = self.advanced_restructure(v5, aggressiveness=0.4)
            v5 = self.split_long_sentences(v5)
            v5 = self.add_filler_phrases(v5, count=1)
            v5 = self.typo_inject(v5, count=1)
            raw_variants.append((v5, "kitchen-sink", changes5))

        # Apply BERT sentence-level filter to each variant
        variants = []
        for var_text, strategy, changes in raw_variants:
            filtered, reverted = self.bert_filter_sentences(text, var_text)
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