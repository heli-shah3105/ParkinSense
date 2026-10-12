"""
=============================================================================
Dysarthric Lexicon & Language Decoder (Pure Python — Zero C++ Dependency)
=============================================================================
Provides clinical vocabulary-constrained decoding, phonetic error correction,
and language modeling for dysarthric speech recognition.

People with dysarthria frequently exhibit:
  - Incomplete consonant articulation (e.g. "cmpter" -> "computer", "bacspas" -> "backspace")
  - Vowel reduction / centralizations (e.g. "papir" -> "paper")
  - Fricative-to-stop substitutions (e.g. "dat" -> "that", "wit" -> "with")
  - Final consonant deletion (e.g. "escap" -> "escape")

This module uses a clinical lexicon (UASpeech + TORGO) plus high-frequency English
vocabulary to snap distorted CTC acoustic outputs to legitimate words with zero
need for external C++ compilers (like KenLM or pyctcdecode).
"""

import os
import re
from typing import Dict, List, Optional, Set, Tuple

from config import UASPEECH_WORDS, MANIFEST_DIR


def damerau_levenshtein(s1: str, s2: str) -> int:
    """Calculate the Damerau-Levenshtein distance between two strings."""
    d = {}
    len1 = len(s1)
    len2 = len(s2)
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,        # deletion
                d[(i, j - 1)] + 1,        # insertion
                d[(i - 1, j - 1)] + cost  # substitution
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + 1)  # transposition

    return d[(len1 - 1, len2 - 1)]


# Standard high-frequency vocabulary for daily & clinical interaction
CORE_ENGLISH_WORDS = {
    # Pronouns & determiners
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "a", "an", "the", "some", "any", "no", "every", "all", "each", "both", "few",
    # Verbs
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "go", "goes", "went", "gone", "come", "came", "see", "saw",
    "say", "said", "get", "got", "make", "made", "know", "knew", "think", "thought",
    "take", "took", "want", "wanted", "need", "needed", "help", "helped", "feel",
    "felt", "hear", "heard", "look", "looked", "stop", "stopped", "start", "started",
    "open", "opened", "close", "closed", "speak", "spoke", "talk", "talked", "call",
    "called", "eat", "ate", "drink", "drank", "sleep", "slept", "sit", "sat", "stand",
    "stood", "walk", "walked", "move", "moved", "turn", "turned", "show", "showed",
    "tell", "told", "ask", "asked", "try", "tried", "leave", "left", "keep", "kept",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    # Prepositions & conjunctions
    "in", "on", "at", "to", "for", "with", "from", "by", "about", "into", "through",
    "after", "before", "between", "under", "over", "above", "below", "up", "down",
    "out", "off", "of", "and", "but", "or", "so", "if", "because", "as", "than",
    "when", "where", "why", "how", "while", "though", "although", "since",
    # Everyday / Clinical / Needs
    "yes", "no", "please", "thanks", "thank", "hello", "hi", "goodbye", "bye",
    "water", "food", "medicine", "doctor", "nurse", "hospital", "bed", "chair",
    "pain", "hurts", "cold", "hot", "tired", "sleepy", "hungry", "thirsty",
    "light", "door", "window", "room", "phone", "glasses", "teeth", "head",
    "hand", "arm", "leg", "foot", "back", "chest", "stomach", "throat",
    "today", "tomorrow", "yesterday", "now", "later", "morning", "afternoon", "night",
    "well", "bad", "okay", "fine", "ready", "ready", "much", "more", "less",
    "very", "too", "quite", "really", "almost", "always", "never", "sometimes",
    # Grandfather passage (common clinical speech assessment text)
    "grandfather", "well", "old", "nearly", "ninety-three", "ninety", "three",
    "years", "yet", "still", "thinks", "swiftly", "young", "man", "long", "flowing",
    "beard", "clings", "chin", "gives", "face", "picturesque", "aspect", "when",
    "speaks", "voice", "quivers", "bit", "twice", "day", "plays", "violin",
    "little", "tune", "except", "rare", "occasions", "asks", "coat", "several",
    "tattered", "patches", "frock", "coat", "grand", "style",
}

# Multi-word standard computer / assessment phrases
MULTI_WORD_PHRASES = [
    "page up", "page down", "caps lock", "num lock", "scroll lock",
    "print screen", "frock coat", "ninety three", "young man",
    "story starters", "thank you", "good morning", "good night",
]


class DysarthricLexiconDecoder:
    """
    Intelligent language and lexicon rescorer tailored for dysarthric speech.
    Provides fast phoneme error correction and vocabulary matching.
    """

    def __init__(self):
        self.vocabulary: Set[str] = set()
        self.word_frequencies: Dict[str, float] = {}
        self.multi_word_phrases: List[str] = MULTI_WORD_PHRASES.copy()

        self._build_lexicon()

    def _build_lexicon(self):
        """Assemble all clinical UASpeech, TORGO, and common English words."""
        # 1. Add all UASpeech words from config
        for word in UASPEECH_WORDS.values():
            w = word.strip().lower()
            if " " in w:
                self.multi_word_phrases.append(w)
                for part in w.split():
                    self._add_word(part, weight=10.0)
            else:
                self._add_word(w, weight=10.0)

        # 2. Add Core English Words
        for w in CORE_ENGLISH_WORDS:
            self._add_word(w, weight=5.0)

        # 3. Harvest from split_manifest.csv if available
        manifest_file = os.path.join(MANIFEST_DIR, "split_manifest.csv")
        if os.path.exists(manifest_file):
            try:
                import pandas as pd
                df = pd.read_csv(manifest_file)
                for transcript in df["transcript"].dropna():
                    clean = re.sub(r"[^a-zA-Z\s']", "", str(transcript).lower()).strip()
                    for tok in clean.split():
                        if len(tok) >= 1:
                            self._add_word(tok, weight=8.0)
            except Exception:
                pass

    def _add_word(self, word: str, weight: float = 1.0):
        w = word.strip().lower()
        if w:
            self.vocabulary.add(w)
            self.word_frequencies[w] = self.word_frequencies.get(w, 0.0) + weight

    def correct_word(self, raw_token: str, min_confidence: float = 1.0) -> str:
        """
        Correct a single token using vocabulary prior & edit distance.
        If the token is already in vocabulary, return it as-is.
        Otherwise, search for the closest match in the lexicon.
        """
        token = raw_token.strip().lower()
        if not token:
            return token

        # Already exact match
        if token in self.vocabulary:
            return token

        # Very short tokens (1 char) should match single letters or common words
        if len(token) == 1:
            return token if token in self.vocabulary else token

        # Determine maximum acceptable edit distance based on token length
        max_dist = 1 if len(token) <= 4 else (2 if len(token) <= 8 else 3)

        best_candidate = token
        best_score = float("inf")

        for candidate in self.vocabulary:
            # Length pruning
            if abs(len(candidate) - len(token)) > max_dist:
                continue

            # First letter match gives huge bonus for dysarthric word recognition
            first_match = (candidate[0] == token[0])
            dist = damerau_levenshtein(token, candidate)

            if dist <= max_dist:
                # Frequency weighting (higher frequency gives lower penalty)
                freq_weight = 1.0 / (1.0 + 0.1 * self.word_frequencies.get(candidate, 1.0))
                # First letter bonus
                letter_penalty = 0.0 if first_match else 0.4
                # Dysarthric vowel omission check: e.g. token 'cmpter' vs candidate 'computer'
                vowel_omit_bonus = 0.0
                if set(token).issubset(set(candidate)):
                    vowel_omit_bonus = -0.3

                score = dist + letter_penalty + vowel_omit_bonus + freq_weight

                if score < best_score:
                    best_score = score
                    best_candidate = candidate

        return best_candidate

    def decode_text(self, text: str) -> str:
        """
        Rescore and correct an entire transcription string.
        Resolves multi-word phrases and individual word distortions.
        """
        if not text:
            return ""

        clean_text = text.strip().lower()

        # 1. Multi-word phrase matching
        for phrase in sorted(self.multi_word_phrases, key=len, reverse=True):
            # Check for close matches to multi-word phrases (e.g. "page up")
            pattern = re.escape(phrase)
            if re.search(r"\b" + pattern + r"\b", clean_text):
                continue  # already exact

            # Fuzzy phrase check
            words_in_phrase = phrase.split()
            tokens = clean_text.split()
            if len(tokens) == len(words_in_phrase):
                dist = sum(damerau_levenshtein(t, w) for t, w in zip(tokens, words_in_phrase))
                if dist <= 2:
                    return phrase

        # 2. Token-by-token correction
        tokens = clean_text.split()
        corrected_tokens = [self.correct_word(tok) for tok in tokens]

        # 3. Post-pass for consecutive tokens matching a multi-word phrase
        merged_str = " ".join(corrected_tokens)
        for phrase in self.multi_word_phrases:
            phrase_nospace = phrase.replace(" ", "")
            if merged_str.replace(" ", "") == phrase_nospace:
                return phrase

        return " ".join(corrected_tokens)

    def decode_word_timestamps(self, words: List[Dict]) -> List[Dict]:
        """
        Apply lexicon correction to word timestamp dictionaries.
        Preserves word start_time and end_time while correcting word string.
        """
        if not words:
            return []

        corrected_words = []
        for item in words:
            raw_w = item.get("word", "")
            corr_w = self.correct_word(raw_w)
            boost = 1.05 if corr_w != raw_w else 1.0
            corrected_words.append({
                "word": corr_w,
                "start_time": item.get("start_time", 0.0),
                "end_time": item.get("end_time", 0.0),
                "confidence": min(1.0, round(item.get("confidence", 0.5) * boost, 3)),
            })
        return corrected_words
