from dataclasses import dataclass
import re
from typing import Literal, Set

_NLP = None
_NLP_ATTEMPTED = False
ExtractorMode = Literal["spacy", "lexical"]


# Domain-specific patterns for code context
CODE_PATTERNS = [
    r"\b\w+\(\)",
    r"\b\w+\.\w+\(\)",
    r"\b[A-Z][a-zA-Z]+Error\b",
    r"\bv\d+\.\d+[\.\d]*\b",
    r"\b\d+[KMGkm]?[Bb]\b",
    r"[a-z_]+/[a-z_./]+\.py",
    r"line\s+\d+",
    r"\bRSS\b|\bRAM\b|\bCPU\b|\bOOM\b",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "if", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "then", "there", "these", "this", "to", "was", "were", "when", "with",
}


@dataclass
class ExtractionResult:
    named_entities: Set[str]
    code_entities: Set[str]
    noun_phrases: Set[str]
    all_keywords: Set[str]


def _get_nlp():
    global _NLP, _NLP_ATTEMPTED
    if _NLP_ATTEMPTED:
        return _NLP

    _NLP_ATTEMPTED = True
    try:
        import spacy

        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            # Allow the package to run without the optional spaCy model.
            _NLP = spacy.blank("en")
    except Exception:
        # spaCy itself may be unavailable or incompatible with the local Python.
        _NLP = None
    return _NLP


def extract(text: str, mode: ExtractorMode = "spacy") -> ExtractionResult:
    if mode not in {"spacy", "lexical"}:
        raise ValueError("mode must be 'spacy' or 'lexical'")

    nlp = _get_nlp() if mode == "spacy" else None
    named_entities = set()
    noun_phrases = set()

    if nlp is not None:
        doc = nlp(text)
        named_entities = {
            ent.text.lower() for ent in doc.ents
            if ent.label_ in ("PRODUCT", "ORG", "GPE", "PERSON", "EVENT", "WORK_OF_ART")
        }

        noun_phrases = {
            chunk.text.lower() for chunk in doc.noun_chunks
            if len(chunk.text.split()) >= 2
        } if doc.has_annotation("DEP") else set()

    code_entities = set()
    for pattern in CODE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        code_entities.update(m.lower() for m in matches)

    lexical_phrases = _lexical_phrases(text)
    if mode == "lexical":
        noun_phrases = lexical_phrases
    elif not named_entities and not noun_phrases:
        noun_phrases = lexical_phrases

    all_keywords = named_entities | code_entities | noun_phrases

    return ExtractionResult(
        named_entities=named_entities,
        code_entities=code_entities,
        noun_phrases=noun_phrases,
        all_keywords=all_keywords,
    )


def overlap_score(keywords_a: Set[str], keywords_b: Set[str]) -> float:
    """
    Returns a score 0.0-1.0 for keyword overlap between two sets.
    Weighted: code entities count more than noun phrases.
    """
    if not keywords_a or not keywords_b:
        return 0.0

    def is_code_entity(k: str) -> bool:
        for pattern in CODE_PATTERNS:
            if re.search(pattern, k, re.IGNORECASE):
                return True
        return False

    def normalize(k: str) -> str:
        # strip determiners
        k = re.sub(r'^(a|an|the|this|that)\s+', '', k)
        # strip parens
        k = k.replace("()", "")
        return k.strip()

    # Map normalized keywords to their computed weights
    weights_a = {normalize(k): 2.0 if is_code_entity(k) else 1.0 for k in keywords_a}
    weights_b = {normalize(k): 2.0 if is_code_entity(k) else 1.0 for k in keywords_b}

    norm_a = set(weights_a.keys())
    norm_b = set(weights_b.keys())

    intersection = norm_a & norm_b
    if not intersection:
        return 0.0

    match_weight = sum((weights_a[k] + weights_b[k]) / 2.0 for k in intersection)
    total_min_weight = min(sum(weights_a.values()), sum(weights_b.values()))

    # Scale result to maximum of 1.0
    return min(match_weight / total_min_weight, 1.0)


def _lexical_phrases(text: str) -> Set[str]:
    tokens = [
        token
        for token in re.findall(r"[a-zA-Z0-9_./:-]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]
    phrases = set(tokens)
    for size in (2, 3):
        for index in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[index:index + size])
            if phrase:
                phrases.add(phrase)
    return phrases
