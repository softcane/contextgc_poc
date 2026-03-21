from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import ceil
from random import Random
from statistics import mean, pstdev
from typing import Any, Callable, Optional
import re

from contextgc_barrier.backend import ContextBackend


SessionTurn = dict[str, str]
TaskBuilder = Callable[[ContextBackend, str, int, float, int], "TaskInstance"]
BackendFactory = Callable[[], ContextBackend]


DATE_PATTERN = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
LONG_DATE_PATTERN = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE)
ROUTE_PATTERN = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\s+/[a-z0-9/_-]+\b", re.IGNORECASE)
IDENTIFIER_PATTERN = re.compile(r"\b(?:[A-Z]+-[A-Z0-9]+|[A-Z]-\d+|SKU-[A-Z]+-\d+)\b")
VERSION_PATTERN = re.compile(r"\bv\d+\.\d+(?:\.\d+)*\b", re.IGNORECASE)
PATH_LINE_PATTERN = re.compile(r"[a-z0-9_./-]+\.py(?:[`'\"]?\s*(?::\s*|\s+line\s+)\d+)?", re.IGNORECASE)
SYMBOL_PATTERN = re.compile(r"\b[a-z_][a-z0-9_]*\(\)", re.IGNORECASE)
QUANTITY_PATTERN = re.compile(
    r"\b\d+\s+(?:days?|business days?|minutes?|hours?)\b|\bnext-day courier\b|\btwo-day expedited\b|\bpriority overnight\b",
    re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"[a-z0-9_./:-]+")
NON_WORDY = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "is", "it", "of", "on", "or", "the", "to", "with"}


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _normalize_free_text(text: str) -> str:
    cleaned = text.lower()
    cleaned = cleaned.replace("`", "")
    cleaned = cleaned.replace("()", "")
    cleaned = cleaned.replace(">", " above ")
    cleaned = cleaned.replace("<", " below ")
    cleaned = re.sub(r"[^a-z0-9@/._:-]+", " ", cleaned)
    return _collapse_spaces(cleaned)


def _normalize_identifier(text: str) -> str:
    return _collapse_spaces(text.upper())


def _normalize_route(text: str) -> str:
    collapsed = _collapse_spaces(text)
    if " " not in collapsed:
        return collapsed.upper()
    method, path = collapsed.split(" ", 1)
    return f"{method.upper()} {path.lower()}"


def _normalize_path_line(text: str) -> str:
    normalized = _normalize_free_text(text)
    return normalized.replace(" line ", ":")


def _normalize_date(text: str) -> str:
    if match := DATE_PATTERN.search(text):
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    if match := LONG_DATE_PATTERN.search(text):
        month = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }[match.group(1).lower()[:3]]
        day = int(match.group(2))
        year = int(match.group(3))
        return datetime(year, month, day).strftime("%Y-%m-%d")
    return _normalize_free_text(text)


NORMALIZERS: dict[str, Callable[[str], str]] = {
    "text": _normalize_free_text,
    "identifier": _normalize_identifier,
    "route": _normalize_route,
    "path_line": _normalize_path_line,
    "date": _normalize_date,
    "email": _normalize_free_text,
    "version": _normalize_free_text,
    "symbol": _normalize_free_text,
    "phrase": _normalize_free_text,
}


PARSER_PATTERNS: dict[str, re.Pattern[str]] = {
    "identifier": IDENTIFIER_PATTERN,
    "route": ROUTE_PATTERN,
    "date": re.compile(rf"{DATE_PATTERN.pattern}|{LONG_DATE_PATTERN.pattern}", re.IGNORECASE),
    "email": EMAIL_PATTERN,
    "version": VERSION_PATTERN,
    "path_line": PATH_LINE_PATTERN,
    "symbol": SYMBOL_PATTERN,
    "quantity": QUANTITY_PATTERN,
}


def normalize_value(value: str, normalizer_id: str) -> str:
    normalizer = NORMALIZERS.get(normalizer_id, _normalize_free_text)
    return normalizer(value)


def build_phrase_pattern(value: str) -> re.Pattern[str]:
    escaped = re.escape(value).replace(r"\ ", r"\s+")
    if re.match(r"^[A-Za-z0-9_ /.-]+$", value):
        return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def extract_parser_candidates(text: str, parser_id: str) -> set[str]:
    if parser_id == "phrase":
        return extract_phrase_candidates(text)
    if parser_id == "path_line":
        return extract_path_line_candidates(text)

    pattern = PARSER_PATTERNS.get(parser_id)
    if pattern is None:
        return extract_phrase_candidates(text)

    candidates: set[str] = set()
    for match in pattern.finditer(text):
        if match.group(0):
            candidates.add(match.group(0))
    return candidates


def extract_phrase_candidates(text: str) -> set[str]:
    clauses = re.split(r"[;:\n]|(?:\s+-\s+)", text)
    candidates: set[str] = set()
    for clause in clauses:
        normalized = _normalize_free_text(clause)
        if not normalized:
            continue
        candidates.add(normalized)
        tokens = normalized.split()
        for size in range(2, min(10, len(tokens)) + 1):
            for index in range(len(tokens) - size + 1):
                candidates.add(" ".join(tokens[index:index + size]))
    return candidates


def extract_path_line_candidates(text: str) -> set[str]:
    candidates: set[str] = set()
    for match in PATH_LINE_PATTERN.finditer(text):
        candidate = _collapse_spaces(match.group(0).replace("`", ""))
        if candidate:
            candidates.add(candidate)
        path_only = _collapse_spaces(match.group(0).replace("`", ""))
        trailing_window = text[match.end(): match.end() + 32]
        line_match = re.search(r"(?:on\s+)?line\s+(\d+)", trailing_window, re.IGNORECASE)
        if line_match and ".py" in path_only and "line" not in path_only and ":" not in path_only:
            candidates.add(f"{path_only} line {line_match.group(1)}")
    return candidates


def token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(_normalize_free_text(left).split())
    right_tokens = set(_normalize_free_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def _stem_token(token: str) -> str:
    token = token.lower()
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def _salient_tokens(text: str) -> set[str]:
    return {
        _stem_token(token)
        for token in WORD_PATTERN.findall(_normalize_free_text(text))
        if token not in NON_WORDY
    }


def _strong_alias_match(alias: str, text: str, normalizer_id: str, parser_id: str) -> Optional[str]:
    exact_match = _exact_alias_match(alias, text, normalizer_id)
    if exact_match is not None:
        return exact_match
    return _fuzzy_alias_match(alias, text, normalizer_id, parser_id)


def _exact_alias_match(alias: str, text: str, normalizer_id: str) -> Optional[str]:
    alias_pattern = build_phrase_pattern(alias)
    if alias_pattern.search(text):
        return alias

    normalized_alias = normalize_value(alias, normalizer_id)
    normalized_text = normalize_value(text, normalizer_id)
    if normalized_alias and normalized_alias in normalized_text:
        return alias
    return None


def _fuzzy_alias_match(alias: str, text: str, normalizer_id: str, parser_id: str) -> Optional[str]:
    normalized_alias = normalize_value(alias, normalizer_id)
    normalized_text = normalize_value(text, normalizer_id)
    if parser_id in {"phrase", "quantity", "path_line"}:
        alias_tokens = _salient_tokens(alias)
        text_tokens = _salient_tokens(text)
        if alias_tokens and alias_tokens <= text_tokens:
            return alias
        if alias_tokens and text_tokens:
            shared_tokens = alias_tokens & text_tokens
            coverage = len(shared_tokens) / len(alias_tokens)
            numeric_alias_tokens = {token for token in alias_tokens if any(ch.isdigit() for ch in token)}
            if len(shared_tokens) >= min(2, len(alias_tokens)) and coverage >= 0.85 and numeric_alias_tokens <= text_tokens:
                return alias
    return None


def choose_audit_sample_size(run_count: int) -> int:
    return min(run_count, max(24, ceil(run_count * 0.10))) if run_count else 0


@dataclass(frozen=True)
class FactStatus:
    name: str
    primary_status: str
    secondary_status: str
    primary_evidence: Optional[str]
    secondary_evidence: Optional[str]
    source_message_indexes: tuple[int, ...]
    distractor_message_indexes: tuple[int, ...]

    @property
    def status(self) -> str:
        return self.primary_status


@dataclass(frozen=True)
class FactSpec:
    name: str
    canonical_value: str
    allowed_aliases: tuple[str, ...] = ()
    wrong_aliases: tuple[str, ...] = ()
    normalizer_id: str = "text"
    parser_id: str = "phrase"
    source_message_indexes: tuple[int, ...] = ()
    distractor_message_indexes: tuple[int, ...] = ()

    @property
    def value(self) -> str:
        return self.canonical_value

    @property
    def all_correct_aliases(self) -> tuple[str, ...]:
        return (self.canonical_value, *self.allowed_aliases)

    def primary_status(self, text: str) -> tuple[str, Optional[str]]:
        exact_wrong_hit = next(
            (
                matched
                for alias in self.wrong_aliases
                if (matched := _exact_alias_match(alias, text, self.normalizer_id))
            ),
            None,
        )
        exact_correct_hit = next(
            (
                matched
                for alias in self.all_correct_aliases
                if (matched := _exact_alias_match(alias, text, self.normalizer_id))
            ),
            None,
        )
        if exact_correct_hit and not exact_wrong_hit:
            return ("correct", exact_correct_hit)
        if exact_wrong_hit:
            return ("wrong", exact_wrong_hit)

        wrong_hit = next(
            (
                matched
                for alias in self.wrong_aliases
                if (matched := _fuzzy_alias_match(alias, text, self.normalizer_id, self.parser_id))
            ),
            None,
        )
        correct_hit = next(
            (
                matched
                for alias in self.all_correct_aliases
                if (matched := _fuzzy_alias_match(alias, text, self.normalizer_id, self.parser_id))
            ),
            None,
        )
        if correct_hit and not wrong_hit:
            return ("correct", correct_hit)
        if wrong_hit:
            return ("wrong", wrong_hit)
        return ("missing", None)

    def secondary_status(self, text: str) -> tuple[str, Optional[str]]:
        candidates = extract_parser_candidates(text, self.parser_id)
        normalized_candidates = {
            normalize_value(candidate, self.normalizer_id): candidate
            for candidate in candidates
        }
        correct_norms = {
            normalize_value(alias, self.normalizer_id): alias
            for alias in self.all_correct_aliases
        }
        wrong_norms = {
            normalize_value(alias, self.normalizer_id): alias
            for alias in self.wrong_aliases
        }
        wrong_hit = next(
            (normalized_candidates[norm] for norm in normalized_candidates if norm in wrong_norms),
            None,
        )
        correct_hit = next(
            (normalized_candidates[norm] for norm in normalized_candidates if norm in correct_norms),
            None,
        )
        if correct_hit is None:
            correct_hit = next(
                (
                    candidate
                    for candidate in candidates
                    for alias in self.all_correct_aliases
                    if _exact_alias_match(alias, candidate, self.normalizer_id)
                ),
                None,
            )
        if wrong_hit is None:
            wrong_hit = next(
                (
                    candidate
                    for candidate in candidates
                    for alias in self.wrong_aliases
                    if _exact_alias_match(alias, candidate, self.normalizer_id)
                ),
                None,
            )
        if correct_hit and not wrong_hit:
            return ("correct", correct_hit)
        if wrong_hit:
            return ("wrong", wrong_hit)
        if correct_hit is None:
            correct_hit = next(
                (
                    candidate
                    for candidate in candidates
                    for alias in self.all_correct_aliases
                    if _fuzzy_alias_match(alias, candidate, self.normalizer_id, self.parser_id)
                ),
                None,
            )
        if wrong_hit is None:
            wrong_hit = next(
                (
                    candidate
                    for candidate in candidates
                    for alias in self.wrong_aliases
                    if _fuzzy_alias_match(alias, candidate, self.normalizer_id, self.parser_id)
                ),
                None,
            )
        if correct_hit and not wrong_hit:
            return ("correct", correct_hit)
        if wrong_hit:
            return ("wrong", wrong_hit)
        return ("missing", None)


@dataclass(frozen=True)
class TaskInstance:
    name: str
    system_prompt: str
    turns: tuple[SessionTurn, ...]
    facts: tuple[FactSpec, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def score_response(self, text: str) -> dict[str, Any]:
        return _score_text_against_facts(self.facts, text)


@dataclass(frozen=True)
class FrozenReplayTask:
    name: str
    system_prompt: str
    messages: tuple[SessionTurn, ...]
    final_user_index: int
    facts: tuple[FactSpec, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def score_response(self, text: str) -> dict[str, Any]:
        return _score_text_against_facts(self.facts, text)


def _score_text_against_facts(
    facts: tuple[FactSpec, ...],
    text: str,
) -> dict[str, Any]:
        fact_results: list[FactStatus] = []
        for fact in facts:
            primary_status, primary_evidence = fact.primary_status(text)
            secondary_status, secondary_evidence = fact.secondary_status(text)
            fact_results.append(
                FactStatus(
                    name=fact.name,
                    primary_status=primary_status,
                    secondary_status=secondary_status,
                    primary_evidence=primary_evidence,
                    secondary_evidence=secondary_evidence,
                    source_message_indexes=fact.source_message_indexes,
                    distractor_message_indexes=fact.distractor_message_indexes,
                )
            )

        correct = [result.name for result in fact_results if result.primary_status == "correct"]
        wrong = [result.name for result in fact_results if result.primary_status == "wrong"]
        missing = [result.name for result in fact_results if result.primary_status == "missing"]
        secondary_correct = [result.name for result in fact_results if result.secondary_status == "correct"]
        contamination = [
            {"fact": result.name, "evidence": result.primary_evidence or result.secondary_evidence}
            for result in fact_results
            if result.primary_status == "wrong"
        ]
        score = max(0.0, (len(correct) - (0.5 * len(wrong))) / len(facts)) if facts else 0.0
        secondary_wrong = [result.name for result in fact_results if result.secondary_status == "wrong"]
        secondary_score = max(
            0.0,
            (len(secondary_correct) - (0.5 * len(secondary_wrong))) / len(facts),
        ) if facts else 0.0
        agreement = all(result.primary_status == result.secondary_status for result in fact_results)
        return {
            "fact_results": [
                {
                    "name": result.name,
                    "status": result.status,
                    "secondary_status": result.secondary_status,
                    "primary_evidence": result.primary_evidence,
                    "secondary_evidence": result.secondary_evidence,
                    "source_message_indexes": list(result.source_message_indexes),
                    "distractor_message_indexes": list(result.distractor_message_indexes),
                }
                for result in fact_results
            ],
            "found": correct,
            "wrong": wrong,
            "missing": missing,
            "missed": [*wrong, *missing],
            "contamination": contamination,
            "contamination_count": len(contamination),
            "score": round(score, 6),
            "secondary_score": round(secondary_score, 6),
            "secondary_found": secondary_correct,
            "secondary_wrong": secondary_wrong,
            "secondary_missing": [result.name for result in fact_results if result.secondary_status == "missing"],
            "scorer_agreement": agreement,
        }


@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    builder: TaskBuilder

    def build(
        self,
        backend: ContextBackend,
        model: str,
        window_budget: int,
        overflow_ratio: float,
        seed: int,
    ) -> TaskInstance:
        return self.builder(backend, model, window_budget, overflow_ratio, seed)


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    provider: str
    model_name: str
    window_limit: int
    backend_factory: BackendFactory
    tokenizer_model: Optional[str] = None
    tags: tuple[str, ...] = ()

    @property
    def is_primary_local(self) -> bool:
        return "primary_local" in self.tags


@dataclass(frozen=True)
class StrategySpec:
    name: str
    label: str
    include_in_adaptive_gate: bool = False
    internal_strategy: Optional[str] = None
    citation_enabled: bool = True
    extractor_mode: str = "spacy"

    def is_applicable(self, model: ModelSpec, window_budget: int) -> bool:
        if self.name in {"score_only", "barrier_lexical"}:
            return model.is_primary_local
        return True


@dataclass
class RunResult:
    task: str
    model: str
    model_name: str
    provider: str
    strategy: str
    window_budget: int
    seed: int
    score: float
    secondary_score: float
    found: list[str]
    wrong: list[str]
    missing: list[str]
    missed: list[str]
    fact_results: list[dict[str, Any]]
    contamination: list[dict[str, Any]]
    contamination_count: int
    scorer_agreement: bool
    final_response: str
    prompt_tokens: int
    usable_prompt_budget: int
    selected_indexes: list[int]
    protected_selected_indexes: list[int]
    cited_selected_indexes: list[int]
    protected_message_indexes: list[int]
    cited_message_indexes: list[int]
    retained_anchor: bool
    retained_distractor: bool
    anchor_protected: bool
    turn_count: int
    session_metadata: dict[str, Any]
    strategy_metadata: dict[str, Any]
    final_prompt_anchor_overlap: float = 0.0
    final_prompt_distractor_overlap: float = 0.0
    blind_id: Optional[str] = None
    barrier_extra_selected: bool = False
    barrier_extra_selected_indexes: list[int] = field(default_factory=list)
    barrier_rescue: bool = False
    barrier_rescue_facts: list[str] = field(default_factory=list)
    audit_required: bool = False

    def key(self) -> tuple[str, str, str, int, int]:
        return (self.task, self.model, self.strategy, self.window_budget, self.seed)

    def to_record(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "model": self.model,
            "model_name": self.model_name,
            "provider": self.provider,
            "strategy": self.strategy,
            "window_budget": self.window_budget,
            "seed": self.seed,
            "score": self.score,
            "secondary_score": self.secondary_score,
            "found": self.found,
            "wrong": self.wrong,
            "missing": self.missing,
            "missed": self.missed,
            "fact_results": self.fact_results,
            "contamination": self.contamination,
            "contamination_count": self.contamination_count,
            "scorer_agreement": self.scorer_agreement,
            "final_response": self.final_response,
            "prompt_tokens": self.prompt_tokens,
            "usable_prompt_budget": self.usable_prompt_budget,
            "selected_indexes": self.selected_indexes,
            "protected_selected_indexes": self.protected_selected_indexes,
            "cited_selected_indexes": self.cited_selected_indexes,
            "protected_message_indexes": self.protected_message_indexes,
            "cited_message_indexes": self.cited_message_indexes,
            "retained_anchor": self.retained_anchor,
            "retained_distractor": self.retained_distractor,
            "anchor_protected": self.anchor_protected,
            "turn_count": self.turn_count,
            "session_metadata": self.session_metadata,
            "strategy_metadata": self.strategy_metadata,
            "final_prompt_anchor_overlap": self.final_prompt_anchor_overlap,
            "final_prompt_distractor_overlap": self.final_prompt_distractor_overlap,
            "blind_id": self.blind_id,
            "barrier_extra_selected": self.barrier_extra_selected,
            "barrier_extra_selected_indexes": self.barrier_extra_selected_indexes,
            "barrier_rescue": self.barrier_rescue,
            "barrier_rescue_facts": self.barrier_rescue_facts,
            "audit_required": self.audit_required,
        }


@dataclass(frozen=True)
class AggregateResult:
    task: str
    model: str
    model_name: str
    provider: str
    strategy: str
    window_budget: int
    mean_score: float
    stddev_score: float
    mean_secondary_score: float
    n: int
    retained_anchor_rate: float
    retained_distractor_rate: float
    anchor_protected_rate: float
    contamination_rate: float
    scorer_agreement_rate: float
    barrier_rescue_rate: float
    delta_vs_barrier: float
    ci_low: float
    ci_high: float
    p_value: float
    wins: int
    ties: int
    losses: int

    @classmethod
    def from_runs(
        cls,
        *,
        task: str,
        model: str,
        model_name: str,
        provider: str,
        strategy: str,
        window_budget: int,
        runs: list[RunResult],
        delta_vs_barrier: float,
        ci_low: float,
        ci_high: float,
        p_value: float,
        wins: int,
        ties: int,
        losses: int,
    ) -> "AggregateResult":
        scores = [run.score for run in runs]
        return cls(
            task=task,
            model=model,
            model_name=model_name,
            provider=provider,
            strategy=strategy,
            window_budget=window_budget,
            mean_score=mean(scores) if scores else 0.0,
            stddev_score=pstdev(scores) if len(scores) > 1 else 0.0,
            mean_secondary_score=mean(run.secondary_score for run in runs) if runs else 0.0,
            n=len(runs),
            retained_anchor_rate=(sum(1 for run in runs if run.retained_anchor) / len(runs)) if runs else 0.0,
            retained_distractor_rate=(sum(1 for run in runs if run.retained_distractor) / len(runs)) if runs else 0.0,
            anchor_protected_rate=(sum(1 for run in runs if run.anchor_protected) / len(runs)) if runs else 0.0,
            contamination_rate=(sum(1 for run in runs if run.contamination_count > 0) / len(runs)) if runs else 0.0,
            scorer_agreement_rate=(sum(1 for run in runs if run.scorer_agreement) / len(runs)) if runs else 0.0,
            barrier_rescue_rate=(sum(1 for run in runs if run.barrier_rescue) / len(runs)) if runs else 0.0,
            delta_vs_barrier=delta_vs_barrier,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            wins=wins,
            ties=ties,
            losses=losses,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "model": self.model,
            "model_name": self.model_name,
            "provider": self.provider,
            "strategy": self.strategy,
            "window_budget": self.window_budget,
            "mean_score": self.mean_score,
            "stddev_score": self.stddev_score,
            "mean_secondary_score": self.mean_secondary_score,
            "n": self.n,
            "retained_anchor_rate": self.retained_anchor_rate,
            "retained_distractor_rate": self.retained_distractor_rate,
            "anchor_protected_rate": self.anchor_protected_rate,
            "contamination_rate": self.contamination_rate,
            "scorer_agreement_rate": self.scorer_agreement_rate,
            "barrier_rescue_rate": self.barrier_rescue_rate,
            "delta_vs_barrier": self.delta_vs_barrier,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
        }


def choose_audit_sample(records: list[RunResult], seed: int = 0) -> list[RunResult]:
    if not records:
        return []

    must_review = [
        run for run in records
        if not run.scorer_agreement or run.contamination_count > 0
    ]
    target = choose_audit_sample_size(len(records))
    if len(must_review) >= target:
        return must_review

    remaining = [run for run in records if run not in must_review]
    rng = Random(seed)
    rng.shuffle(remaining)
    return must_review + remaining[: max(0, target - len(must_review))]
