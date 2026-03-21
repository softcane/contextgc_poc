from __future__ import annotations

from random import Random
from typing import Iterable
import math
import re

from contextgc_barrier.backend import ContextBackend

from .specs import FactSpec, TaskInstance, TaskSpec, token_overlap_ratio


LEAKAGE_THRESHOLD = 0.18
REPLAY_LEAKAGE_THRESHOLD = 0.10
DEBUGGING_REPLAY_TARGET_TOKENS = 5200
DEBUGGING_REPLAY_TOKEN_TOLERANCE = 150

DEBUGGING_VALUES = {
    "function": {
        "process_batch()": ("process_batch()", "process_batch"),
        "flush_backlog()": ("flush_backlog()", "flush_backlog"),
    },
    "trigger": {
        "batch_size > 1000": ("batch_size > 1000", "batch size above 1000", "batches above 1000", "batch_size exceeds 1000"),
        "queue_depth > 4096": ("queue_depth > 4096", "queue depth above 4096", "queues above 4096", "queue_depth exceeds 4096"),
    },
    "symptom": {
        "RSS grows by 200MB per call": ("RSS grows by 200MB per call", "rss grows by 200mb per call", "memory rises by 200mb each call", "rss to grow by 200mb per call"),
        "RSS climbs by 160MB per retry": ("RSS climbs by 160MB per retry", "rss climbs by 160mb per retry", "memory climbs by 160mb on each retry", "rss to climb by 160mb per retry"),
    },
    "version": {
        "v2.3.1": ("v2.3.1",),
        "v2.3.4": ("v2.3.4",),
    },
    "file_line": {
        "src/batch_processor.py line 247": ("src/batch_processor.py line 247", "src/batch_processor.py:247"),
        "src/retry_worker.py line 188": ("src/retry_worker.py line 188", "src/retry_worker.py:188"),
    },
    "root_cause": {
        "numpy cache retains the input array after warmup": (
            "numpy cache retains the input array after warmup",
            "the numpy cache retains the input array after warmup",
            "numpy cache retains the input array",
            "the input array is retained in the numpy cache after warmup",
            "input array retained in the numpy cache after warmup",
        ),
        "retry buffer keeps the warmed arrays pinned in the numpy cache": (
            "retry buffer keeps the warmed arrays pinned in the numpy cache",
            "the retry buffer keeps warmed arrays pinned in the numpy cache",
            "retry buffer pinning warmed arrays in the numpy cache",
            "retry_buffer pinning warmed arrays in the numpy cache",
            "retry buffers pinning data in the numpy cache",
            "retry buffers pinning arrays in the numpy cache",
            "retry buffer pinning data in the numpy cache",
            "numpy arrays being pinned in the cache",
            "warmed arrays pinned in the numpy cache",
            "retry buffer pins the warmed arrays in the numpy cache",
            "retry_buffer pins the warmed arrays in the numpy cache",
            "retry_buffer keeps the warmed arrays pinned in the numpy cache",
        ),
    },
    "remediation": {
        "cache.clear() after each batch": (
            "cache.clear() after each batch",
            "run cache.clear() after each batch",
            "add a cache.clear() call after each batch",
            "adding a cache.clear() call immediately after each batch",
            "cache clear immediately after each batch",
            "clears the cache immediately after each batch",
        ),
        "clear the numpy cache after each retry window": (
            "clear the numpy cache after each retry window",
            "clear the numpy cache after every retry window",
            "cache.clear() immediately after each retry window",
            "add a cache.clear() call immediately after each retry window",
            "adding a cache.clear() call immediately after each retry window",
            "cache clear after each retry window",
            "cache clear operation immediately after each retry window",
            "clears the numpy cache immediately after each retry window",
            "clears the cache immediately after the retry window completes",
            "which clears the numpy cache immediately after each retry window",
        ),
    },
}

DOCUMENT_VALUES = {
    "policy_name": {
        "Data Export Exception Policy": ("Data Export Exception Policy",),
        "Restricted Dataset Release Policy": ("Restricted Dataset Release Policy",),
    },
    "retention_period": {
        "45 days": ("45 days",),
        "60 days": ("60 days",),
    },
    "exception": {
        "security incidents only": ("security incidents only", "security incident requests only"),
        "audited regulator requests only": ("audited regulator requests only", "audited regulator request only"),
    },
    "approver": {
        "Director of Trust Operations": ("Director of Trust Operations",),
        "VP of Security Governance": ("VP of Security Governance",),
    },
    "effective_date": {
        "2025-09-15": ("2025-09-15", "September 15, 2025"),
        "2025-11-01": ("2025-11-01", "November 1, 2025"),
    },
    "form_id": {
        "FORM-EX17": ("FORM-EX17",),
        "RDR-204": ("RDR-204",),
    },
    "escalation_contact": {
        "policy-ops@acme.test": ("policy-ops@acme.test",),
        "risk-desk@acme.test": ("risk-desk@acme.test",),
    },
}

CODING_VALUES = {
    "module": {
        "services/export_pipeline.py": ("services/export_pipeline.py",),
        "services/bulk_sync.py": ("services/bulk_sync.py",),
    },
    "feature_flag": {
        "EXPORT_V2_GUARD": ("EXPORT_V2_GUARD",),
        "BULK_SYNC_CANARY": ("BULK_SYNC_CANARY",),
    },
    "endpoint": {
        "POST /v1/exports/replay": ("POST /v1/exports/replay",),
        "POST /v1/sync/requeue": ("POST /v1/sync/requeue",),
    },
    "test_name": {
        "test_replay_preserves_cursor": ("test_replay_preserves_cursor",),
        "test_sync_requeue_is_idempotent": ("test_sync_requeue_is_idempotent",),
    },
    "metric": {
        "exports.replay.cursor_mismatch": ("exports.replay.cursor_mismatch",),
        "sync.requeue.duplicate_jobs": ("sync.requeue.duplicate_jobs",),
    },
    "migration_date": {
        "2026-01-12": ("2026-01-12", "January 12, 2026"),
        "2026-02-03": ("2026-02-03", "February 3, 2026"),
    },
    "rollback": {
        "disable the flag and drain replay jobs": (
            "disable the flag and drain replay jobs",
            "turn off the flag and drain replay jobs",
        ),
        "disable the canary and flush staged sync jobs": (
            "disable the canary and flush staged sync jobs",
            "turn off the canary and flush staged sync jobs",
        ),
    },
}

SUPPORT_VALUES = {
    "order_id": {
        "A-10472": ("A-10472",),
        "C-88314": ("C-88314",),
    },
    "sku": {
        "SKU-THERM-9": ("SKU-THERM-9",),
        "SKU-COOL-4": ("SKU-COOL-4",),
    },
    "defect": {
        "thermostat fails after 8 minutes": (
            "thermostat fails after 8 minutes",
            "thermostat failure after 8 minutes",
        ),
        "cooling fan stalls on speed 3": (
            "cooling fan stalls on speed 3",
            "cooling fan stalling on speed 3",
        ),
    },
    "incident_date": {
        "2026-02-18": ("2026-02-18", "February 18, 2026"),
        "2026-02-24": ("2026-02-24", "February 24, 2026"),
    },
    "refund": {
        "full refund to original payment method": (
            "full refund to original payment method",
            "a full refund to the original payment method",
        ),
        "store credit plus shipping reimbursement": (
            "store credit plus shipping reimbursement",
            "store credit with shipping reimbursement",
        ),
    },
    "replacement_eta": {
        "replacement ships within 2 business days": (
            "replacement ships within 2 business days",
            "replacement ships in 2 business days",
        ),
        "replacement ships by next-day courier": (
            "replacement ships by next-day courier",
            "replacement ships via next-day courier",
        ),
    },
    "shipping_tier": {
        "priority overnight": ("priority overnight",),
        "two-day expedited": ("two-day expedited",),
    },
}


def build_task_specs() -> list[TaskSpec]:
    return [
        TaskSpec(
            name="debugging",
            description="Long debugging session with split anchors and confusable side-case incidents.",
            builder=_build_debugging_task,
        ),
        TaskSpec(
            name="document_qa",
            description="Document Q&A with active and archived policy excerpts using overlapping vocabulary.",
            builder=_build_document_qa_task,
        ),
        TaskSpec(
            name="multi_step_coding",
            description="Rollout handoff with active and stale implementation briefs.",
            builder=_build_multi_step_coding_task,
        ),
        TaskSpec(
            name="customer_support",
            description="Support closure thread with active and historical case details.",
            builder=_build_customer_support_task,
        ),
    ]


def _build_debugging_task(
    backend: ContextBackend,
    model: str,
    window_budget: int,
    overflow_ratio: float,
    seed: int,
) -> TaskInstance:
    return _build_debugging_variant_task(
        backend=backend,
        model=model,
        seed=seed,
        template_name="incident_closeout",
        window_budget=window_budget,
        overflow_ratio=overflow_ratio,
    )


def build_debugging_replay_seed_task(
    backend: ContextBackend,
    model: str,
    *,
    source_seed: int,
    template_name: str,
    target_total_tokens: int = DEBUGGING_REPLAY_TARGET_TOKENS,
) -> TaskInstance:
    return _build_debugging_variant_task(
        backend=backend,
        model=model,
        seed=source_seed,
        template_name=template_name,
        target_total_tokens=target_total_tokens,
        leakage_threshold=REPLAY_LEAKAGE_THRESHOLD,
    )


def _build_debugging_variant_task(
    *,
    backend: ContextBackend,
    model: str,
    seed: int,
    template_name: str,
    window_budget: int | None = None,
    overflow_ratio: float = 1.15,
    target_total_tokens: int | None = None,
    leakage_threshold: float = LEAKAGE_THRESHOLD,
) -> TaskInstance:
    rng = Random(seed)
    anchor = _choose_fact_values(DEBUGGING_VALUES, rng)
    distractor = _distractor_values(DEBUGGING_VALUES, anchor)
    final_prompt = _choose_prompt_variant(
        seed=seed,
        anchor_values=anchor.values(),
        distractor_values=distractor.values(),
        prompt_variants=_debugging_final_prompt_variants(template_name),
        leakage_threshold=leakage_threshold,
    )
    template_turns = _debugging_template_turns(
        template_name=template_name,
        anchor=anchor,
        distractor=distractor,
        final_prompt=final_prompt,
    )
    if window_budget is None and target_total_tokens is None:
        raise ValueError("debugging variant task requires either window_budget or target_total_tokens")
    turns = _materialize_turns(
        backend=backend,
        model=model,
        window_budget=window_budget or target_total_tokens or DEBUGGING_REPLAY_TARGET_TOKENS,
        overflow_ratio=overflow_ratio,
        seed=seed,
        template_turns=template_turns,
        target_total_tokens=target_total_tokens,
    )
    facts = _make_facts(
        anchor=anchor,
        distractor=distractor,
        source_turns={
            "function": (0,),
            "trigger": (0,),
            "symptom": (0,),
            "version": (0,),
            "file_line": (1,),
            "root_cause": (1,),
            "remediation": (2,),
        },
        distractor_turns={name: (6, 10) for name in anchor},
        normalizer_ids={
            "function": "symbol",
            "trigger": "phrase",
            "symptom": "phrase",
            "version": "version",
            "file_line": "path_line",
            "root_cause": "phrase",
            "remediation": "phrase",
        },
        parser_ids={
            "function": "symbol",
            "trigger": "phrase",
            "symptom": "phrase",
            "version": "version",
            "file_line": "path_line",
            "root_cause": "phrase",
            "remediation": "phrase",
        },
        value_aliases=DEBUGGING_VALUES,
    )
    return TaskInstance(
        name="debugging",
        system_prompt="You are a terse engineering assistant. Keep track of the active incident even when similar side-cases appear later.",
        turns=tuple(turns),
        facts=facts,
        metadata=_task_metadata(
            seed=seed,
            final_prompt=final_prompt,
            anchor=anchor,
            distractor=distractor,
            facts=facts,
            anchor_turns=(0, 1, 2),
            distractor_turns=(6, 10),
        ),
    )


def _debugging_final_prompt_variants(template_name: str) -> tuple[str, ...]:
    if template_name == "root_cause_recap":
        return (
            "Write one short plain-text sentence for the release manager, with no heading or bullets. Name the active failing routine, the live reproduction condition, the production behavior teams observed, the shipped build, the code location engineering edited, the real diagnosis, and the exact mitigation that shipped. Use the live incident, not the stale case.",
            "Draft the diagnosis handoff as one short plain-text sentence, not a heading, bullet list, or template. Cover the live failure path, what triggers it, what production showed, the affected build, where engineers patched it, the actual root cause, and the exact cleanup step that went out. Ignore the stale side-case.",
            "Prepare the final diagnosis note for the active incident as one short plain-text sentence with no markdown. Include the true routine, trigger, runtime behavior, version, patch location, diagnosis, and the exact shipped remediation. Do not use the stale case.",
        )
    return (
        "Write the short incident closeout for the release manager. Name the active failure path, the condition that reliably trips it, how it behaves in production, the affected shipped build, where engineering patched it, what caused it, and the corrective step that shipped. Use the live incident, not the side-case.",
        "Draft the postmortem close note in one paragraph. Explain which active code path failed, when it reliably reproduces, the production behavior teams saw, the affected release, the code location engineers touched, the diagnosis, and the shipped fix. Ignore the archived side-case.",
        "Prepare the engineering handoff paragraph for the active incident. Cover the real failing routine, the trigger condition, the runtime behavior, the shipped version, the patch location, the diagnosis, and the mitigation that went out. Do not use the side-case.",
    )


def _debugging_template_turns(
    *,
    template_name: str,
    anchor: dict[str, str],
    distractor: dict[str, str],
    final_prompt: str,
) -> list[dict[str, str]]:
    if template_name == "root_cause_recap":
        return [
            {
                "role": "user",
                "content": (
                    "Live incident recap for tonight's escalation:\n"
                    f"- Active failing routine: {anchor['function']} on build {anchor['version']}.\n"
                    f"- Reliable trigger: {anchor['trigger']}.\n"
                    f"- Production behavior: {anchor['symptom']}."
                ),
            },
            {
                "role": "tool",
                "content": (
                    "Diagnosis board for the live incident:\n"
                    f"- Patch location: {anchor['file_line']}.\n"
                    f"- Root-cause note: {anchor['root_cause']}."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Mitigation note for the live incident. Keep this exact shipped cleanup step in the release-ready recap: "
                    f"{anchor['remediation']}."
                ),
            },
            {
                "role": "user",
                "content": "Before the stale notes land, restate the live diagnosis and the patch location in one short plain-text sentence.",
            },
            {"role": "tool", "content": "{noise:1:retry telemetry export}"},
            {
                "role": "user",
                "content": (
                    "Now restate the true root cause and the exact shipped cleanup step in one short plain-text sentence, "
                    "without quoting the earlier notes. Name the cleanup step explicitly."
                ),
            },
            {
                "role": "tool",
                "content": (
                    "Stale side-case from an older rollback. It is not the live incident:\n"
                    f"- Failing routine: {distractor['function']} on build {distractor['version']}.\n"
                    f"- Trigger: {distractor['trigger']}.\n"
                    f"- Runtime behavior: {distractor['symptom']}.\n"
                    f"- Suspected location: {distractor['file_line']}.\n"
                    f"- Root cause: {distractor['root_cause']}.\n"
                    f"- Remediation: {distractor['remediation']}."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Ignore the stale case and in one short plain-text sentence explain why the live incident reproduces, "
                    "what engineering changed, and which cleanup step actually shipped."
                ),
            },
            {"role": "tool", "content": "{noise:2:memory graph export}"},
            {
                "role": "tool",
                "content": (
                    "Archived remediation chatter keeps repeating the wrong diagnosis and fix: "
                    f"{distractor['root_cause']} and {distractor['remediation']}. Do not carry those into the live incident note."
                ),
            },
            {
                "role": "user",
                "content": (
                    "I only need one short plain-text release-ready sentence now. No heading, no bullets; include the live diagnosis and exact cleanup step."
                ),
            },
            {"role": "tool", "content": "{noise:3:profiling dump}"},
            {"role": "user", "content": final_prompt},
        ]

    return [
        {
            "role": "user",
            "content": (
                "Active incident handoff for tonight's release review:\n"
                f"- The failure path is {anchor['function']} on build {anchor['version']}.\n"
                f"- It becomes reproducible once {anchor['trigger']}.\n"
                f"- In production the symptom is that {anchor['symptom']}."
            ),
        },
        {
            "role": "tool",
            "content": (
                "Active incident trace excerpt:\n"
                f"- Patch location under discussion: {anchor['file_line']}.\n"
                f"- Working diagnosis: {anchor['root_cause']}."
            ),
        },
        {
            "role": "user",
            "content": f"Ship-note reminder for the active incident: the corrective step is to {anchor['remediation']}.",
        },
        {
            "role": "user",
            "content": "Before more logs arrive, tell me exactly which routine and threshold define the live incident.",
        },
        {"role": "tool", "content": "{noise:1:retry telemetry export}"},
        {
            "role": "user",
            "content": "Now give me the exact patch location and the exact cleanup action tied to the live incident.",
        },
        {
            "role": "tool",
            "content": (
                "Archived side-case from a different canary. Do not use it for the live incident:\n"
                f"- Failing path: {distractor['function']} on build {distractor['version']}.\n"
                f"- Reproduction condition: {distractor['trigger']}.\n"
                f"- Runtime behavior: {distractor['symptom']}.\n"
                f"- Suspected location: {distractor['file_line']}.\n"
                f"- Diagnosis: {distractor['root_cause']}.\n"
                f"- Fix: {distractor['remediation']}."
            ),
        },
        {
            "role": "user",
            "content": "Ignore the side-case and restate the live diagnosis in plain English without quoting the original wording.",
        },
        {"role": "tool", "content": "{noise:2:profiling dump}"},
        {
            "role": "user",
            "content": "Which shipped build is still implicated, and what production behavior makes this incident worth blocking on?",
        },
        {
            "role": "tool",
            "content": (
                "Meeting note with overlapping vocabulary from the old canary: release managers kept discussing "
                f"{distractor['version']}, {distractor['file_line']}, and {distractor['function']} because that side-case looked similar."
            ),
        },
        {
            "role": "user",
            "content": "I only need the live incident closeout now. Keep it short and release-ready.",
        },
        {"role": "tool", "content": "{noise:3:memory graph export}"},
        {"role": "user", "content": final_prompt},
    ]


def _build_document_qa_task(
    backend: ContextBackend,
    model: str,
    window_budget: int,
    overflow_ratio: float,
    seed: int,
) -> TaskInstance:
    rng = Random(seed * 7 + 11)
    anchor = _choose_fact_values(DOCUMENT_VALUES, rng)
    distractor = _distractor_values(DOCUMENT_VALUES, anchor)
    final_prompt = _choose_prompt_variant(
        seed=seed,
        anchor_values=anchor.values(),
        distractor_values=distractor.values(),
        prompt_variants=(
            "Reply to counsel with one short paragraph that states the active controlling rule, how long it lasts, which requests qualify, who approves it, when it took effect, which paperwork is required, and where the escalation goes. Use the live rule, not the archived one.",
            "Draft the final policy answer in one paragraph. Summarize the active governing rule, the retention limit, the qualifying request type, the sign-off role, the activation moment, the form teams must file, and the escalation route. Ignore the archived rule.",
            "Prepare the compliance response for the active rule set. Cover the controlling rule name, how long it remains in force, what category of request is eligible, who signs it, when it began, which filing artifact is required, and the escalation destination. Do not use the archived rule.",
        ),
    )
    template_turns = [
        {
            "role": "user",
            "content": (
                "Active policy excerpt for today's answer:\n"
                f"- Governing rule: {anchor['policy_name']}.\n"
                f"- Retention limit: {anchor['retention_period']}.\n"
                f"- Qualifying request type: {anchor['exception']}."
            ),
        },
        {
            "role": "tool",
            "content": (
                "Active approval note:\n"
                f"- Signing role: {anchor['approver']}.\n"
                f"- Activation date: {anchor['effective_date']}.\n"
                f"- Required filing artifact: {anchor['form_id']}."
            ),
        },
        {
            "role": "user",
            "content": f"Escalation note for the active rule: route risky questions to {anchor['escalation_contact']}.",
        },
        {
            "role": "user",
            "content": "Before the archive material shows up, tell me exactly which signing role and filing artifact belong to the active rule.",
        },
        {"role": "tool", "content": "{noise:1:appendix excerpts}"},
        {
            "role": "user",
            "content": "Now give me the exact escalation route and the exact qualifying request type for the active rule.",
        },
        {
            "role": "tool",
            "content": (
                "Archived policy excerpt that looks similar but is not controlling:\n"
                f"- Governing rule: {distractor['policy_name']}.\n"
                f"- Retention limit: {distractor['retention_period']}.\n"
                f"- Qualifying request type: {distractor['exception']}.\n"
                f"- Signing role: {distractor['approver']}.\n"
                f"- Activation date: {distractor['effective_date']}.\n"
                f"- Filing artifact: {distractor['form_id']}.\n"
                f"- Escalation route: {distractor['escalation_contact']}."
            ),
        },
        {
            "role": "user",
            "content": "Ignore the archived rule and restate the active rule in plain English without listing fields.",
        },
        {"role": "tool", "content": "{noise:2:archived meeting notes}"},
        {
            "role": "user",
            "content": "What timing boundary still matters for the active rule, and where does a risky request go next?",
        },
        {
            "role": "tool",
            "content": (
                "Legal thread from the archive keeps repeating overlapping terms like "
                f"{distractor['policy_name']}, {distractor['approver']}, and {distractor['form_id']} even though that is not the active rule."
            ),
        },
        {
            "role": "user",
            "content": "I only need the live answer now. Keep it brief and ready to send.",
        },
        {"role": "tool", "content": "{noise:3:retention exception archive}"},
        {"role": "user", "content": final_prompt},
    ]
    turns = _materialize_turns(
        backend=backend,
        model=model,
        window_budget=window_budget,
        overflow_ratio=overflow_ratio,
        seed=seed,
        template_turns=template_turns,
    )
    facts = _make_facts(
        anchor=anchor,
        distractor=distractor,
        source_turns={
            "policy_name": (0,),
            "retention_period": (0,),
            "exception": (0,),
            "approver": (1,),
            "effective_date": (1,),
            "form_id": (1,),
            "escalation_contact": (2,),
        },
        distractor_turns={name: (6, 10) for name in anchor},
        normalizer_ids={
            "policy_name": "text",
            "retention_period": "phrase",
            "exception": "phrase",
            "approver": "text",
            "effective_date": "date",
            "form_id": "identifier",
            "escalation_contact": "email",
        },
        parser_ids={
            "policy_name": "phrase",
            "retention_period": "quantity",
            "exception": "phrase",
            "approver": "phrase",
            "effective_date": "date",
            "form_id": "identifier",
            "escalation_contact": "email",
        },
        value_aliases=DOCUMENT_VALUES,
    )
    return TaskInstance(
        name="document_qa",
        system_prompt="You answer compliance questions. Preserve the active rule even if an archived rule with similar vocabulary appears later.",
        turns=tuple(turns),
        facts=facts,
        metadata=_task_metadata(
            seed=seed,
            final_prompt=final_prompt,
            anchor=anchor,
            distractor=distractor,
            facts=facts,
            anchor_turns=(0, 1, 2),
            distractor_turns=(6, 10),
        ),
    )


def _build_multi_step_coding_task(
    backend: ContextBackend,
    model: str,
    window_budget: int,
    overflow_ratio: float,
    seed: int,
) -> TaskInstance:
    rng = Random(seed * 13 + 19)
    anchor = _choose_fact_values(CODING_VALUES, rng)
    distractor = _distractor_values(CODING_VALUES, anchor)
    final_prompt = _choose_prompt_variant(
        seed=seed,
        anchor_values=anchor.values(),
        distractor_values=distractor.values(),
        prompt_variants=(
            "Write the release handoff paragraph for the on-call engineer. Cover the active code area, the rollout switch, the request surface touched by the change, the regression check to watch, the production metric to monitor, when rollout begins, and the fallback move if it goes sideways. Use the active rollout, not the stale canary.",
            "Prepare the handoff note for the active rollout in one paragraph. Mention the real owning module, the gating switch, the request surface involved, the regression check, the watch metric, the rollout date, and the fallback action. Ignore the stale canary brief.",
            "Draft the rollout summary for operations. Name the active code area, its gate, the touched request path, the regression check, the watch metric, the rollout start, and the fallback step. Do not use the stale canary plan.",
        ),
    )
    template_turns = [
        {
            "role": "user",
            "content": (
                "Active rollout brief:\n"
                f"- Owning code area: {anchor['module']}.\n"
                f"- Rollout gate: {anchor['feature_flag']}.\n"
                f"- Request surface: {anchor['endpoint']}."
            ),
        },
        {
            "role": "tool",
            "content": (
                "Active rollout checklist:\n"
                f"- Regression check: {anchor['test_name']}.\n"
                f"- Watch metric: {anchor['metric']}.\n"
                f"- Rollout date: {anchor['migration_date']}."
            ),
        },
        {
            "role": "user",
            "content": f"Fallback note for the active rollout: {anchor['rollback']}.",
        },
        {
            "role": "user",
            "content": "Before the stale brief arrives, tell me exactly which gate and request surface belong to the active rollout.",
        },
        {"role": "tool", "content": "{noise:1:test run output}"},
        {
            "role": "user",
            "content": "Now give me the exact regression check and fallback step tied to the active rollout.",
        },
        {
            "role": "tool",
            "content": (
                "Stale canary brief from an older rollout. Do not use it for the active handoff:\n"
                f"- Owning code area: {distractor['module']}.\n"
                f"- Rollout gate: {distractor['feature_flag']}.\n"
                f"- Request surface: {distractor['endpoint']}.\n"
                f"- Regression check: {distractor['test_name']}.\n"
                f"- Watch metric: {distractor['metric']}.\n"
                f"- Rollout date: {distractor['migration_date']}.\n"
                f"- Fallback: {distractor['rollback']}."
            ),
        },
        {
            "role": "user",
            "content": "Ignore the stale canary and restate the active rollout in plain English without a checklist.",
        },
        {"role": "tool", "content": "{noise:2:unrelated CI shards}"},
        {
            "role": "user",
            "content": "What date and monitoring hook still matter for the active rollout?",
        },
        {
            "role": "tool",
            "content": (
                "Old migration notes keep repeating overlapping items such as "
                f"{distractor['feature_flag']}, {distractor['endpoint']}, and {distractor['metric']} even though they belong to the stale canary."
            ),
        },
        {
            "role": "user",
            "content": "I only need the live handoff now. Make it operations-ready.",
        },
        {"role": "tool", "content": "{noise:3:alert snapshots}"},
        {"role": "user", "content": final_prompt},
    ]
    turns = _materialize_turns(
        backend=backend,
        model=model,
        window_budget=window_budget,
        overflow_ratio=overflow_ratio,
        seed=seed,
        template_turns=template_turns,
    )
    facts = _make_facts(
        anchor=anchor,
        distractor=distractor,
        source_turns={
            "module": (0,),
            "feature_flag": (0,),
            "endpoint": (0,),
            "test_name": (1,),
            "metric": (1,),
            "migration_date": (1,),
            "rollback": (2,),
        },
        distractor_turns={name: (6, 10) for name in anchor},
        normalizer_ids={
            "module": "path_line",
            "feature_flag": "identifier",
            "endpoint": "route",
            "test_name": "identifier",
            "metric": "identifier",
            "migration_date": "date",
            "rollback": "phrase",
        },
        parser_ids={
            "module": "path_line",
            "feature_flag": "identifier",
            "endpoint": "route",
            "test_name": "identifier",
            "metric": "identifier",
            "migration_date": "date",
            "rollback": "phrase",
        },
        value_aliases=CODING_VALUES,
    )
    return TaskInstance(
        name="multi_step_coding",
        system_prompt="You are a rollout assistant. Preserve the active rollout even when a stale canary brief with similar vocabulary appears later.",
        turns=tuple(turns),
        facts=facts,
        metadata=_task_metadata(
            seed=seed,
            final_prompt=final_prompt,
            anchor=anchor,
            distractor=distractor,
            facts=facts,
            anchor_turns=(0, 1, 2),
            distractor_turns=(6, 10),
        ),
    )


def _build_customer_support_task(
    backend: ContextBackend,
    model: str,
    window_budget: int,
    overflow_ratio: float,
    seed: int,
) -> TaskInstance:
    rng = Random(seed * 23 + 29)
    anchor = _choose_fact_values(SUPPORT_VALUES, rng)
    distractor = _distractor_values(SUPPORT_VALUES, anchor)
    final_prompt = _choose_prompt_variant(
        seed=seed,
        anchor_values=anchor.values(),
        distractor_values=distractor.values(),
        prompt_variants=(
            "Write the final reply the support lead will send. Mention the active case reference, product, failure description, incident day, make-good offered, replacement timing, and delivery service. Use the live case, not the historical case.",
            "Draft the closing note for the active customer case. Include the live case reference, the product involved, what failed, the day of the incident, the compensation offered, when the replacement goes out, and the delivery service. Ignore the historical case.",
            "Prepare the final customer response in one short paragraph. Cover the live case reference, the product, the failure, when it happened, the compensation, the replacement timing, and the shipping service. Do not use the historical case.",
        ),
    )
    template_turns = [
        {
            "role": "user",
            "content": (
                "Active CRM intake summary:\n"
                f"- Case reference: {anchor['order_id']}.\n"
                f"- Product: {anchor['sku']}.\n"
                f"- Reported failure: {anchor['defect']}.\n"
                f"- Incident day: {anchor['incident_date']}."
            ),
        },
        {
            "role": "tool",
            "content": (
                "Active compensation note:\n"
                f"- Make-good: {anchor['refund']}.\n"
                f"- Replacement timing: {anchor['replacement_eta']}."
            ),
        },
        {
            "role": "user",
            "content": f"Shipping note for the active case: use {anchor['shipping_tier']}.",
        },
        {
            "role": "user",
            "content": "Before the old case history shows up, tell me exactly what compensation and replacement timing belong to the active case.",
        },
        {"role": "tool", "content": "{noise:1:crm timeline export}"},
        {
            "role": "user",
            "content": "Now give me the exact shipping service and the exact case reference for the active case.",
        },
        {
            "role": "tool",
            "content": (
                "Historical case with similar vocabulary. Do not use it for the active reply:\n"
                f"- Case reference: {distractor['order_id']}.\n"
                f"- Product: {distractor['sku']}.\n"
                f"- Failure: {distractor['defect']}.\n"
                f"- Incident day: {distractor['incident_date']}.\n"
                f"- Make-good: {distractor['refund']}.\n"
                f"- Replacement timing: {distractor['replacement_eta']}.\n"
                f"- Shipping service: {distractor['shipping_tier']}."
            ),
        },
        {
            "role": "user",
            "content": "Ignore the historical case and restate the active case in plain English without a bullet list.",
        },
        {"role": "tool", "content": "{noise:2:warehouse notes}"},
        {
            "role": "user",
            "content": "What timing and delivery promise still matter for the live customer case?",
        },
        {
            "role": "tool",
            "content": (
                "Old case review keeps repeating terms like "
                f"{distractor['order_id']}, {distractor['sku']}, and {distractor['shipping_tier']} even though that belongs to the historical case."
            ),
        },
        {
            "role": "user",
            "content": "I only need the live customer closeout now. Keep it ready to send.",
        },
        {"role": "tool", "content": "{noise:3:order amendment history}"},
        {"role": "user", "content": final_prompt},
    ]
    turns = _materialize_turns(
        backend=backend,
        model=model,
        window_budget=window_budget,
        overflow_ratio=overflow_ratio,
        seed=seed,
        template_turns=template_turns,
    )
    facts = _make_facts(
        anchor=anchor,
        distractor=distractor,
        source_turns={
            "order_id": (0,),
            "sku": (0,),
            "defect": (0,),
            "incident_date": (0,),
            "refund": (1,),
            "replacement_eta": (1,),
            "shipping_tier": (2,),
        },
        distractor_turns={name: (6, 10) for name in anchor},
        normalizer_ids={
            "order_id": "identifier",
            "sku": "identifier",
            "defect": "phrase",
            "incident_date": "date",
            "refund": "phrase",
            "replacement_eta": "phrase",
            "shipping_tier": "phrase",
        },
        parser_ids={
            "order_id": "identifier",
            "sku": "identifier",
            "defect": "phrase",
            "incident_date": "date",
            "refund": "phrase",
            "replacement_eta": "quantity",
            "shipping_tier": "quantity",
        },
        value_aliases=SUPPORT_VALUES,
    )
    return TaskInstance(
        name="customer_support",
        system_prompt="You are a support assistant. Keep the live customer case straight even when a historical case with similar wording appears later.",
        turns=tuple(turns),
        facts=facts,
        metadata=_task_metadata(
            seed=seed,
            final_prompt=final_prompt,
            anchor=anchor,
            distractor=distractor,
            facts=facts,
            anchor_turns=(0, 1, 2),
            distractor_turns=(6, 10),
        ),
    )


def _choose_fact_values(values: dict[str, dict[str, tuple[str, ...]]], rng: Random) -> dict[str, str]:
    return {
        name: rng.choice(list(pool.keys()))
        for name, pool in values.items()
    }


def _distractor_values(values: dict[str, dict[str, tuple[str, ...]]], anchor: dict[str, str]) -> dict[str, str]:
    distractor = {}
    for name, pool in values.items():
        distractor[name] = next(value for value in pool if value != anchor[name])
    return distractor


def _choose_prompt_variant(
    *,
    seed: int,
    anchor_values: Iterable[str],
    distractor_values: Iterable[str],
    prompt_variants: tuple[str, ...],
    leakage_threshold: float = LEAKAGE_THRESHOLD,
) -> str:
    anchor_text = " ".join(anchor_values)
    distractor_text = " ".join(distractor_values)
    scored_variants = sorted(
        (
            (
                token_overlap_ratio(prompt, anchor_text),
                token_overlap_ratio(prompt, distractor_text),
                index,
                prompt,
            )
            for index, prompt in enumerate(prompt_variants)
        ),
        key=lambda item: (item[0], item[1], (seed + item[2]) % len(prompt_variants)),
    )
    best = scored_variants[0]
    if best[0] > leakage_threshold:
        raise ValueError(f"No prompt variant passed the leakage threshold: {best[0]:.3f}")
    return best[3]


def _make_facts(
    *,
    anchor: dict[str, str],
    distractor: dict[str, str],
    source_turns: dict[str, tuple[int, ...]],
    distractor_turns: dict[str, tuple[int, ...]],
    normalizer_ids: dict[str, str],
    parser_ids: dict[str, str],
    value_aliases: dict[str, dict[str, tuple[str, ...]]],
) -> tuple[FactSpec, ...]:
    facts = []
    for name, value in anchor.items():
        facts.append(
            FactSpec(
                name=name,
                canonical_value=value,
                allowed_aliases=tuple(alias for alias in value_aliases[name][value] if alias != value),
                wrong_aliases=value_aliases[name][distractor[name]],
                normalizer_id=normalizer_ids.get(name, "text"),
                parser_id=parser_ids.get(name, "phrase"),
                source_message_indexes=tuple(_turn_to_message_index(turn) for turn in source_turns[name]),
                distractor_message_indexes=tuple(_turn_to_message_index(turn) for turn in distractor_turns[name]),
            )
        )
    return tuple(facts)


def _task_metadata(
    *,
    seed: int,
    final_prompt: str,
    anchor: dict[str, str],
    distractor: dict[str, str],
    facts: tuple[FactSpec, ...],
    anchor_turns: tuple[int, ...],
    distractor_turns: tuple[int, ...],
) -> dict[str, object]:
    anchor_text = " ".join(anchor.values())
    distractor_text = " ".join(distractor.values())
    return {
        "seed": seed,
        "reviewer_version": "debugging_replay",
        "anchor_indexes": [_turn_to_message_index(turn) for turn in anchor_turns],
        "distractor_indexes": [_turn_to_message_index(turn) for turn in distractor_turns],
        "fact_source_indexes": {
            fact.name: list(fact.source_message_indexes)
            for fact in facts
        },
        "final_prompt": final_prompt,
        "final_prompt_anchor_overlap": token_overlap_ratio(final_prompt, anchor_text),
        "final_prompt_distractor_overlap": token_overlap_ratio(final_prompt, distractor_text),
    }


def _turn_to_message_index(turn_index: int) -> int:
    return 1 + (2 * turn_index)


def _materialize_turns(
    *,
    backend: ContextBackend,
    model: str,
    window_budget: int,
    overflow_ratio: float,
    seed: int,
    template_turns: list[dict[str, str]],
    target_total_tokens: int | None = None,
) -> list[dict[str, str]]:
    placeholder_messages = [{"role": "system", "content": "placeholder"}]
    placeholder_messages.extend(
        {
            "role": turn["role"],
            "content": "placeholder block"
            if turn["content"].startswith("{noise:")
            else turn["content"],
        }
        for turn in template_turns
    )
    base_tokens = backend.count_tokens(placeholder_messages, model=model)
    target_tokens = (
        max(base_tokens + 1, target_total_tokens)
        if target_total_tokens is not None
        else max(base_tokens + 1, int(window_budget * overflow_ratio))
    )
    noise_indexes = [index for index, turn in enumerate(template_turns) if turn["content"].startswith("{noise:")]
    min_noise_budget = 120 if target_total_tokens is not None else 600
    min_per_noise_tokens = 80 if target_total_tokens is not None else 250
    noise_budget = max(min_noise_budget, target_tokens - base_tokens)
    per_noise_tokens = max(min_per_noise_tokens, math.ceil(noise_budget / max(1, len(noise_indexes))))

    turns = []
    for index, turn in enumerate(template_turns):
        content = turn["content"]
        if content.startswith("{noise:"):
            match = re.match(r"\{noise:(\d+):(.+)\}", content)
            if match is None:
                raise ValueError(f"Invalid noise placeholder: {content}")
            dump_id = int(match.group(1))
            topic = match.group(2)
            content = _build_noise_message(
                backend=backend,
                model=model,
                dump_id=dump_id,
                target_tokens=per_noise_tokens,
                topic=topic,
                seed=seed + index,
            )
        turns.append({"role": turn["role"], "content": content})
    return turns


def _build_noise_message(
    *,
    backend: ContextBackend,
    model: str,
    dump_id: int,
    target_tokens: int,
    topic: str,
    seed: int,
) -> str:
    rng = Random(seed)
    header = (
        f"Unrelated {topic} dump {dump_id}. The identifiers below belong to a different workflow "
        "and should not overwrite the active case.\n"
    )
    header_tokens = backend.count_tokens([{"role": "tool", "content": header}], model=model)
    sample_count = 180
    sample_body = "\n".join(_noise_line(rng, dump_id, row, topic) for row in range(sample_count))
    sample_tokens = backend.count_tokens(
        [{"role": "tool", "content": header + sample_body}],
        model=model,
    )
    per_line_tokens = max(1.0, (sample_tokens - header_tokens) / sample_count)
    line_count = max(1, math.ceil((target_tokens - header_tokens) / per_line_tokens))

    while True:
        body = "\n".join(_noise_line(rng, dump_id, row, topic) for row in range(line_count))
        content = header + body
        current_tokens = backend.count_tokens([{"role": "tool", "content": content}], model=model)
        if current_tokens >= target_tokens:
            return content
        line_count += max(1, math.ceil((target_tokens - current_tokens) / per_line_tokens))


def _noise_line(rng: Random, dump_id: int, row: int, topic: str) -> str:
    service = rng.choice(["audit_queue", "retention_worker", "sync_gateway", "case_router"])
    module = rng.choice(["jobs/retry.py", "services/replicator.py", "ui/console.ts", "ops/backfill.py"])
    status = rng.choice(["queued", "retrying", "backoff", "healthy", "stalled"])
    return (
        f"[dump={dump_id} row={row}] topic={topic} svc={service} mod={module} "
        f"line={100 + (row % 500)} build=v9.{row % 7}.{row % 11} shard={row % 19} "
        f"latency_ms={40 + (row % 280)} heap={64 + (row % 17)}MB status={status}"
    )
