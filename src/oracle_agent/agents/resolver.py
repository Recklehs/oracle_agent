"""조사 결과 번들을 검토해 최종 판정하는 판정 Agent와 resolve() orchestration."""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import SelectModel
from pydantic_ai.models.openai import OpenAIResponsesModelSettings
from pydantic_ai.usage import UsageLimits

from oracle_agent.agents.provider import production_model
from oracle_agent.agents.search_backends import SearchBackend, create_search_backend
from oracle_agent.agents.searcher import (
    MAX_OUTPUT_RETRIES,
    EvidenceBundle,
    EvidenceFitness,
    investigate,
    normalize_url,
)
from oracle_agent.models import InvestigationInput, InvestigationResult, NonEmptyText


logger = logging.getLogger(__name__)

JUDGE_INSTRUCTIONS = """
당신은 조사 Agent가 수집한 증거 번들로 prediction market의 YES/NO 종료 결과를 판정하는
판정 Agent다. 잘못된 자동 판정보다 ESCALATED를 우선한다.

판정 규칙:
- 증거를 새로 만들거나 고칠 수 없다. 번들의 evidence와 적합성 검토만 근거로 사용한다.
- YES 또는 NO 자동 판정에는 FINAL 적합성 증거만 사용한다. official FINAL 증거 하나 또는
  서로 독립적인 원출처의 high_trust FINAL 증거 둘이 필요하다.
- 같은 원출처의 재게시는 독립 증거로 중복 계산하지 않는다.
- 권위 있는(official, high_trust) FINAL·PRELIMINARY 증거가 YES와 NO로 충돌하면 ESCALATED다.
- 자기검토 필수 항목이 하나라도 거짓이거나 missing_research가 있으면 YES/NO를 제출하지 않는다.
- 종료 조건 해석이 모호하거나 증거가 부족하면 구체적인 escalation_reason과 함께 ESCALATED를
  제출한다.
- summary에는 판정과 그 근거를 요약한다.
""".strip()


@dataclass
class JudgeDeps:
    investigation: InvestigationInput
    bundle: EvidenceBundle


def finalize_decision(
    ctx: RunContext[JudgeDeps],
    decision: Literal["YES", "NO", "ESCALATED"],
    summary: NonEmptyText,
    escalation_reason: NonEmptyText | None = None,
) -> InvestigationResult:
    """판정을 조사 번들과 대조해 자동 판정 안전 조건을 검사한다."""
    investigation = ctx.deps.investigation
    bundle = ctx.deps.bundle

    def escalate(reason: str) -> InvestigationResult:
        return InvestigationResult(
            prediction_id=investigation.prediction_id,
            decision="ESCALATED",
            summary=summary,
            evidence=bundle.evidence,
            escalation_reason=reason,
        )

    def retry_or_escalate(reason: str) -> InvestigationResult:
        if ctx.retry < MAX_OUTPUT_RETRIES:
            raise ModelRetry(reason)
        return escalate(reason)

    fitness_by_url = {
        normalize_url(review.url): review.fitness
        for review in bundle.evidence_reviews
    }
    authoritative_directions = {
        item["supports"]
        for item in bundle.evidence
        if item["authority"] in {"official", "high_trust"}
        and item["supports"] != "INCONCLUSIVE"
        and fitness_by_url[normalize_url(item["url"])]
        in {EvidenceFitness.FINAL, EvidenceFitness.PRELIMINARY}
    }
    if authoritative_directions == {"YES", "NO"}:
        return escalate("권위 있는 출처가 YES와 NO로 충돌합니다.")

    if decision in {"YES", "NO"}:
        self_review = bundle.self_review
        if (
            not all(
                (
                    self_review.criteria_clear,
                    self_review.result_period_complete,
                    self_review.findings_match_sources,
                    self_review.duplicate_publishers_checked,
                    self_review.contradiction_search_complete,
                )
            )
            or self_review.missing_research
        ):
            return retry_or_escalate("YES/NO 자동 판정 전 자기검토를 모두 완료해야 합니다.")

        matching_evidence = [
            item
            for item in bundle.evidence
            if item["supports"] == decision
            and fitness_by_url[normalize_url(item["url"])] is EvidenceFitness.FINAL
        ]
        independent_publishers = {
            item["original_publisher"].casefold()
            for item in matching_evidence
            if item["authority"] in {"official", "high_trust"}
        }
        if (
            any(item["authority"] == "official" for item in matching_evidence)
            or len(independent_publishers) >= 2
        ):
            return InvestigationResult(
                prediction_id=investigation.prediction_id,
                decision=decision,
                summary=summary,
                evidence=bundle.evidence,
            )
        return retry_or_escalate(
            f"{decision} 자동 판정에는 FINAL 증거가 필요합니다. 권위 있는 독립 원출처는 "
            f"현재 {len(independent_publishers)}개이며 2개 필요합니다."
        )

    if escalation_reason is None:
        return retry_or_escalate("ESCALATED에는 구체적인 이관 사유가 필요합니다.")
    return escalate(escalation_reason)


_agent = Agent(
    model=None,
    output_type=finalize_decision,
    instructions=JUDGE_INSTRUCTIONS,
    deps_type=JudgeDeps,
    capabilities=[SelectModel(lambda _ctx: production_model())],
    retries={"output": MAX_OUTPUT_RETRIES},
    max_concurrency=2,
    model_settings=OpenAIResponsesModelSettings(openai_reasoning_effort="medium"),
)


@_agent.instructions
def _judge_context(ctx: RunContext[JudgeDeps]) -> str:
    investigation = ctx.deps.investigation.model_dump(
        mode="json", exclude={"prediction_id"}
    )
    bundle = ctx.deps.bundle.model_dump(mode="json", exclude={"gate_failure"})
    return (
        "현재 조사 입력:\n"
        + json.dumps(investigation, ensure_ascii=False, indent=2)
        + "\n\n조사 Agent의 증거 번들:\n"
        + json.dumps(bundle, ensure_ascii=False, indent=2)
    )


async def resolve(
    investigation: InvestigationInput,
    *,
    search_backend: SearchBackend | None = None,
) -> InvestigationResult:
    if investigation.resolve_after > datetime.now(UTC):
        return InvestigationResult(
            prediction_id=investigation.prediction_id,
            decision="ESCALATED",
            summary="아직 판정 가능한 시점이 되지 않았습니다.",
            evidence=[],
            escalation_reason=(
                f"판정 가능 시점은 {investigation.resolve_after.isoformat()} 이후입니다."
            ),
        )

    backend = search_backend or create_search_backend()
    bundle = await investigate(investigation, backend)
    if bundle.gate_failure:
        return InvestigationResult(
            prediction_id=investigation.prediction_id,
            decision="ESCALATED",
            summary=bundle.summary,
            evidence=bundle.evidence,
            escalation_reason=bundle.gate_failure,
        )

    result = await _agent.run(
        "조사 번들을 검토하고 최종 판정을 제출하세요.",
        deps=JudgeDeps(investigation=investigation, bundle=bundle),
        usage_limits=UsageLimits(request_limit=4, output_tokens_limit=8_000),
    )
    usage = result.usage
    logger.info(
        "agent usage requests=%s tool_calls=%s input_tokens=%s output_tokens=%s",
        usage.requests,
        usage.tool_calls,
        usage.input_tokens,
        usage.output_tokens,
    )
    return result.output
