from typing import Literal
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl
from pydantic_ai import Agent, ModelRetry, RunContext

from oracle_agent.models import (
    Evidence,
    InvestigationInput,
    InvestigationResult,
    NonEmptyText,
)


MAX_OUTPUT_RETRIES = 3


def _normalize_url(value: AnyHttpUrl | str) -> tuple[str, str, str, str]:
    parsed = urlsplit(str(value))
    scheme = parsed.scheme.lower()
    port = parsed.port
    default_port = {"http": 80, "https": 443}.get(scheme)
    authority = parsed.hostname or ""
    if port is not None and port != default_port:
        authority = f"{authority}:{port}"
    return scheme, authority.casefold(), parsed.path.rstrip("/") or "/", parsed.query


def _retry_or_escalate(
    ctx: RunContext[InvestigationInput],
    summary: NonEmptyText,
    evidence: list[Evidence],
    reason: str,
) -> InvestigationResult:
    if ctx.retry < MAX_OUTPUT_RETRIES:
        raise ModelRetry(reason)
    return InvestigationResult(
        prediction_id=ctx.deps.prediction_id,
        decision="ESCALATED",
        summary=summary,
        evidence=evidence,
        escalation_reason=reason,
    )


def finalize_investigation(
    ctx: RunContext[InvestigationInput],
    decision: Literal["YES", "NO", "ESCALATED"],
    summary: NonEmptyText,
    evidence: list[Evidence],
    escalation_reason: NonEmptyText | None = None,
) -> InvestigationResult:
    """코드 소유 필드를 결합하고 자동 판정 안전 조건을 검사한다."""
    observed_urls = {_normalize_url(item["url"]) for item in evidence}
    missing_official_urls = [
        str(url)
        for url in ctx.deps.official_sources
        if _normalize_url(url) not in observed_urls
    ]
    if missing_official_urls:
        return _retry_or_escalate(
            ctx,
            summary,
            evidence,
            f"확인하지 않은 공식 URL: {', '.join(missing_official_urls)}",
        )

    authoritative_directions = {
        item["supports"]
        for item in evidence
        if item["authority"] in {"official", "high_trust"}
        and item["supports"] != "INCONCLUSIVE"
    }
    if authoritative_directions == {"YES", "NO"}:
        return InvestigationResult(
            prediction_id=ctx.deps.prediction_id,
            decision="ESCALATED",
            summary=summary,
            evidence=evidence,
            escalation_reason="권위 있는 출처가 YES와 NO로 충돌합니다.",
        )

    if decision in {"YES", "NO"}:
        matching_evidence = [item for item in evidence if item["supports"] == decision]
        if any(item["authority"] == "official" for item in matching_evidence):
            return InvestigationResult(
                prediction_id=ctx.deps.prediction_id,
                decision=decision,
                summary=summary,
                evidence=evidence,
            )

        independent_publishers = {
            item["original_publisher"].casefold()
            for item in matching_evidence
            if item["authority"] in {"official", "high_trust"}
        }
        if len(independent_publishers) >= 2:
            return InvestigationResult(
                prediction_id=ctx.deps.prediction_id,
                decision=decision,
                summary=summary,
                evidence=evidence,
            )

    reason = escalation_reason or "자동 판정에 필요한 독립적이고 권위 있는 증거가 부족합니다."
    return _retry_or_escalate(ctx, summary, evidence, reason)


_agent = Agent(
    model=None,
    output_type=finalize_investigation,
    deps_type=InvestigationInput,
    retries={"output": MAX_OUTPUT_RETRIES},
    max_concurrency=2,
)
