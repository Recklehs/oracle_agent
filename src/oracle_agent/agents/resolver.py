"""조사 번들을 검토해 최종 판정하는 판정 Agent와 resolve() orchestration.

판정 Agent는 검증 규칙을 포함한 단일 프롬프트로 조사 번들을 검토하고,
pydantic_ai의 native structured output으로 `InvestigationResult` 양식에 맞춰
결과를 반환한다.
"""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic_ai import Agent, NativeOutput, RunContext
from pydantic_ai.capabilities import SelectModel
from pydantic_ai.models.openai import OpenAIResponsesModelSettings
from pydantic_ai.usage import UsageLimits

from oracle_agent.agents.provider import judge_model
from oracle_agent.agents.search_backends import SearchBackend, create_search_backend
from oracle_agent.agents.searcher import (
    MAX_OUTPUT_RETRIES,
    EvidenceBundle,
    investigate,
)
from oracle_agent.models import InvestigationInput, InvestigationResult


logger = logging.getLogger(__name__)

JUDGE_INSTRUCTIONS = """
당신은 조사 Agent가 수집한 증거 번들로 prediction market의 YES/NO 종료 결과를 판정하는
판정 Agent다.

조사 번들의 증거, 적합성 검토, 자기검토를 종합적으로 검토해 prediction의 종료 결과를
어떻게 판정하는 것이 옳을지 스스로 판단한다. 종료 조건 해석이 애매하거나, 증거가
미흡하거나 서로 상충하거나, 조사가 불완전하다고 판단되면 부족한 부분을 구체적으로
설명하는 escalation_reason과 함께 ESCALATED를 제출한다. 잘못된 자동 판정보다
ESCALATED를 우선한다.

출력 양식:
- 증거를 새로 만들거나 고칠 수 없다. 결과의 evidence에는 번들의 evidence를 순서와 내용
  그대로 복사하고, prediction_id에는 조사 입력의 prediction_id를 그대로 복사한다.
- YES/NO 판정에는 escalation_reason을 넣지 않고, ESCALATED에는 반드시 넣는다.
- summary에는 판정과 그 근거를 요약한다.
""".strip()


@dataclass
class JudgeDeps:
    investigation: InvestigationInput
    bundle: EvidenceBundle


_agent = Agent(
    model=None,
    output_type=NativeOutput(InvestigationResult),
    instructions=JUDGE_INSTRUCTIONS,
    deps_type=JudgeDeps,
    capabilities=[SelectModel(lambda _ctx: judge_model())],
    retries={"output": MAX_OUTPUT_RETRIES},
    max_concurrency=2,
    model_settings=OpenAIResponsesModelSettings(openai_reasoning_effort="medium"),
)


@_agent.instructions
def _judge_context(ctx: RunContext[JudgeDeps]) -> str:
    # resolve_after는 resolve()의 시간 게이트 전용이라 판정 컨텍스트에서 제외한다.
    investigation = ctx.deps.investigation.model_dump(
        mode="json", exclude={"resolve_after"}
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
        "조사 번들을 검증 규칙에 따라 검토하고 최종 판정을 제출하세요.",
        deps=JudgeDeps(investigation=investigation, bundle=bundle),
        usage_limits=UsageLimits(request_limit=4, output_tokens_limit=16_000),
    )
    usage = result.usage
    logger.info(
        "agent usage requests=%s tool_calls=%s input_tokens=%s output_tokens=%s",
        usage.requests,
        usage.tool_calls,
        usage.input_tokens,
        usage.output_tokens,
    )
    return result.output.model_copy(
        update={"prediction_id": investigation.prediction_id}
    )
