"""모델 provider와 검색 backend가 공유하는 HTTP 재시도와 production 모델 구성."""

import logging
from functools import cache

import httpx
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from tenacity import RetryCallState, retry_if_exception, stop_after_attempt, wait_exponential


PRODUCTION_MODEL_NAME = "gpt-5.6-luna"
JUDGE_MODEL_NAME = "gpt-5.6-terra"
RETRYABLE_HTTP_STATUSES = {429, 502, 503, 504}
logger = logging.getLogger(__name__)


def _is_retryable_http_error(error: BaseException) -> bool:
    if isinstance(error, httpx.TransportError):
        return True
    return isinstance(error, httpx.HTTPStatusError) and (
        error.response.status_code in RETRYABLE_HTTP_STATUSES
    )


def _log_provider_retry(retry_state: RetryCallState) -> None:
    error = retry_state.outcome.exception() if retry_state.outcome else None
    response = error.response if isinstance(error, httpx.HTTPStatusError) else None
    logger.warning(
        "provider retry attempt=%s status=%s wait=%s retry_after=%s remaining_requests=%s remaining_tokens=%s",
        retry_state.attempt_number,
        response.status_code if response else None,
        retry_state.next_action.sleep if retry_state.next_action else None,
        response.headers.get("retry-after") if response else None,
        response.headers.get("x-ratelimit-remaining-requests") if response else None,
        response.headers.get("x-ratelimit-remaining-tokens") if response else None,
    )


def _raise_if_retryable_status(response: httpx.Response) -> None:
    # 재시도 대상 상태코드만 transport 예외로 바꾼다. 그 외(401 등)는 응답을 그대로
    # 통과시켜 상위 레이어(openai SDK 등)가 올바른 오류로 해석하게 한다.
    if response.status_code in RETRYABLE_HTTP_STATUSES:
        response.raise_for_status()


def retrying_transport(
    wrapped: httpx.AsyncBaseTransport | None = None,
) -> AsyncTenacityTransport:
    return AsyncTenacityTransport(
        RetryConfig(
            retry=retry_if_exception(_is_retryable_http_error),
            stop=stop_after_attempt(6),
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=2, max=60),
                max_wait=120,
            ),
            before_sleep=_log_provider_retry,
            reraise=True,
        ),
        wrapped=wrapped,
        validate_response=_raise_if_retryable_status,
    )


def retrying_async_client(timeout: float = 60) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=retrying_transport(), timeout=timeout)


def _build_model(model_name: str) -> OpenAIResponsesModel:
    openai_client = AsyncOpenAI(http_client=retrying_async_client(), max_retries=0)
    return OpenAIResponsesModel(
        model_name,
        provider=OpenAIProvider(openai_client=openai_client),
    )


@cache
def production_model() -> OpenAIResponsesModel:
    return _build_model(PRODUCTION_MODEL_NAME)


@cache
def judge_model() -> OpenAIResponsesModel:
    return _build_model(JUDGE_MODEL_NAME)


async def aclose_cached_models() -> None:
    """캐시된 모델의 HTTP 클라이언트를 닫고 캐시를 비운다."""
    for factory in (production_model, judge_model):
        if factory.cache_info().currsize:
            await factory().client.close()
        factory.cache_clear()
