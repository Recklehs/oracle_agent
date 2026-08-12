import asyncio

import httpx
import pytest
from tenacity import RetryAction, RetryCallState, Retrying

from oracle_agent.agents import provider


def _http_status_error(response: httpx.Response) -> httpx.HTTPStatusError:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        return error
    raise AssertionError("HTTP 오류 응답이 필요합니다")


def _재시도_상태(response: httpx.Response) -> RetryCallState:
    state = RetryCallState(Retrying(), None, (), {})
    state.set_exception((httpx.HTTPStatusError, _http_status_error(response), None))
    state.next_action = RetryAction(1)
    return state


class TestProviderHttp재시도:
    def test_timeout과_지정된_http_상태만_재시도한다(self):
        timeout = httpx.ReadTimeout("timeout")
        rate_limit = httpx.Response(
            429,
            request=httpx.Request("GET", "https://api.openai.com"),
        )
        bad_request = httpx.Response(
            400,
            request=httpx.Request("GET", "https://api.openai.com"),
        )

        assert provider._is_retryable_http_error(timeout)
        assert provider._is_retryable_http_error(_http_status_error(rate_limit))
        assert not provider._is_retryable_http_error(_http_status_error(bad_request))

    def test_재시도_가능한_http_오류는_최초_포함_여섯번_시도한다(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(503, request=request)

        async def no_sleep(_seconds: float) -> None:
            return None

        async def request() -> None:
            transport = provider.retrying_transport(httpx.MockTransport(handler))
            transport.config["sleep"] = no_sleep
            async with httpx.AsyncClient(transport=transport) as client:
                with pytest.raises(httpx.HTTPStatusError):
                    await client.get("https://api.openai.com/test")

        asyncio.run(request())

        assert attempts == 6

    def test_재시도_대상이_아닌_http_오류는_응답을_그대로_통과시킨다(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(401, request=request)

        async def request() -> httpx.Response:
            transport = provider.retrying_transport(httpx.MockTransport(handler))
            async with httpx.AsyncClient(transport=transport) as client:
                return await client.get("https://api.openai.com/test")

        response = asyncio.run(request())

        # 401 같은 오류는 transport가 삼키지 않아야 상위 SDK가 올바른 예외를 낸다.
        assert response.status_code == 401
        assert attempts == 1

    def test_provider_재시도전에_상태와_대기시간을_기록한다(self, caplog):
        state = _재시도_상태(
            httpx.Response(
                429,
                headers={"retry-after": "20"},
                request=httpx.Request("POST", "https://api.openai.com"),
            )
        )

        provider._log_provider_retry(state)

        assert "429" in caplog.text
        assert "20" in caplog.text
