import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from oracle_agent.agents import search_backends
from oracle_agent.agents.search_backends import (
    BraveSearchBackend,
    ExaSearchBackend,
    GeminiGroundingBackend,
    OpenAIWebSearchBackend,
    TavilySearchBackend,
    create_search_backend,
    fetch_exa_contents,
)


class Test백엔드선택:
    def test_지원하지_않는_이름이면_거절한다(self):
        with pytest.raises(ValueError, match="지원하지 않는"):
            create_search_backend("unknown")

    def test_환경변수로_지정한_backend를_만든다(self, monkeypatch):
        monkeypatch.setenv("ORACLE_SEARCH_BACKEND", "exa")
        monkeypatch.setenv("EXA_API_KEY", "exa-key")

        backend = create_search_backend()

        assert isinstance(backend, ExaSearchBackend)

    def test_지정이_없으면_openai_기준선을_만든다(self, monkeypatch):
        monkeypatch.delenv("ORACLE_SEARCH_BACKEND", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        backend = create_search_backend()

        assert isinstance(backend, OpenAIWebSearchBackend)

    @pytest.mark.parametrize(
        ("backend_name", "env_key"),
        [
            ("gemini", "GEMINI_API_KEY"),
            ("exa", "EXA_API_KEY"),
            ("tavily", "TAVILY_API_KEY"),
            # brave는 registry에서 임시 비활성이라 키 검사 전에 거절된다.
            # ("brave", "BRAVE_API_KEY"),
        ],
        ids=["gemini", "exa", "tavily"],
    )
    def test_api_키가_없으면_구체적으로_실패한다(
        self, monkeypatch, backend_name: str, env_key: str
    ):
        monkeypatch.delenv(env_key, raising=False)

        with pytest.raises(ValueError, match=env_key):
            create_search_backend(backend_name)

    def test_임시_비활성인_brave는_거절한다(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "brave-key")

        with pytest.raises(ValueError, match="지원하지 않는"):
            create_search_backend("brave")


class TestHttp오류전파:
    # transport의 validate_response가 재시도 대상 상태코드만 던지도록 완화되었으므로,
    # backend가 직접 raise_for_status()로 오류 응답을 거부하는지 검증한다.
    @pytest.mark.parametrize(
        "factory",
        [GeminiGroundingBackend, ExaSearchBackend, TavilySearchBackend, BraveSearchBackend],
        ids=["gemini", "exa", "tavily", "brave"],
    )
    def test_http_오류_응답이면_상태_오류를_던진다(self, factory):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "unauthorized"})

        backend = factory(
            api_key="test-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(backend.search("사건 공식 결과"))


class TestExa검색:
    def test_요청을_보내고_결과를_중복_제거해_변환한다(self):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["key"] = request.headers["x-api-key"]
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://a.example/result",
                            "title": "결과 A",
                            "publishedDate": "2026-08-01",
                        },
                        {"url": "https://a.example/result", "title": "중복 결과"},
                        {"url": "https://b.example/result", "title": None},
                    ]
                },
            )

        backend = ExaSearchBackend(
            api_key="exa-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        results = asyncio.run(backend.search("사건 공식 결과", ["a.example"]))

        assert captured["url"] == search_backends.EXA_SEARCH_URL
        assert captured["key"] == "exa-key"
        assert captured["body"]["query"] == "사건 공식 결과"
        assert captured["body"]["includeDomains"] == ["a.example"]
        assert [str(result.url) for result in results] == [
            "https://a.example/result",
            "https://b.example/result",
        ]
        assert results[0].title == "결과 A"
        assert results[0].published_at == "2026-08-01"
        assert results[1].title == "b.example"


class TestExa본문조회:
    def _client(self, handler) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def test_렌더링된_본문이_있으면_url_title_content를_반환한다(self):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["key"] = request.headers["x-api-key"]
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://a.example/page",
                            "title": "렌더링된 제목",
                            "text": "렌더링된 본문",
                        }
                    ]
                },
            )

        result = asyncio.run(
            fetch_exa_contents(
                "https://a.example/page",
                api_key="exa-key",
                http_client=self._client(handler),
            )
        )

        assert captured["url"] == search_backends.EXA_CONTENTS_URL
        assert captured["key"] == "exa-key"
        assert captured["body"] == {
            "urls": ["https://a.example/page"],
            "text": True,
            "livecrawl": "always",
            "livecrawlTimeout": 10_000,
        }
        assert result == {
            "url": "https://a.example/page",
            "title": "렌더링된 제목",
            "content": "렌더링된 본문",
        }

    def test_본문이_최대_길이를_넘으면_잘라서_반환한다(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"results": [{"url": "https://a.example/page", "text": "가나다라마"}]},
            )

        result = asyncio.run(
            fetch_exa_contents(
                "https://a.example/page",
                api_key="exa-key",
                http_client=self._client(handler),
                max_content_length=3,
            )
        )

        assert result is not None
        assert result["content"] == "가나다"

    def test_api_key가_없으면_none을_반환한다(self, monkeypatch):
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, json={"results": []})

        result = asyncio.run(
            fetch_exa_contents("https://a.example/page", http_client=self._client(handler))
        )

        assert result is None
        assert calls == []

    def test_결과가_비어_있으면_none을_반환한다(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": []})

        result = asyncio.run(
            fetch_exa_contents(
                "https://a.example/page",
                api_key="exa-key",
                http_client=self._client(handler),
            )
        )

        assert result is None

    def test_http_오류이면_none을_반환한다(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "server"})

        result = asyncio.run(
            fetch_exa_contents(
                "https://a.example/page",
                api_key="exa-key",
                http_client=self._client(handler),
            )
        )

        assert result is None


class TestTavily검색:
    def test_요청을_보내고_결과를_변환한다(self):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers["Authorization"]
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://a.example/result",
                            "title": "결과 A",
                            "content": "요약문",
                            "published_date": "2026-08-02",
                        },
                        {"url": "https://b.example/result", "title": ""},
                    ]
                },
            )

        backend = TavilySearchBackend(
            api_key="tavily-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        results = asyncio.run(backend.search("사건 공식 결과", ["a.example"]))

        assert captured["url"] == search_backends.TAVILY_SEARCH_URL
        assert captured["auth"] == "Bearer tavily-key"
        assert captured["body"]["query"] == "사건 공식 결과"
        assert captured["body"]["include_domains"] == ["a.example"]
        assert [str(result.url) for result in results] == [
            "https://a.example/result",
            "https://b.example/result",
        ]
        assert results[0].title == "결과 A"
        assert results[0].published_at == "2026-08-02"
        assert results[1].title == "b.example"

    def test_도메인_제한이_없으면_include_domains를_보내지_않는다(self):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"results": []})

        backend = TavilySearchBackend(
            api_key="tavily-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        results = asyncio.run(backend.search("사건 공식 결과"))

        assert results == []
        assert "include_domains" not in captured["body"]


class TestBrave검색:
    def test_요청을_보내고_결과를_변환한다(self):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url.copy_with(query=None))
            captured["query"] = request.url.params["q"]
            captured["count"] = request.url.params["count"]
            captured["token"] = request.headers["X-Subscription-Token"]
            return httpx.Response(
                200,
                json={
                    "web": {
                        "results": [
                            {
                                "url": "https://a.example/result",
                                "title": "결과 A",
                                "page_age": "2026-08-02T00:00:00",
                            },
                            {"url": "https://a.example/result", "title": "중복"},
                            {"url": "https://b.example/result"},
                        ]
                    }
                },
            )

        backend = BraveSearchBackend(
            api_key="brave-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        results = asyncio.run(backend.search("사건 공식 결과", ["a.example"]))

        assert captured["url"] == search_backends.BRAVE_SEARCH_URL
        assert captured["token"] == "brave-key"
        assert captured["query"] == "사건 공식 결과 (site:a.example)"
        assert captured["count"] == str(search_backends.MAX_RESULTS_PER_SEARCH)
        assert [str(result.url) for result in results] == [
            "https://a.example/result",
            "https://b.example/result",
        ]
        assert results[0].title == "결과 A"
        assert results[0].published_at == "2026-08-02T00:00:00"
        assert results[1].title == "b.example"

    def test_web_결과가_없으면_빈_목록을_반환한다(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        backend = BraveSearchBackend(
            api_key="brave-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        results = asyncio.run(backend.search("사건 공식 결과"))

        assert results == []


class TestGemini검색:
    def _handler(self, captured: dict[str, object]) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "generateContent" in url:
                captured["key"] = request.headers["x-goog-api-key"]
                body = json.loads(request.content)
                captured["text"] = body["contents"][0]["parts"][0]["text"]
                captured["tools"] = body["tools"]
                return httpx.Response(
                    200,
                    json={
                        "candidates": [
                            {
                                "groundingMetadata": {
                                    "groundingChunks": [
                                        {
                                            "web": {
                                                "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/ok",
                                                "title": "a.example",
                                            }
                                        },
                                        {
                                            "web": {
                                                "uri": "https://direct.example/page",
                                                "title": "직접 결과",
                                            }
                                        },
                                        {
                                            "web": {
                                                "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/broken"
                                            }
                                        },
                                    ]
                                }
                            }
                        ]
                    },
                )
            if url.endswith("/ok"):
                return httpx.Response(
                    302, headers={"location": "https://real.example/result"}
                )
            return httpx.Response(404)

        return httpx.MockTransport(handler)

    def test_grounding_결과의_redirect를_실제_url로_풀고_실패한_결과는_버린다(self):
        captured: dict[str, object] = {}
        backend = GeminiGroundingBackend(
            api_key="gemini-key",
            http_client=httpx.AsyncClient(transport=self._handler(captured)),
        )

        results = asyncio.run(backend.search("사건 공식 결과", ["a.example"]))

        assert captured["key"] == "gemini-key"
        assert captured["tools"] == [{"google_search": {}}]
        assert "site:a.example" in str(captured["text"])
        assert [str(result.url) for result in results] == [
            "https://real.example/result",
            "https://direct.example/page",
        ]
        assert results[0].title == "a.example"

    def test_grounding_metadata가_없으면_빈_결과를_반환한다(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"candidates": [{}]})

        backend = GeminiGroundingBackend(
            api_key="gemini-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        results = asyncio.run(backend.search("사건 공식 결과"))

        assert results == []


class TestOpenAI검색:
    def _backend(self, output: list) -> tuple[OpenAIWebSearchBackend, list[dict]]:
        calls: list[dict] = []

        class _Responses:
            async def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(output=output)

        client = SimpleNamespace(responses=_Responses())
        return OpenAIWebSearchBackend(openai_client=client, model="test-search"), calls

    def test_인용과_source를_합쳐_결과로_변환한다(self):
        output = [
            SimpleNamespace(
                type="web_search_call",
                action=SimpleNamespace(
                    sources=[
                        SimpleNamespace(url="https://cited.example/result"),
                        SimpleNamespace(url="https://extra.example/result"),
                    ]
                ),
            ),
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(
                        annotations=[
                            SimpleNamespace(
                                type="url_citation",
                                url="https://cited.example/result",
                                title="인용 제목",
                            )
                        ]
                    )
                ],
            ),
        ]
        backend, calls = self._backend(output)

        results = asyncio.run(backend.search("사건 공식 결과", ["example.com"]))

        assert calls[0]["model"] == "test-search"
        assert calls[0]["tools"] == [
            {"type": "web_search", "filters": {"allowed_domains": ["example.com"]}}
        ]
        assert calls[0]["tool_choice"] == {"type": "web_search"}
        assert calls[0]["include"] == ["web_search_call.action.sources"]
        assert [str(result.url) for result in results] == [
            "https://cited.example/result",
            "https://extra.example/result",
        ]
        assert results[0].title == "인용 제목"
        assert results[1].title == "extra.example"

    def test_도메인_제한이_없으면_filters를_보내지_않는다(self):
        backend, calls = self._backend([])

        results = asyncio.run(backend.search("사건 공식 결과"))

        assert results == []
        assert calls[0]["tools"] == [{"type": "web_search"}]
