"""조사 Agent가 갈아 끼울 수 있는 검색 backend 모듈.

모든 backend는 같은 `search()` 계약으로 후보 URL 목록만 반환한다. 검색 결과는
후보 선택에만 사용하고, 증거 인정은 항상 `web_fetch` 원문 조회로만 한다.
"""

import logging
import os
from collections.abc import Sequence
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
from openai import AsyncOpenAI
from pydantic import AnyHttpUrl, BaseModel, ValidationError

from oracle_agent.agents.provider import retrying_async_client, retrying_transport
from oracle_agent.models import NonEmptyText


MAX_RESULTS_PER_SEARCH = 8
DEFAULT_OPENAI_SEARCH_MODEL = "gpt-5.6-luna"
DEFAULT_GEMINI_SEARCH_MODEL = "gemini-3.5-flash-lite"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
EXA_SEARCH_URL = "https://api.exa.ai/search"
EXA_CONTENTS_URL = "https://api.exa.ai/contents"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    url: AnyHttpUrl
    title: NonEmptyText
    snippet: str | None = None
    published_at: str | None = None


class SearchBackend(Protocol):
    name: str

    async def search(
        self, query: str, target_domains: Sequence[str] = ()
    ) -> list[SearchResult]: ...


def _result(
    url: object,
    title: object = None,
    *,
    snippet: str | None = None,
    published_at: object = None,
) -> SearchResult | None:
    if not isinstance(url, str):
        return None
    hostname = urlsplit(url).hostname
    try:
        return SearchResult(
            url=url,
            title=title if isinstance(title, str) and title.strip() else hostname,
            snippet=snippet,
            published_at=published_at if isinstance(published_at, str) else None,
        )
    except ValidationError:
        return None


def _dedupe(results: list[SearchResult | None]) -> list[SearchResult]:
    unique: dict[str, SearchResult] = {}
    for result in results:
        if result is not None:
            unique.setdefault(str(result.url), result)
    return list(unique.values())[:MAX_RESULTS_PER_SEARCH]


class OpenAIWebSearchBackend:
    """OpenAI Responses API의 hosted web_search로 검색하는 기준선 backend."""

    name = "openai"

    def __init__(
        self,
        openai_client: AsyncOpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self._client = openai_client or AsyncOpenAI(
            http_client=retrying_async_client(), max_retries=0
        )
        self._model = model or os.environ.get(
            "ORACLE_OPENAI_SEARCH_MODEL", DEFAULT_OPENAI_SEARCH_MODEL
        )

    async def search(
        self, query: str, target_domains: Sequence[str] = ()
    ) -> list[SearchResult]:
        tool: dict[str, Any] = {"type": "web_search"}
        if target_domains:
            tool["filters"] = {"allowed_domains": list(target_domains)}
        response = await self._client.responses.create(
            model=self._model,
            input=(
                "다음 검색어로 웹을 검색하고 찾은 출처를 인용으로 나열하세요. "
                f"다른 설명은 쓰지 마세요: {query}"
            ),
            tools=[tool],
            tool_choice={"type": "web_search"},
            include=["web_search_call.action.sources"],
        )
        # 제목이 있는 인용을 먼저 수집해 같은 URL의 source가 제목을 덮지 않게 한다.
        results: list[SearchResult | None] = []
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for content in getattr(item, "content", None) or []:
                    for annotation in getattr(content, "annotations", None) or []:
                        if getattr(annotation, "type", None) == "url_citation":
                            results.append(
                                _result(
                                    getattr(annotation, "url", None),
                                    getattr(annotation, "title", None),
                                )
                            )
        for item in response.output:
            if getattr(item, "type", None) == "web_search_call":
                action = getattr(item, "action", None)
                for source in getattr(action, "sources", None) or []:
                    results.append(_result(getattr(source, "url", None)))
        return _dedupe(results)


class GeminiGroundingBackend:
    """Gemini google_search grounding으로 검색하고 redirect URL을 실제 URL로 푼다."""

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("gemini 검색 backend에는 GEMINI_API_KEY가 필요합니다.")
        self._model = model or os.environ.get(
            "ORACLE_GEMINI_SEARCH_MODEL", DEFAULT_GEMINI_SEARCH_MODEL
        )
        self._client = http_client or retrying_async_client()

    async def search(
        self, query: str, target_domains: Sequence[str] = ()
    ) -> list[SearchResult]:
        search_text = query
        if target_domains:
            sites = " OR ".join(f"site:{domain}" for domain in target_domains)
            search_text = f"{query} ({sites})"
        response = await self._client.post(
            f"{GEMINI_BASE_URL}/models/{self._model}:generateContent",
            headers={"x-goog-api-key": self._api_key},
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": (
                                    "웹을 검색해 다음 검색어의 출처를 찾으세요: "
                                    f"{search_text}"
                                )
                            }
                        ]
                    }
                ],
                "tools": [{"google_search": {}}],
            },
        )
        response.raise_for_status()
        candidates = response.json().get("candidates") or []
        metadata = (candidates[0].get("groundingMetadata") or {}) if candidates else {}
        results: list[SearchResult | None] = []
        for chunk in metadata.get("groundingChunks") or []:
            web = chunk.get("web") or {}
            resolved = await self._resolve_redirect(web.get("uri"))
            if resolved is None:
                logger.info("gemini redirect 해석 실패 uri=%s", web.get("uri"))
                continue
            results.append(_result(resolved, web.get("title")))
        return _dedupe(results)

    async def _resolve_redirect(self, uri: object) -> str | None:
        if not isinstance(uri, str):
            return None
        if urlsplit(uri).hostname != "vertexaisearch.cloud.google.com":
            return uri
        try:
            response = await self._client.get(uri, follow_redirects=False)
        except httpx.HTTPError:
            return None
        if response.is_redirect:
            return response.headers.get("location")
        return None


class ExaSearchBackend:
    """Exa /search REST API로 의미 기반 검색을 수행하는 backend."""

    name = "exa"

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("EXA_API_KEY")
        if not self._api_key:
            raise ValueError("exa 검색 backend에는 EXA_API_KEY가 필요합니다.")
        self._client = http_client or retrying_async_client()

    async def search(
        self, query: str, target_domains: Sequence[str] = ()
    ) -> list[SearchResult]:
        body: dict[str, Any] = {
            "query": query,
            "type": "auto",
            "numResults": MAX_RESULTS_PER_SEARCH,
        }
        if target_domains:
            body["includeDomains"] = list(target_domains)
        response = await self._client.post(
            EXA_SEARCH_URL,
            headers={"x-api-key": self._api_key},
            json=body,
        )
        response.raise_for_status()
        return _dedupe(
            [
                _result(
                    item.get("url"),
                    item.get("title"),
                    published_at=item.get("publishedDate"),
                )
                for item in response.json().get("results") or []
            ]
        )


async def fetch_exa_contents(
    url: str,
    *,
    api_key: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    max_content_length: int = 50_000,
) -> dict[str, str] | None:
    """Exa /contents로 JS 렌더링된 페이지 본문을 조회한다.

    EXA_API_KEY가 없거나 조회에 실패하면 None을 반환해 호출측이 기존 경로를 유지한다.
    """
    key = api_key or os.environ.get("EXA_API_KEY")
    if not key:
        return None
    body = {
        "urls": [url],
        "text": True,
        # 날짜에 민감한 판정에 오래된 캐시 본문이 섞이지 않게 항상 새로 크롤링한다.
        "livecrawl": "always",
        "livecrawlTimeout": 10_000,
    }
    client = http_client or retrying_async_client()
    try:
        response = await client.post(
            EXA_CONTENTS_URL,
            headers={"x-api-key": key},
            json=body,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        logger.warning("exa contents 조회 실패 url=%s error=%s", url, error)
        return None
    finally:
        if http_client is None:
            await client.aclose()
    results = response.json().get("results") or []
    item = results[0] if results else {}
    text = item.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    hostname = urlsplit(url).hostname or url
    title = item.get("title")
    return {
        "url": item.get("url") or url,
        "title": title if isinstance(title, str) and title.strip() else hostname,
        "content": text[:max_content_length],
    }


class TavilySearchBackend:
    """LLM Agent 조사용으로 설계된 Tavily Search REST API backend."""

    name = "tavily"

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self._api_key:
            raise ValueError("tavily 검색 backend에는 TAVILY_API_KEY가 필요합니다.")
        self._client = http_client or retrying_async_client()

    async def search(
        self, query: str, target_domains: Sequence[str] = ()
    ) -> list[SearchResult]:
        body: dict[str, Any] = {
            "query": query,
            "max_results": MAX_RESULTS_PER_SEARCH,
            "search_depth": "basic",
        }
        if target_domains:
            body["include_domains"] = list(target_domains)
        response = await self._client.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=body,
        )
        response.raise_for_status()
        return _dedupe(
            [
                _result(
                    item.get("url"),
                    item.get("title"),
                    snippet=None,
                    published_at=item.get("published_date"),
                )
                for item in response.json().get("results") or []
            ]
        )


class BraveSearchBackend:
    """독립 인덱스를 가진 Brave Search REST API backend."""

    name = "brave"

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("BRAVE_API_KEY")
        if not self._api_key:
            raise ValueError("brave 검색 backend에는 BRAVE_API_KEY가 필요합니다.")
        self._client = http_client or retrying_async_client()

    async def search(
        self, query: str, target_domains: Sequence[str] = ()
    ) -> list[SearchResult]:
        search_text = query
        if target_domains:
            sites = " OR ".join(f"site:{domain}" for domain in target_domains)
            search_text = f"{query} ({sites})"
        response = await self._client.get(
            BRAVE_SEARCH_URL,
            headers={
                "X-Subscription-Token": self._api_key,
                "Accept": "application/json",
            },
            params={"q": search_text, "count": MAX_RESULTS_PER_SEARCH},
        )
        response.raise_for_status()
        web = response.json().get("web") or {}
        return _dedupe(
            [
                _result(
                    item.get("url"),
                    item.get("title"),
                    snippet=None,
                    published_at=item.get("page_age"),
                )
                for item in web.get("results") or []
            ]
        )


BACKEND_FACTORIES: dict[str, type] = {
    OpenAIWebSearchBackend.name: OpenAIWebSearchBackend,
    GeminiGroundingBackend.name: GeminiGroundingBackend,
    ExaSearchBackend.name: ExaSearchBackend,
    TavilySearchBackend.name: TavilySearchBackend,
    # BRAVE_API_KEY 발급 실패로 임시 비활성. 키 확보 후 주석 해제.
    # BraveSearchBackend.name: BraveSearchBackend,
}


def create_search_backend(name: str | None = None) -> SearchBackend:
    chosen = (name or os.environ.get("ORACLE_SEARCH_BACKEND") or "openai").strip().lower()
    factory = BACKEND_FACTORIES.get(chosen)
    if factory is None:
        supported = ", ".join(sorted(BACKEND_FACTORIES))
        raise ValueError(f"지원하지 않는 검색 backend: {chosen} (사용 가능: {supported})")
    return factory()
