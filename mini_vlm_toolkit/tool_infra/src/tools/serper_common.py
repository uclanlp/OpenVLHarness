"""Shared infrastructure for Serper-based tools.

Provides:
- ``SerperClient`` — thin wrapper for Serper API calls (auth, retry, error handling).
- ``get_serper_api_key`` — load key from env / file.
- ``format_search_results`` — unified result formatter for all Serper text-search endpoints.
- ``strip_think_tags`` — remove ``<think>...</think>`` blocks from LLM output.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM output cleaning
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output (e.g. Qwen3 reasoning traces)."""
    return _THINK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Serper API endpoints
# ---------------------------------------------------------------------------
SERPER_SEARCH_BASE = "https://google.serper.dev"
SERPER_SCRAPE_BASE = "https://scrape.serper.dev"

ALL_SEARCH_TYPES = [
    "search", "images", "news", "scholar",
    "maps", "places", "shopping", "patents",
]

# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------
_SERPER_KEY_ENV = "SERPER_API_KEY"
# Fallback: look for a key file next to any of the Serper tool directories.
_KEY_FILE_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "serper_text_search", "serper_api_key"),
    os.path.join(os.path.dirname(__file__), "serper_image_search", "serper_api_key"),
    os.path.join(os.path.dirname(__file__), "serper_webpage_visit", "serper_api_key"),
    os.path.join(os.path.dirname(__file__), "serper_google_search", "serper_api_key"),
]

_cached_api_key: Optional[str] = None


def get_serper_api_key() -> str:
    """Load Serper API key from env var or local key file."""
    global _cached_api_key
    if _cached_api_key:
        return _cached_api_key

    env_key = os.getenv(_SERPER_KEY_ENV, "").strip()
    if env_key:
        _cached_api_key = env_key
        return env_key

    for path in _KEY_FILE_CANDIDATES:
        resolved = os.path.normpath(path)
        if os.path.exists(resolved):
            with open(resolved, "r") as f:
                key = f.read().strip()
            if key:
                _cached_api_key = key
                return key

    raise FileNotFoundError(
        f"Missing Serper API key. Set {_SERPER_KEY_ENV} env var, "
        f"or create a serper_api_key file in one of the Serper tool directories."
    )


# ---------------------------------------------------------------------------
# SerperClient
# ---------------------------------------------------------------------------

# HTTP error codes treated as transient (worth retrying with backoff).
# Anything outside this set is treated as permanent (fail fast).
_TRANSIENT_HTTP_CODES = frozenset({500, 502, 503, 504, 429})
_SERPER_MAX_RETRIES = 3
_SERPER_BACKOFF_BASE_SEC = 1.0
_SERPER_BACKOFF_MAX_SEC = 4.0


class SerperRequestError(RuntimeError):
    """Structured failure for a single Serper API call.

    Carries enough metadata for the caller (a Search / Image_Search /
    Webpage_Visit tool) to render an agent-readable error message with
    transient-vs-permanent guidance + cross-tool recovery suggestions.

    Attributes:
        status_code: HTTP status code if the failure was an HTTP error
            response (None for network / timeout / parse errors).
        transient: Whether the failure is expected to recover on retry
            (True for 5xx / 429 / network / timeout, False for 4xx).
        endpoint: Which Serper endpoint failed (e.g. ``search``, ``lens``,
            ``scrape``). Used in the rendered agent message.
        attempts: How many attempts were made before giving up. Useful
            so the agent doesn't unnecessarily retry a fully-saturated
            endpoint immediately.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        transient: bool = False,
        endpoint: str = "",
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.transient = transient
        self.endpoint = endpoint
        self.attempts = attempts


def _classify_http_error(exc: requests.HTTPError) -> Tuple[Optional[int], bool]:
    """Return (status_code, is_transient) for an HTTPError raised by raise_for_status."""
    code = None
    try:
        if exc.response is not None:
            code = int(exc.response.status_code)
    except Exception:
        code = None
    if code is None:
        # Network-level error masquerading as HTTPError; treat as transient.
        return None, True
    return code, code in _TRANSIENT_HTTP_CODES


def _call_serper_with_retry(
    *,
    endpoint_label: str,
    do_request: "callable",
    max_retries: int = _SERPER_MAX_RETRIES,
) -> Dict[str, Any]:
    """Call a Serper endpoint with retry on transient failures.

    ``do_request`` must be a zero-arg callable that performs the HTTP
    request and returns a parsed JSON dict. It is expected to:
        * raise ``requests.HTTPError`` on a non-2xx HTTP response
        * raise ``requests.Timeout`` / ``requests.ConnectionError`` on
          network failure
        * raise ``ValueError`` / ``json.JSONDecodeError`` on body parse
          failure (we treat these as permanent)

    Transient failures (5xx / 429 / network / timeout) are retried up to
    ``max_retries`` times with exponential backoff (1s, 2s, ...).
    Permanent failures (4xx) fail fast on the first attempt.

    On final failure, raises ``SerperRequestError`` with structured
    metadata (status code, transient flag, endpoint label, attempt count)
    so the caller can format an agent-readable message with recovery
    guidance.
    """
    import time
    last_code: Optional[int] = None
    last_is_transient = False
    last_error: str = ""
    for attempt in range(1, max_retries + 1):
        try:
            return do_request()
        except requests.HTTPError as exc:
            code, transient = _classify_http_error(exc)
            last_code, last_is_transient = code, transient
            last_error = f"HTTP {code} from {endpoint_label}" if code else f"HTTP error: {exc}"
        except requests.Timeout as exc:
            last_code = None
            last_is_transient = True
            last_error = f"network timeout calling {endpoint_label}: {exc}"
        except requests.ConnectionError as exc:
            last_code = None
            last_is_transient = True
            last_error = f"connection error calling {endpoint_label}: {exc}"
        except (ValueError, KeyError) as exc:
            # Body parse failure / unexpected response shape — permanent.
            raise SerperRequestError(
                f"Serper {endpoint_label} returned an unparseable response: {exc}",
                status_code=None,
                transient=False,
                endpoint=endpoint_label,
                attempts=attempt,
            ) from exc

        logger.warning(
            "Serper %s call failed (attempt %d/%d, transient=%s): %s",
            endpoint_label, attempt, max_retries, last_is_transient, last_error,
        )
        if not last_is_transient or attempt >= max_retries:
            break
        # Exponential backoff with cap.
        backoff = min(_SERPER_BACKOFF_BASE_SEC * (2 ** (attempt - 1)), _SERPER_BACKOFF_MAX_SEC)
        time.sleep(backoff)

    raise SerperRequestError(
        f"Serper {endpoint_label} failed after {attempt} attempt(s): {last_error}",
        status_code=last_code,
        transient=last_is_transient,
        endpoint=endpoint_label,
        attempts=attempt,
    )


# ---------------------------------------------------------------------------
# Parallel image fetch (used by multimodal search summarizer)
# ---------------------------------------------------------------------------

_IMAGE_FETCH_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _encode_image_bytes_to_data_url(
    data: bytes, *, content_type_hint: str, max_dim: int, jpeg_quality: int,
) -> Optional[str]:
    """Decode image bytes (via PIL), resize if either dim exceeds ``max_dim``,
    re-encode as JPEG, base64-encode into a ``data:image/jpeg;base64,...`` URL.

    Returns ``None`` on decode / encode failure.
    """
    import base64
    import io
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return None
    if img.mode not in ("RGB", "L"):
        try:
            img = img.convert("RGB")
        except Exception:
            return None
    w, h = img.size
    longest = max(w, h)
    if longest > max_dim:
        scale = max_dim / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        try:
            img = img.resize(new_size, Image.LANCZOS)
        except Exception:
            return None
    out = io.BytesIO()
    try:
        img.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
    except Exception:
        return None
    b64 = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def fetch_images_parallel(
    urls: List[str],
    *,
    total_timeout: float = 12.0,
    per_image_timeout: float = 4.0,
    max_bytes: int = 2_000_000,
    max_dim: int = 1024,
    jpeg_quality: int = 85,
    max_workers: int = 8,
) -> Dict[str, Optional[str]]:
    """Fetch a list of HTTP(S) image URLs in parallel, return ``{url: data_url}``.

    Used by the multimodal summarizer in Text_Search / Image_Search to bring
    image bytes into the summarizer's input. Per-URL failures degrade
    gracefully (entry value is ``None``) so the summarizer can still rerank
    on text alone for that result.

    Args:
        urls: HTTP(S) URLs of candidate images. Deduplicated internally.
        total_timeout: Hard ceiling for the entire batch. Any fetch still
            outstanding after this is recorded as ``None`` and we move on.
        per_image_timeout: Per-request connect + read timeout.
        max_bytes: Reject responses whose ``Content-Length`` exceeds this,
            or whose body exceeds this during streaming read.
        max_dim: Resize so neither side exceeds this. 1024 is a good
            cap for vision-LM input (matches typical model patch grid).
        jpeg_quality: JPEG re-encode quality (0-95). 85 is visually lossless
            for typical thumbnails and keeps the base64 ~30-100KB.
        max_workers: Thread pool size.

    Returns:
        ``{url: data_url_or_None}`` — preserves input order is NOT guaranteed
        for the dict, but every URL passed in is a key.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    out: Dict[str, Optional[str]] = {}
    seen: set = set()
    deduped: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u_clean = u.strip()
        if not u_clean or u_clean in seen:
            continue
        if not u_clean.lower().startswith(("http://", "https://")):
            continue
        seen.add(u_clean)
        deduped.append(u_clean)
        out[u_clean] = None

    if not deduped:
        return out

    headers = {"User-Agent": _IMAGE_FETCH_USER_AGENT, "Accept": "image/*"}

    def _fetch_one(url: str) -> Optional[str]:
        try:
            resp = requests.get(
                url, headers=headers, timeout=per_image_timeout, stream=True,
            )
            resp.raise_for_status()
            content_type = (resp.headers.get("Content-Type") or "").lower()
            content_length = resp.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        return None
                except ValueError:
                    pass
            body = bytearray()
            for chunk in resp.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                body.extend(chunk)
                if len(body) > max_bytes:
                    return None
            return _encode_image_bytes_to_data_url(
                bytes(body),
                content_type_hint=content_type,
                max_dim=max_dim,
                jpeg_quality=jpeg_quality,
            )
        except Exception:
            return None

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(max_workers, len(deduped))) as pool:
        future_to_url = {pool.submit(_fetch_one, u): u for u in deduped}
        try:
            for fut in as_completed(future_to_url, timeout=total_timeout):
                u = future_to_url[fut]
                try:
                    out[u] = fut.result()
                except Exception:
                    out[u] = None
        except Exception:
            # ``as_completed`` raises TimeoutError once total_timeout is hit;
            # leave any still-pending entries as ``None`` and proceed.
            pass
        # Best-effort cancel of any still-pending futures so the pool can
        # close quickly. Python doesn't kill in-flight requests, but the
        # per-image timeout caps their lifetime.
        for fut in future_to_url:
            if not fut.done():
                fut.cancel()
    elapsed = time.monotonic() - started
    logger.info(
        "fetch_images_parallel: %d urls, %d fetched, %d failed, %.2fs",
        len(deduped),
        sum(1 for v in out.values() if v),
        sum(1 for v in out.values() if v is None),
        elapsed,
    )
    return out


class SerperClient:
    """Thin HTTP client for Serper API calls.

    All three endpoints (``search``, ``lens``, ``scrape``) wrap their
    network call in ``_call_serper_with_retry``, so transient outages
    (5xx / 429 / network / timeout) are retried up to 3 times with
    exponential backoff before raising ``SerperRequestError``. Callers
    should catch ``SerperRequestError`` and render a structured,
    agent-readable failure message with cross-tool recovery hints — see
    each tool's ``_format_serper_error`` helper.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self._api_key = api_key
        self.timeout = timeout

    @property
    def api_key(self) -> str:
        if not self._api_key:
            self._api_key = get_serper_api_key()
        return self._api_key

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def search(
        self,
        query: str,
        search_type: str = "search",
        num: int = 10,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call a Serper text-search endpoint.

        Args:
            query: Search query.
            search_type: One of ALL_SEARCH_TYPES.
            num: Number of results.
            gl: Country code (e.g. "us").
            hl: Language code (e.g. "en").

        Returns:
            Raw JSON response dict from Serper.

        Raises:
            ValueError: invalid ``search_type``.
            SerperRequestError: network / HTTP failure after retries.
        """
        if search_type not in ALL_SEARCH_TYPES:
            raise ValueError(f"Invalid search_type: {search_type!r}. Choose from {ALL_SEARCH_TYPES}")

        url = f"{SERPER_SEARCH_BASE}/{search_type}"
        payload: Dict[str, Any] = {"q": query, "num": num}
        if gl:
            payload["gl"] = gl
        if hl:
            payload["hl"] = hl

        def _do():
            response = requests.post(url, headers=self._headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        return _call_serper_with_retry(endpoint_label=f"search({search_type})", do_request=_do)

    def lens(self, image_url: str) -> Dict[str, Any]:
        """Call Serper Google Lens (reverse image search).

        Args:
            image_url: Public URL of the image.

        Returns:
            Raw JSON response dict.

        Raises:
            SerperRequestError: network / HTTP failure after retries.
        """
        url = f"{SERPER_SEARCH_BASE}/lens"
        payload = {"url": image_url}

        def _do():
            response = requests.post(url, headers=self._headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()

        return _call_serper_with_retry(endpoint_label="lens", do_request=_do)

    def scrape(self, page_url: str) -> Dict[str, Any]:
        """Call Serper webpage scrape endpoint.

        Args:
            page_url: URL to scrape.

        Returns:
            Dict with keys 'text', 'metadata', optionally 'jsonld', 'credits'.
        """
        payload = {"url": page_url}
        response = requests.post(
            SERPER_SCRAPE_BASE, headers=self._headers, json=payload, timeout=60,
        )
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Result formatters
# ---------------------------------------------------------------------------
def _http_image_url(url: object) -> str:
    """Return ``url`` only if it is a non-empty http(s) URL.

    Some Serper endpoints (notably /shopping, ~75% of items) inline
    images as ``data:image/...;base64,...`` URLs. Emitting those into
    the agent text bloats context massively (tens of KB per item) and
    they are not registerable by ImageStateMemory's URL rewriter
    (which only matches ``https?://``). Skip non-http URLs so the
    formatter neither emits a useless ``Image:`` line nor leaks the
    inline base64 blob into the agent's context.
    """
    s = str(url or "").strip()
    return s if s.lower().startswith(("http://", "https://")) else ""


def _display_domain(item: Dict[str, Any], link: str) -> str:
    """Return a short domain label for a search result.

    Prefers Serper's pre-cleaned ``displayLink`` (e.g. ``www.reddit.com``) and
    falls back to ``urlparse(link).netloc``. Strips the leading ``www.``.
    """
    display = item.get("displayLink")
    if isinstance(display, str) and display.strip():
        candidate = display.strip()
    else:
        try:
            candidate = urlparse(str(link or "")).netloc or ""
        except Exception:
            candidate = ""
    candidate = candidate.lower()
    if candidate.startswith("www."):
        candidate = candidate[4:]
    return candidate


def _display_url_compact(item: Dict[str, Any], link: str, max_len: int = 80) -> str:
    """Return ``domain/slug-bearing path`` with scheme / ``www.`` / query stripped.

    Keeps the URL's path (which often carries the answer entity — e.g. the
    Wikipedia slug ``/wiki/Apple_Inc.`` or the news article title slug) so
    the model can use it as a direct semantic cue, while still being shorter
    than the full ``https://...`` URL.

    If the resulting string exceeds ``max_len`` chars (long Reddit / FB /
    deep query URLs), middle-truncate to keep the informative tail.
    """
    if not isinstance(link, str) or not link.strip():
        return _display_domain(item, link)
    try:
        parsed = urlparse(link)
    except Exception:
        return _display_domain(item, link)
    host = (parsed.netloc or "").lower().removeprefix("www.")
    path = parsed.path or ""
    compact = (host + path).rstrip("/")
    if not compact:
        return _display_domain(item, link)
    if len(compact) <= max_len:
        return compact
    # Long URL: keep host and the last segment (usually the slug carrying
    # the entity name), with "..." in between.
    tail_seg = path.rstrip("/").rsplit("/", 1)[-1] if path else ""
    if tail_seg and (len(host) + 4 + len(tail_seg)) <= max_len + 10:
        return f"{host}/.../{tail_seg}"
    # Fallback: straight head truncation.
    return compact[: max_len - 3] + "..."


def format_search_results(search_type: str, data: Dict[str, Any], max_results: int = 10) -> str:
    """Format raw Serper response into agent-readable text."""
    formatter = _FORMATTERS.get(search_type, _format_generic)
    return formatter(data, max_results)


def _format_organic(data: Dict[str, Any], max_results: int) -> str:
    """Format /search, /scholar, /patents responses (organic key).

    Also surfaces image URLs from auxiliary sections that some queries
    return alongside organic results — ``topStories`` (breaking-news
    queries on /search) and per-item ``thumbnailUrl`` (every /patents
    item, the patent's main figure). Without this, a "Apple earnings"
    search would silently lose all topStories thumbnails, and patent
    queries would lose every patent diagram.
    """
    lines = []

    # Knowledge graph (if present)
    kg = data.get("knowledgeGraph")
    if kg:
        lines.append(f"[Knowledge Graph] {kg.get('title', '')}: {kg.get('description', '')}")
        kg_image_url = _http_image_url(kg.get("imageUrl"))
        if kg_image_url:
            # ImageStateMemory routes URLs trailed by ` <img>` directly
            # into the image registry as ``tool_generated_image_N``.
            lines.append(f"  Image: {kg_image_url} <img>")
        for k, v in (kg.get("attributes") or {}).items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    # Answer box (if present)
    ab = data.get("answerBox")
    if ab:
        lines.append(f"[Answer Box] {ab.get('title', '')}")
        lines.append(ab.get("snippet", ""))
        lines.append("")

    # Top Stories (breaking-news subsection on /search; each item carries
    # an imageUrl). Emit BEFORE organic so freshness signals lead.
    top_stories = data.get("topStories", [])[:max_results]
    if top_stories:
        lines.append("[Top Stories]")
        for i, item in enumerate(top_stories, 1):
            title = item.get("title", "")
            link = item.get("link", "")
            source = item.get("source", "")
            date = item.get("date", "")
            img = _http_image_url(item.get("imageUrl")) or _http_image_url(item.get("thumbnailUrl"))
            lines.append(f"{i}. {title}")
            if link:
                meta_bits = " | ".join(filter(None, [source, date]))
                if meta_bits:
                    lines.append(f"   {meta_bits}   {link}")
                else:
                    lines.append(f"   URL: {link}")
            if img:
                lines.append(f"   Image: {img} <img>")
        lines.append("")

    items = data.get("organic", [])[:max_results]
    for i, item in enumerate(items, 1):
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        extra_parts = []
        if item.get("year"):
            extra_parts.append(f"Year: {item['year']}")
        extra = " | ".join(extra_parts)
        # Per-item image (patents' main figure thumbnailUrl; some /search
        # rich items also carry imageUrl). Patents' ``figures[]`` array can
        # contain 5-13 entries — intentionally emit only the main thumbnail
        # to avoid context bloat; agent can visit the patent for more.
        per_item_img = (
            _http_image_url(item.get("thumbnailUrl"))
            or _http_image_url(item.get("imageUrl"))
        )
        lines.append(f"{i}. {title}")
        # Emit ``domain/slug-bearing-path`` + full URL on one line. The path
        # portion is kept plainly (no scheme/query) so the slug remains visible
        # as a semantic cue (e.g. ``en.wikipedia.org/wiki/Apple_Inc.``), which
        # the agent can read directly without expanding the URL.
        url_compact = _display_url_compact(item, link)
        if url_compact and link:
            lines.append(f"   Source: {url_compact}   {link}")
        elif link:
            lines.append(f"   URL: {link}")
        if snippet:
            lines.append(f"   {snippet}")
        if extra:
            lines.append(f"   {extra}")
        if per_item_img:
            lines.append(f"   Image: {per_item_img} <img>")
    return "\n".join(lines)


def _format_images(data: Dict[str, Any], max_results: int) -> str:
    items = data.get("images", [])[:max_results]
    lines = []
    for i, item in enumerate(items, 1):
        title = item.get("title", "")
        # Prefer source-quality imageUrl; fall back to gstatic thumbnailUrl.
        # _http_image_url also filters out inline data: URLs.
        img_url = _http_image_url(item.get("imageUrl")) or _http_image_url(item.get("thumbnailUrl"))
        source = item.get("source", "")
        link = item.get("link", "")
        w = item.get("imageWidth", "")
        h = item.get("imageHeight", "")
        lines.append(f"{i}. {title}")
        # ImageStateMemory routes URLs trailed by ` <img>` directly
        # into the image registry as ``tool_generated_image_N``.
        if img_url:
            lines.append(f"   Image: {img_url} <img>")
        # Emit BOTH the source domain (semantic cue) AND the full source
        # page URL so downstream LLM curation / memory rewrite have a
        # real ``https://...`` to work with. Without the full URL the
        # LLM cannot fill the required ``url`` field for search_type=images
        # and memory cannot register it as ``webpage_url_N``.
        if source and link:
            lines.append(f"   Source: {source}   {link}")
        elif link:
            lines.append(f"   URL: {link}")
        elif source:
            lines.append(f"   Source: {source}")
        if w and h:
            lines.append(f"   Size: {w}x{h}")
    return "\n".join(lines)


def _format_news(data: Dict[str, Any], max_results: int) -> str:
    items = data.get("news", [])[:max_results]
    lines = []
    for i, item in enumerate(items, 1):
        title = item.get("title", "")
        link = item.get("link", "")
        source = item.get("source", "")
        date = item.get("date", "")
        snippet = item.get("snippet", "")
        thumbnail = _http_image_url(item.get("imageUrl")) or _http_image_url(item.get("thumbnailUrl"))
        lines.append(f"{i}. {title}")
        url_compact = _display_url_compact(item, link)
        if url_compact and link:
            lines.append(f"   Source: {url_compact}   {link}")
        elif link:
            lines.append(f"   URL: {link}")
        # ImageStateMemory routes URLs trailed by ` <img>` directly
        # into the image registry as ``tool_generated_image_N``.
        if thumbnail:
            lines.append(f"   Image: {thumbnail} <img>")
        meta = " | ".join(filter(None, [source, date]))
        if meta:
            lines.append(f"   {meta}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


def _format_maps(data: Dict[str, Any], max_results: int) -> str:
    """Format /maps responses (places array with rich location data).

    Serper /maps fields per item (verified on a 20-result probe 2026-05-19):
      always: title, address, latitude, longitude, rating, ratingCount,
              type, types, openingHours, thumbnailUrl, cid, fid, placeId
      ~95%:   priceLevel
      ~70%:   website
      ~50%:   phoneNumber
      ~30%:   description

    Emit a Webpage_Visit-able URL on every item: prefer ``website`` (the
    real business homepage), fall back to ``cid`` → Google Maps deeplink
    (always available). Without this fallback, ~30% of /maps results had
    no URL at all, blocking follow-up. ``openingHours`` is omitted
    intentionally — a 7-day dict per result inflates context badly;
    Webpage_Visit on the emitted URL is the cheaper way to fetch hours.
    """
    items = data.get("places", [])[:max_results]
    lines = []
    for i, item in enumerate(items, 1):
        title = item.get("title", "")
        address = item.get("address", "")
        rating = item.get("rating", "")
        count = item.get("ratingCount", "")
        place_type = item.get("type", "")
        phone = item.get("phoneNumber", "")
        website = item.get("website", "")
        description = item.get("description", "")
        price_level = item.get("priceLevel", "")
        cid = item.get("cid", "")
        img_url = _http_image_url(item.get("thumbnailUrl")) or _http_image_url(item.get("imageUrl"))
        lines.append(f"{i}. {title}")
        if place_type:
            lines.append(f"   Type: {place_type}")
        if address:
            lines.append(f"   Address: {address}")
        if rating:
            lines.append(f"   Rating: {rating}" + (f" ({count} reviews)" if count else ""))
        if price_level:
            lines.append(f"   Price level: {price_level}")
        if phone:
            lines.append(f"   Phone: {phone}")
        # Prefer the real business homepage; fall back to Google Maps
        # deeplink via ``cid`` so every result has a follow-up URL.
        if website:
            lines.append(f"   Website: {website}")
        elif cid:
            lines.append(f"   URL: https://www.google.com/maps?cid={cid}")
        if description:
            lines.append(f"   Description: {description}")
        # ImageStateMemory routes URLs trailed by ` <img>` directly
        # into the image registry as ``tool_generated_image_N``.
        if img_url:
            lines.append(f"   Image: {img_url} <img>")
    return "\n".join(lines)


def _format_places(data: Dict[str, Any], max_results: int) -> str:
    """Format /places responses.

    Serper /places returns: title, address, latitude, longitude, rating,
    ratingCount, priceLevel, category, cid.  Notably absent (vs /maps):
    phoneNumber, website, thumbnailUrl.  ``cid`` is the Google Maps place
    id — emit it as a deeplink URL so Webpage_Visit_Tool can follow up.
    """
    items = data.get("places", [])[:max_results]
    lines = []
    for i, item in enumerate(items, 1):
        title = item.get("title", "")
        address = item.get("address", "")
        rating = item.get("rating", "")
        count = item.get("ratingCount", "")
        category = item.get("category", "")
        price_level = item.get("priceLevel", "")
        cid = item.get("cid", "")
        lat = item.get("latitude", "")
        lng = item.get("longitude", "")
        lines.append(f"{i}. {title}")
        if category:
            lines.append(f"   Category: {category}")
        if address:
            lines.append(f"   Address: {address}")
        if rating:
            lines.append(f"   Rating: {rating}" + (f" ({count} reviews)" if count else ""))
        if price_level:
            lines.append(f"   Price level: {price_level}")
        if cid:
            lines.append(f"   URL: https://www.google.com/maps?cid={cid}")
        if lat and lng:
            lines.append(f"   Coords: {lat}, {lng}")
    return "\n".join(lines)


def _format_shopping(data: Dict[str, Any], max_results: int) -> str:
    """Format /shopping responses.

    Serper /shopping returns: title, source, link, price, imageUrl,
    rating, ratingCount, productId.  ``rating`` + ``ratingCount`` are
    primary product-quality signals; emit them so price-comparison
    queries get review context, not just price.
    """
    items = data.get("shopping", [])[:max_results]
    lines = []
    for i, item in enumerate(items, 1):
        title = item.get("title", "")
        price = item.get("price", "")
        link = item.get("link", "")
        source = item.get("source", "")
        rating = item.get("rating", "")
        count = item.get("ratingCount", "")
        # Shopping items frequently carry data: URLs (~75% on /shopping);
        # _http_image_url filters those out so we don't bloat agent context.
        img_url = _http_image_url(item.get("imageUrl")) or _http_image_url(item.get("thumbnailUrl"))
        lines.append(f"{i}. {title}")
        if price:
            lines.append(f"   Price: {price}")
        if rating:
            lines.append(f"   Rating: {rating}" + (f" ({count} reviews)" if count else ""))
        if link:
            lines.append(f"   URL: {link}")
        if source:
            lines.append(f"   Source: {source}")
        # ImageStateMemory routes URLs trailed by ` <img>` directly
        # into the image registry as ``tool_generated_image_N``.
        if img_url:
            lines.append(f"   Image: {img_url} <img>")
    return "\n".join(lines)


def _format_generic(data: Dict[str, Any], max_results: int) -> str:
    """Fallback formatter: try organic, then first list-like value."""
    if "organic" in data:
        return _format_organic(data, max_results)
    for key, val in data.items():
        if isinstance(val, list) and val:
            lines = []
            for i, item in enumerate(val[:max_results], 1):
                if isinstance(item, dict):
                    title = item.get("title", "")
                    link = item.get("link", item.get("url", ""))
                    lines.append(f"{i}. {title}")
                    if link:
                        lines.append(f"   URL: {link}")
                else:
                    lines.append(f"{i}. {item}")
            return "\n".join(lines)
    return str(data)


_FORMATTERS = {
    "search": _format_organic,
    "scholar": _format_organic,
    "patents": _format_organic,
    "images": _format_images,
    "news": _format_news,
    "maps": _format_maps,
    "places": _format_places,
    "shopping": _format_shopping,
}


