import os
import threading
import time

import requests

from openvlharness.toolsmith import GeneratedTool

HEADERS = {"User-Agent": "OpenVLHarness-ToolSmith/0.1 (https://github.com/uclanlp/OpenVLHarness)",
           "Accept": "application/json"}
MUSICBRAINZ = "https://musicbrainz.org/ws/2"
TMDB = "https://api.themoviedb.org/3"
# MusicBrainz allows one request per second; the calls of this process take turns.
TURN = threading.Lock()
LAST = [0.0]


def wait_for_turn():
    """Return once a second has passed since this process's last MusicBrainz request."""
    with TURN:
        time.sleep(max(0.0, LAST[0] + 1.0 - time.monotonic()))
        LAST[0] = time.monotonic()


def get_json(url, params):
    for attempt in range(3):
        if url.startswith(MUSICBRAINZ):
            wait_for_turn()
        response = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if response.status_code not in {429, 503}:
            break
        time.sleep(1.5 * (attempt + 1))
    response.raise_for_status()
    return response.json()


class MediaCatalog(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="media_catalog",
            tool_description=(
                "Canonical metadata of a creative work or media person: music artists and releases from MusicBrainz; "
                "films, TV series and people from TMDB (needs TMDB_API_KEY). Returns {found, canonical_title, "
                "alternate_titles, native_title, source, ...}. DO NOT USE FOR: identifying works from an image."),
            input_schema={"type": "object", "properties": {
                "title": {"type": "string", "description": "Name of the work, artist or person."},
                "media_type": {"type": "string", "enum": ["film", "tv", "music", "person"]}},
                "required": ["title", "media_type"]})

    def execute(self, title="", media_type=""):
        if not title.strip() or media_type not in {"film", "tv", "music", "person"}:
            return {"found": False, "error": "title and a media_type of film, tv, music or person are required"}
        try:
            return self.music(title.strip()) if media_type == "music" else self.tmdb(title.strip(), media_type)
        except (requests.RequestException, ValueError) as exc:
            # Request errors quote the URL, whose query holds TMDB_API_KEY; report the status or kind only.
            response = getattr(exc, "response", None)
            reason = f"HTTP {response.status_code}" if response is not None else type(exc).__name__
            return {"found": False, "error": f"{media_type} lookup failed: {reason}"}

    @staticmethod
    def music(title):
        # An exact artist or release name wins over the first fuzzy hit of either search.
        fallback = None
        for entity in ("artist", "release"):
            search = get_json(f"{MUSICBRAINZ}/{entity}", {"query": title, "fmt": "json", "limit": 5})
            hits = search.get(f"{entity}s") or []
            exact = next((h for h in hits if (h.get("name") or h.get("title") or "").lower() == title.lower()), None)
            if exact:
                return MediaCatalog.music_record(entity, exact)
            fallback = fallback or (hits and (entity, hits[0]))
        return MediaCatalog.music_record(*fallback) if fallback else {"found": False}

    @staticmethod
    def music_record(entity, hit):
        # A release lookup returns its artists only when asked for their credits.
        inc = "aliases+artist-credits" if entity == "release" else "aliases"
        record = get_json(f"{MUSICBRAINZ}/{entity}/{hit['id']}", {"fmt": "json", "inc": inc})
        aliases = record.get("aliases") or []
        result = {"found": True, "media_type": "music", "source": "musicbrainz", "id": record.get("id"),
                  "canonical_title": record.get("name") or record.get("title", ""),
                  "alternate_titles": [a["name"] for a in aliases if a.get("name")],
                  "native_title": next((a["name"] for a in aliases if a.get("primary") and a.get("name")), ""),
                  "type": record.get("type", ""), "country": record.get("country", "")}
        if entity == "artist":
            return {**result, "life_span": record.get("life-span", {})}
        credit = record.get("artist-credit") or [{}]
        return {**result, "date": record.get("date", ""), "artist": credit[0].get("name", "")}

    @staticmethod
    def tmdb(title, media_type):
        key = os.environ.get("TMDB_API_KEY")
        if not key:
            return {"found": False, "error": "TMDB_API_KEY is not set"}
        endpoint = {"film": "movie", "tv": "tv", "person": "person"}[media_type]
        results = get_json(f"{TMDB}/search/{endpoint}", {"api_key": key, "query": title}).get("results") or []
        if not results:
            return {"found": False}
        # As for music, an exact title or name wins over the first hit of the search.
        hit = next((result for result in results
                    if (result.get("title") or result.get("name") or "").lower() == title.lower()), results[0])
        params = {"api_key": key}
        if media_type != "person":
            params["append_to_response"] = "alternative_titles"
        record = get_json(f"{TMDB}/{endpoint}/{hit['id']}", params)
        if media_type == "person":
            names = record.get("also_known_as") or []
            return {"found": True, "media_type": "person", "source": "tmdb", "id": record.get("id"),
                    "canonical_title": record.get("name", ""), "alternate_titles": names,
                    "native_title": names[0] if names else "", "birthday": record.get("birthday"),
                    "deathday": record.get("deathday"), "place_of_birth": record.get("place_of_birth", ""),
                    "known_for_department": record.get("known_for_department", "")}
        alternatives = record.get("alternative_titles") or {}
        tv = media_type == "tv"
        return {"found": True, "media_type": media_type, "source": "tmdb", "id": record.get("id"),
                "canonical_title": record.get("name" if tv else "title", ""),
                "alternate_titles": [a["title"] for a in alternatives.get("titles") or alternatives.get("results") or []
                                     if a.get("title")],
                "native_title": record.get("original_name" if tv else "original_title", ""),
                "original_language": record.get("original_language", ""),
                "release_date": record.get("first_air_date" if tv else "release_date", ""),
                "genres": [g["name"] for g in record.get("genres") or [] if g.get("name")],
                "overview": record.get("overview", "")}
