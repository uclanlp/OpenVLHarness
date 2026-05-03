"""Regression test: pin Serper API field names that our formatters read.

Background: ``serper_image_search/tool.py`` previously read
``item.get("thumbnail")`` but Serper's Google Lens (`/lens`) returns
``thumbnailUrl`` / ``imageUrl``. The typo silently dropped 100% of
``Image:`` line emissions in reverse image search results. Audit on
2026-05-02 found three additional formatters (``_format_shopping``,
``_format_maps``, webpage_visit ``og:image``) that were also dropping
available image fields.

This test snapshots the exact Serper field names we depend on and
exercises each formatter against fixtures shaped like real Serper
responses. If Serper renames a field, the formatter contract breaks
and this test fails — surfacing the regression before agents in
production silently lose image refs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add tool_infra/src to path so we can import ``tools.*`` directly without
# pulling in the broken ``server.app`` import chain that other tests trip on.
_TOOL_INFRA_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_TOOL_INFRA_SRC))


def test_serper_lens_fixture_shape():
    """Serper /lens organic items must carry imageUrl + thumbnailUrl."""
    from tools.serper_image_search.tool import Image_Search_Tool

    t = Image_Search_Tool.__new__(Image_Search_Tool)

    # Real shape from /lens organic[0] (captured 2026-05-02).
    real_item = {
        "title": "Some Eiffel Tower lookalike",
        "link": "https://example.com/page",
        "source": "Example",
        "imageUrl": "https://example.com/full.jpg",
        "thumbnailUrl": "https://encrypted-tbn0.gstatic.com/x",
        "position": 1,
    }
    out = t._extract_item(real_item)
    assert out is not None
    assert out["thumbnail"] == "https://example.com/full.jpg", (
        "imageUrl must be primary image source for Image_Search"
    )

    # Thumbnail-only fallback (some items lack imageUrl).
    out = t._extract_item({
        "title": "X",
        "link": "https://x.com/p",
        "thumbnailUrl": "https://encrypted-tbn0.gstatic.com/y",
    })
    assert out["thumbnail"] == "https://encrypted-tbn0.gstatic.com/y"

    # Old wrong field name must not match — guards against re-introducing
    # the typo by accident.
    out = t._extract_item({
        "title": "X",
        "link": "https://x.com/p",
        "thumbnail": "this-is-the-wrong-field-name",
    })
    assert out["thumbnail"] == "", (
        "field name 'thumbnail' (no Url suffix) does not exist in "
        "Serper /lens response — must be ignored"
    )


def test_image_search_format_emits_img_sentinel():
    from tools.serper_image_search.tool import Image_Search_Tool

    t = Image_Search_Tool.__new__(Image_Search_Tool)
    formatted = t._format_raw_results([{
        "title": "Eiffel",
        "link": "https://example.com/page",
        "source": "",
        "thumbnail": "https://encrypted-tbn0.gstatic.com/x",
    }])
    assert "Image: https://encrypted-tbn0.gstatic.com/x <img>" in formatted, (
        "image_search must emit 'Image: <url> <img>' line — sentinel "
        "drives ImageStateMemory routing"
    )


def test_format_images_thumbnail_fallback():
    from tools.serper_common import _format_images

    # imageUrl present → use it
    out = _format_images({"images": [{
        "title": "X",
        "imageUrl": "https://primary/full.jpg",
        "thumbnailUrl": "https://fallback/thumb",
    }]}, 5)
    assert "Image: https://primary/full.jpg <img>" in out

    # imageUrl missing → fall back to thumbnailUrl
    out = _format_images({"images": [{
        "title": "X",
        "thumbnailUrl": "https://fallback/thumb",
    }]}, 5)
    assert "Image: https://fallback/thumb <img>" in out


def test_format_news_thumbnail_keys():
    from tools.serper_common import _format_news

    # Real /news shape — only imageUrl is present
    out = _format_news({"news": [{
        "title": "Story",
        "link": "https://x.com/a",
        "source": "X",
        "date": "1d",
        "snippet": "...",
        "imageUrl": "https://news.cdn/img.jpg",
    }]}, 5)
    assert "Image: https://news.cdn/img.jpg <img>" in out


def test_format_shopping_emits_image_line():
    from tools.serper_common import _format_shopping

    # Real /shopping shape — has imageUrl, was previously dropped
    out = _format_shopping({"shopping": [{
        "title": "iPhone",
        "price": "$999",
        "link": "https://shop.com",
        "source": "Apple",
        "imageUrl": "https://cdn/p.jpg",
    }]}, 5)
    assert "Image: https://cdn/p.jpg <img>" in out, (
        "shopping must emit Image: line; this was a missed feature pre-fix"
    )


def test_data_url_image_is_filtered_out():
    """Real /shopping returns ~75% data:image/* URLs. They must NOT leak
    into agent text (would bloat context and not be registerable by
    ImageStateMemory's URL rewriter, which only matches https?://).
    """
    from tools.serper_common import (
        _format_shopping, _format_images, _format_news, _format_organic, _format_maps,
    )
    from tools.serper_image_search.tool import Image_Search_Tool

    fake_data_url = "data:image/webp;base64," + "X" * 200

    # Shopping
    out = _format_shopping({"shopping": [{
        "title": "iPhone", "price": "$1", "link": "https://x", "source": "X",
        "imageUrl": fake_data_url,
    }]}, 5)
    assert "data:image" not in out, "shopping must not emit data: URL"
    assert "Image:" not in out, "no Image: line when only data: URL is available"

    # Images
    out = _format_images({"images": [{
        "title": "X", "imageUrl": fake_data_url,
    }]}, 5)
    assert "data:image" not in out
    assert "Image:" not in out

    # Image_Search reverse — fall through data: → http thumbnailUrl
    t = Image_Search_Tool.__new__(Image_Search_Tool)
    extracted = t._extract_item({
        "title": "X", "link": "https://x.com/p",
        "imageUrl": fake_data_url, "thumbnailUrl": "https://encrypted-tbn0/y",
    })
    assert extracted["thumbnail"] == "https://encrypted-tbn0/y"

    # data:-only entry → empty thumbnail (Image: line skipped)
    extracted = t._extract_item({
        "title": "X", "link": "https://x.com/p",
        "imageUrl": fake_data_url,
    })
    assert extracted["thumbnail"] == ""

    # News
    out = _format_news({"news": [{
        "title": "X", "link": "https://x.com", "source": "X", "date": "1d",
        "snippet": "...", "imageUrl": fake_data_url,
    }]}, 5)
    assert "data:image" not in out

    # /search Knowledge Graph
    out = _format_organic({
        "knowledgeGraph": {
            "title": "X", "description": "...",
            "imageUrl": fake_data_url,
        },
        "organic": [],
    }, 5)
    assert "data:image" not in out

    # /maps
    out = _format_maps({"places": [{
        "title": "X", "address": "1 St",
        "thumbnailUrl": fake_data_url,
    }]}, 5)
    assert "data:image" not in out


def test_format_maps_emits_image_line():
    from tools.serper_common import _format_maps

    # Real /maps shape — has thumbnailUrl, phoneNumber, website
    out = _format_maps({"places": [{
        "title": "Cafe",
        "address": "1 St",
        "rating": 4.5,
        "ratingCount": 200,
        "type": "Cafe",
        "phoneNumber": "555-1234",
        "website": "https://cafe.com",
        "thumbnailUrl": "https://lh3.googleusercontent.com/x",
    }]}, 5)
    assert "Phone: 555-1234" in out
    assert "Website: https://cafe.com" in out
    assert "Image: https://lh3.googleusercontent.com/x <img>" in out


def test_format_organic_emits_patent_thumbnail():
    """/patents items each carry a ``thumbnailUrl`` (the patent's main
    figure). Without this, every patent search lost all image refs.
    """
    from tools.serper_common import _format_organic

    out = _format_organic({"organic": [{
        "title": "Battery system patent",
        "link": "https://patents.google.com/patent/US11190026B2",
        "snippet": "...",
        "thumbnailUrl": "https://patentimages.storage.googleapis.com/x.png",
    }]}, 5)
    assert "Image: https://patentimages.storage.googleapis.com/x.png <img>" in out, (
        "patents per-item thumbnailUrl must emit Image: line"
    )


def test_format_organic_emits_top_stories():
    """/search responses with topStories block carry imageUrl per item."""
    from tools.serper_common import _format_organic

    out = _format_organic({
        "organic": [{"title": "Generic", "link": "https://x.com", "snippet": "..."}],
        "topStories": [
            {
                "title": "Apple Q2 earnings",
                "link": "https://finance.yahoo.com/x",
                "source": "Yahoo Finance",
                "date": "1 day ago",
                "imageUrl": "https://encrypted-tbn0.gstatic.com/x",
            },
        ],
    }, 5)
    assert "[Top Stories]" in out
    assert "Image: https://encrypted-tbn0.gstatic.com/x <img>" in out
    assert "Apple Q2 earnings" in out


def test_format_organic_no_top_stories_block_when_empty():
    """No Top Stories section when there's no topStories data."""
    from tools.serper_common import _format_organic

    out = _format_organic({"organic": [{
        "title": "X", "link": "https://x.com", "snippet": "...",
    }]}, 5)
    assert "[Top Stories]" not in out


def test_webpage_visit_returns_text_only():
    """Webpage_Visit is intentionally text-only — it must NOT emit any
    ``Page image:`` line nor consume ``og:image`` / ``jsonld.image`` /
    ``twitter:image``. Image discovery is the search tools' job; mixing
    image emission here would conflate semantics (search-found image vs.
    page hero image) and bloat the agent's ``tool_generated_image_N``
    namespace with often-irrelevant social-card thumbnails.
    """
    from tools.serper_webpage_visit.tool import Webpage_Visit_Tool

    out = Webpage_Visit_Tool._format_success(
        url="https://x.com/page",
        goal="g",
        evidence="ev",
        summary="sum",
    )
    assert "Page image:" not in out
    assert "<img>" not in out
    assert "Image:" not in out
