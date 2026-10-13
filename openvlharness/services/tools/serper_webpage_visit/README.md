# Webpage extraction

Use the shared [search setup](../serper_text_search/README.md). The router
injects the configured auxiliary model; the paper default is greedy
Qwen3-VL-8B-Instruct with a 16384-token extraction cap. The public tool takes
`urls` (up to five) and `goal`. Runtime webpage references are resolved before dispatch.

Serper scraping processes each page independently. Keep successful pages when
a sibling URL fails, retaining its error alongside original rendered text.
Structured page metadata is preserved before formatting for the capability's
canonical `pages` result.
