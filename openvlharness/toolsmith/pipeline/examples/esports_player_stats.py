from openvlharness.toolsmith import GeneratedTool, ToolCallError


def page_answers(page):
    """Yield the first summary line of each fetched page; failed fetches carry no answer.

    Each page's block starts with "The useful information in" and ends with its summary,
    after the page's own text, which may hold "=======" or "Summary:" lines too.
    """
    for block in str(page or "").split("\n\n=======\n\nThe useful information in "):
        summary = block.rsplit("\n\nSummary: \n", 1)[1].strip() if "\n\nSummary: \n" in block else ""
        if summary and not summary.startswith(("FETCH FAILED", "No summary generated")):
            yield summary.splitlines()[0][:120]


class EsportsPlayerStats(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="esports_player_stats",
            tool_description=("Read one statistic of a professional esports player from the player's Liquipedia page, "
                              "e.g. total earnings or main role. DO NOT USE FOR: traditional sports or team rosters."),
            input_schema={"type": "object", "properties": {
                "player": {"type": "string", "description": "Player handle as written on Liquipedia."},
                "game": {"type": "string", "description": "Liquipedia wiki, e.g. 'leagueoflegends' or 'counterstrike'."},
                "statistic": {"type": "string", "description": "The statistic to read, e.g. 'approximate total winnings'."}},
                "required": ["player", "game", "statistic"]})

    def execute(self, player="", game="", statistic=""):
        if not (player.strip() and game.strip() and statistic.strip()):
            return {"error": "player, game and statistic are required"}
        url = f"https://liquipedia.net/{game.strip().lower()}/{player.strip().replace(' ', '_')}"
        try:
            page = self.call_tool("Webpage_Visit_Tool", urls=[url],
                                  goal=f"Report the {statistic} of {player}. Reply with the value only.")
        except ToolCallError as exc:
            return {"error": str(exc)}
        return next(page_answers(page), None) or {"error": f"{statistic} not found on {url}"}
