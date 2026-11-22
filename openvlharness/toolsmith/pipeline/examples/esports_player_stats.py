from openvlharness.toolsmith import GeneratedTool, ToolCallError


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
        summary = page.split("Summary:", 1)[-1].strip() if isinstance(page, str) else ""
        return summary.splitlines()[0][:120] if summary else {"error": f"{statistic} not found on {url}"}
