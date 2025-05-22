import json


class GameSettings:
    def __init__(self, game_type=None):
        with open("settings.json", "r") as f:
            all_settings = json.load(f)
        if game_type is None:
            game_type = all_settings["game"]
        self.settings = all_settings[game_type]
        self.board_size = tuple(self.settings["board_size"])
        self.captures_enabled = self.settings["captures_enabled"]
        self.tournament_rules_enabled = self.settings["tournament_rules_enabled"]
        self.connect_n = self.settings["connect_n"]
        self.db_path = self.settings["db-path"]
