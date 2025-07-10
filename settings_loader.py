import json


def load_settings(game_type="gomoku_simple"):
    with open("settings.json", "r") as f:
        all_settings = json.load(f)

    settings = all_settings[game_type]

    return {
        "board_size": tuple(settings["board_size"]),
        "captures_enabled": settings["captures_enabled"],
        "tournament_rules_enabled": settings["tournament_rules_enabled"],
        "connect_n": settings["connect_n"],
        "table_name": settings["table_name"],
        "db_path": all_settings["db-path"],
        "model": settings["model"],
        "puct_exploration_constant": all_settings["puct_exploration_constant"],
    }


# Example usage:
_settings = load_settings()
BOARD_SIZE = _settings["board_size"]
CAPTURES_ENABLED = _settings["captures_enabled"]
TOURNAMENT_RULES_ENABLED = _settings["tournament_rules_enabled"]
CONNECT_N = _settings["connect_n"]
TABLE_NAME = _settings["table_name"]
DB_PATH = _settings["db_path"]
MODEL = _settings["model"]
PUCT_EXPLORATION_CONSTANT = _settings["puct_exploration_constant"]
