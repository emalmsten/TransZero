
import datetime
from pathlib import Path

def reset_names(cfg):
    game_name = cfg.game_name
    game_str = cfg.game_name
    if cfg.game_name == "frozen_lake":
        game_name = Path(game_name) / cfg.custom_map
        game_str += f"_{cfg.custom_map}"

    path = cfg.root / "results" / game_name / cfg.network
    cfg.name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}{cfg.append}'
    cfg.log_name = f"{game_str}_{cfg.network}_{cfg.name}"
    cfg.results_path = path / cfg.name


def refresh(cfg):
    if cfg.testing:
        cfg.debug_mode = True
        print("Testing mode enabled. Enabling debug_mode")
    if cfg.debug_mode:
        cfg.logger = None
        cfg.save_model = False
        print("Debug mode enabled. Disabling GPU operations, logger, and model saving")
        cfg.train_on_gpu = False
        cfg.selfplay_on_gpu = False
        cfg.reanalyse_on_gpu = False
    if cfg.network != "double":
        cfg.show_preds = False
    reset_names(cfg)

def print_config(cfg):
    for attr, value in vars(cfg).items():
        print(f"{attr}: {value}")