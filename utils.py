
import datetime

def reset_names(cfg):
    path = cfg.root / "results" / cfg.game_name / cfg.custom_map / cfg.network
    cfg.name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}{cfg.append}'
    cfg.log_name = f"{cfg.game_name}_{cfg.custom_map}_{cfg.network}_{cfg.name}"
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