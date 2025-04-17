import json
import datetime
from pathlib import Path

from trans_zero.utils.muzero_logger import get_wandb_config


# todo refactor this to not introduce strange bugs


def init_config(config, new_config, restart_wandb_id):
    # getting the wandb config and overwriting the relevant fields of the config
    changed_values = {}
    if restart_wandb_id is not None:
        print(f"Using config from: {restart_wandb_id}")
        wandb_config = get_wandb_config(config.wandb_entity, config.wandb_project_name, restart_wandb_id)
        config, changed_values = overwrite_config(config, wandb_config, changed_values = {})

    # Overwrite the config with the provided config, takes precedence over wandb config
    if config:
        config, changed_values = overwrite_config(config, new_config, changed_values)

    print("Values changed:")
    # if there are changed values, print them
    if changed_values:
        print_config(changed_values)

    # refresh the config, this is necessary to update values that depend on other values
    config = refresh(config)

    print_config(config)

    return config


def overwrite_config(std_game_config, new_config, changed_values):
    if type(new_config) is str:
        print(f"Config is string")
        new_config = json.loads(new_config)

    if isinstance(new_config, dict):
        for param, value in new_config.items():
            if hasattr(std_game_config, param):
                setattr(std_game_config, param, value)
                # if the old value is not the new value, add it to the changed values
                if getattr(std_game_config, param) != value:
                    if param == "game_name":
                        raise AttributeError(f"Cannot change game_name after initialization.")
                    changed_values[param] = value
            else:
                raise AttributeError(
                    f"Config has no attribute '{param}'. Check the config file for the complete list of parameters."
                )

    # if game_name in cha
    return std_game_config, changed_values



def set_attributes(cfg, attributes, values, reason=""):
    if not isinstance(attributes, list):
        attributes = [attributes]
        values = [values]

    num_changed_values = 0
    for value, attribute in zip(values, attributes):
        # check that the attribute is different from the current value
        if not getattr(cfg, attribute) == value:
            setattr(cfg, attribute, value)
            if not num_changed_values == 0:
                print(f"\n Automatically changed:")
            print(f"{attribute}: {value}")
            num_changed_values += 1

    if num_changed_values > 0:
        print(f"Due to: {reason} \n")


# todo explain this, basically resets the config values that depend on other values
def refresh(cfg):
    game_name = cfg.game_name

    if not cfg.testing:
        reason = "testing is False"
        set_attributes(cfg, "show_preds", False, reason)

    if cfg.testing:
        reason = "testing is True"
        set_attributes(cfg, "show_preds", True, reason)

    if cfg.debug_mode:
        reason = "debug mode is True"
        set_attributes(cfg,
                       ["logger", "save_model", "train_on_gpu", "selfplay_on_gpu", "reanalyse_on_gpu", "num_workers"],
                            [None, False, False, False, False, 1], reason)

    cfg.name, cfg.log_name, cfg.results_path = cfg.set_names_and_paths()

    # special case for custom grid
    if game_name == "custom_grid":
        cfg.max_moves = cfg.get_max_moves(cfg.custom_map)
        cfg.observation_shape = cfg.get_observation_shape(cfg.pov)  #(1, min(7, int((cfg.custom_map[0])) + 1), 7)

    return cfg



def print_config(cfg):
    for attr, value in vars(cfg).items():
        print(f"{attr}: {value}")