"""Shared HeMAC scenario definitions, so training and evaluation run on identical environments.

Each entry is a kwargs dict accepted by `HeMAC_v0.env()` / `HeMAC_v0.parallel_env()`. Keeping them
here means a policy trained with `train.py --task <name>` is evaluated by `eval.py` on exactly the
same configuration; a policy is only comparable to another one trained on the same scenario.
"""

import numpy as np

# Mirrors `env_kwargs_level1` in eval.py.
PENTAGON = dict(
    time_factor=0.5,
    area_size=(500, 500),
    max_cycles=600,
    render_ratio=0.6,
    n_observers=3,
    n_drones=6,
    n_provisioners=0,
    min_obstacles=1,
    max_obstacles=2,
    rescuing_targets=True,
    observer_comm_range=300,
    patrol_config={
        "benchmark": True,
        "area": [(100, 100), (250, 100), (820, 480), (600, 800), (100, 620)],  # pentagon
    },
    poi_config=[{"speed": 1.0, "dimension": [8, 8], "spawn_mode": "random"}],
    drone_config={
        "drones_starting_pos": [],
        "drone_ui_dimension": 16,
        "drone_max_speed": 10,
        "drone_max_charge": 9999,
        "discrete_action_space": False,  # True if QMIX else false
    },
    drone_sensor={"model": "RoundCamera", "params": {"sensing_range": 30}},
    observer_sensor={"model": "ForwardFacingCamera", "params": {"hfov": np.pi / 6, "sensing_range": 100}},
    provisioner_sensor={"model": "ForwardFacingCamera", "params": {"hfov": np.pi / 2, "sensing_range": 30}},
)

# A deliberately small "Simple Fleet" style scenario: one drone, one observer, no obstacles.
# Useful as a sanity task, since the full pentagon scenario terminates on the first collision
# of any of its six drones and therefore gives very short episodes early in training.
SIMPLE_FLEET = dict(
    time_factor=1,
    area_size=(1000, 1000),
    max_cycles=600,
    n_observers=1,
    n_drones=1,
    n_provisioners=0,
    min_obstacles=0,
    max_obstacles=0,
    rescuing_targets=False,
    observer_comm_range=150,
    patrol_config={
        "benchmark": True,
        "area": [(100, 100), (250, 100), (820, 480), (600, 800), (100, 620)],  # pentagon
    },
    poi_config=[{"speed": 2.0, "dimension": [8, 8], "spawn_mode": "random"}],
    drone_config={
        "drones_starting_pos": [],
        "drone_ui_dimension": 16,
        "drone_max_speed": 10,
        "drone_max_charge": 100,
        "discrete_action_space": False,
    },
    drone_sensor={"model": "RoundCamera", "params": {"sensing_range": 50}},
    observer_sensor={"model": "ForwardFacingCamera", "params": {"hfov": np.pi / 6, "sensing_range": 200}},
)

SCENARIOS = {
    "pentagon": PENTAGON,
    "simple_fleet": SIMPLE_FLEET,
}


def get_scenario(name: str) -> dict:
    """Return a copy of the named scenario kwargs.

    Args:
    ----
        name (str): Scenario key, one of `SCENARIOS`.

    Returns:
    -------
        dict: Environment kwargs, safe to mutate by the caller.

    """
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {sorted(SCENARIOS)}")
    return dict(SCENARIOS[name])
