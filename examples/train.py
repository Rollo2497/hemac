"""Train independent RL policies on HeMAC and save checkpoints that eval.py can load.

HeMAC is heterogeneous: drones, observers and provisioners have different observation and action
spaces, so a single shared network over all agents is not applicable without padding the spaces
into a common one (which would discard exactly the heterogeneity this benchmark is about). Instead
this script trains *independent learners* — one Stable-Baselines3 model per policy group — which is
the IPPO setting, and matches the per-agent checkpoints eval.py already expects.

Each learner is trained through `SingleAgentAdapter`, which exposes one agent as a plain Gymnasium
env while every other agent acts with its current (frozen) policy. Learners take turns across
several rounds, so each one keeps improving against increasingly competent team-mates.

Checkpoints are written as `{task}_{algorithm}_{agent}_{timestamp}.zip` in the working directory,
which is the pattern eval.py globs for.

Usage:
    python examples/train.py --task simple_fleet --rounds 4 --steps-per-round 20000
    python examples/eval.py   # evaluates the sanity-check baseline; see --help notes in the README
"""

import argparse
import datetime
import os
import shutil
import tempfile

import gymnasium
import numpy as np
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from hemac import HeMAC_v0

try:  # `python examples/train.py` puts examples/ on sys.path; `python -m examples.train` does not
    from scenarios import get_scenario
except ImportError:
    from examples.scenarios import get_scenario

ALGOS = {"PPO": PPO, "SAC": SAC, "DQN": DQN}

# SAC needs a continuous action space, DQN a discrete one. PPO handles both, which is why it is the
# default: observers and provisioners are always Discrete(5), while drones depend on drone_config.
CONTINUOUS_ONLY = ("SAC",)
DISCRETE_ONLY = ("DQN",)


def agent_type(agent: str) -> str:
    """Return the type prefix of an agent name.

    Args:
    ----
        agent (str): Agent key such as "drone_2".

    Returns:
    -------
        str: The type, such as "drone".

    """
    return agent.rsplit("_", 1)[0]


def coerce_action(action, space):
    """Coerce a policy output into a value the environment's space accepts.

    HeMAC validates every action with `space.contains(...)` and raises otherwise, so discrete
    actions must be plain integers and continuous ones must be in-bounds float32.

    Args:
    ----
        action: Raw action from a policy or from `space.sample()`.
        space (gymnasium.Space): The target agent's action space.

    Returns:
    -------
        The action, converted to match `space`.

    """
    if isinstance(space, gymnasium.spaces.Discrete):
        return int(np.asarray(action).reshape(-1)[0])
    action = np.asarray(action, dtype=np.float32).reshape(space.shape)
    return np.clip(action, space.low, space.high)


class PolicyBook:
    """Hold one model per policy group and resolve which model drives which agent."""

    def __init__(self, groups: dict):
        """Build the book.

        Args:
        ----
            groups (dict): Mapping of group key to the list of agent names it controls.

        """
        self.groups = groups
        self.agent_to_group = {agent: key for key, members in groups.items() for agent in members}
        self.models = {}

    def model_for(self, agent: str):
        """Return the model controlling `agent`, or None while it is still untrained."""
        return self.models.get(self.agent_to_group[agent])

    def snapshot(self, directory: str, algorithm: str):
        """Freeze the current policies to disk and return a picklable handle to them.

        Live SB3 models hold weakrefs and cannot be cloudpickled into worker processes, so
        parallel rollouts ship file paths instead and each worker loads its own copy.

        Args:
        ----
            directory (str): Directory to write the checkpoints into.
            algorithm (str): Key of `ALGOS` used to reload them.

        Returns:
        -------
            PolicySnapshot: Handle usable in place of this book.

        """
        paths = {}
        for key, model in self.models.items():
            path = os.path.join(directory, f"partner_{key}")
            model.save(path)
            paths[key] = f"{path}.zip"
        return PolicySnapshot(dict(self.agent_to_group), paths, algorithm)


class PolicySnapshot:
    """Picklable, read-only view of frozen partner policies, loaded lazily per process."""

    def __init__(self, agent_to_group: dict, paths: dict, algorithm: str):
        """Build the snapshot from an agent-to-group map and group-to-checkpoint paths."""
        self.agent_to_group = agent_to_group
        self.paths = paths
        self.algorithm = algorithm
        self._loaded = {}

    def __getstate__(self):
        """Drop loaded models so only paths cross the process boundary."""
        return {"agent_to_group": self.agent_to_group, "paths": self.paths, "algorithm": self.algorithm}

    def __setstate__(self, state):
        """Restore paths and start with an empty per-process model cache."""
        self.__dict__.update(state)
        self._loaded = {}

    def model_for(self, agent: str):
        """Return the frozen model controlling `agent`, loading it on first use."""
        key = self.agent_to_group[agent]
        path = self.paths.get(key)
        if path is None:
            return None
        if key not in self._loaded:
            self._loaded[key] = ALGOS[self.algorithm].load(path, device="cpu")
        return self._loaded[key]


class SingleAgentAdapter(gymnasium.Env):
    """Expose a single HeMAC agent as a Gymnasium env, with all other agents frozen.

    The team-mates become part of the environment dynamics, which is what makes vanilla
    single-agent SB3 applicable to a multi-agent problem.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_kwargs: dict, learner: str, book: PolicyBook, seed: int = None):
        """Build the adapter.

        Args:
        ----
            env_kwargs (dict): Kwargs forwarded to `HeMAC_v0.parallel_env`.
            learner (str): Name of the agent being trained.
            book (PolicyBook): Policies used to drive the other agents.
            seed (int): Seed for the first reset and for partner action sampling.

        """
        super().__init__()
        self.env = HeMAC_v0.parallel_env(**env_kwargs)
        self.learner = learner
        self.book = book
        self._seed = seed
        self._obs = {}

        self.observation_space = self.env.observation_space(learner)
        self.action_space = self.env.action_space(learner)

    def reset(self, *, seed=None, options=None):
        """Reset the underlying multi-agent env and return the learner's observation."""
        seed = seed if seed is not None else self._seed
        self._seed = None  # only seed the first episode, so later episodes vary
        obs, infos = self.env.reset(seed=seed)
        self._obs = obs
        return obs[self.learner], infos.get(self.learner, {})

    def _partner_action(self, agent: str):
        """Return the action of a non-learning agent, from its policy or at random."""
        space = self.env.action_space(agent)
        model = self.book.model_for(agent)
        if model is None:
            return space.sample()
        action, _ = model.predict(self._obs[agent], deterministic=False)
        return coerce_action(action, space)

    def step(self, action):
        """Step every agent once and return the learner's transition."""
        actions = {self.learner: coerce_action(action, self.action_space)}
        for agent in self.env.agents:
            if agent != self.learner:
                actions[agent] = self._partner_action(agent)

        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        # A terminated agent is dropped from the returned dicts, so fall back to the last
        # observation to keep returning something shaped like the observation space.
        learner_obs = obs.get(self.learner, self._obs[self.learner])
        self._obs = obs if obs else self._obs

        terminated = terminations.get(self.learner, True)
        truncated = truncations.get(self.learner, False)
        return (
            learner_obs,
            float(rewards.get(self.learner, 0.0)),
            bool(terminated),
            bool(truncated and not terminated),
            infos.get(self.learner, {}),
        )

    def close(self):
        """Close the underlying environment."""
        self.env.close()


def make_env_fn(env_kwargs: dict, learner: str, book, seed: int):
    """Return a zero-argument factory that builds one monitored learner environment.

    Used for both `DummyVecEnv` and `SubprocVecEnv`. For the subprocess case the closure
    (including the frozen partner policies in `book`) is cloudpickled into each worker, which
    is sound because partner policies do not change within a round.

    Args:
    ----
        env_kwargs (dict): Kwargs forwarded to `HeMAC_v0.parallel_env`.
        learner (str): Name of the agent being trained.
        book (PolicyBook): Policies driving the other agents.
        seed (int): Seed for this worker's first episode.

    Returns:
    -------
        callable: Factory producing a `Monitor`-wrapped `SingleAgentAdapter`.

    """

    def _init():
        return Monitor(SingleAgentAdapter(env_kwargs, learner, book, seed=seed))

    return _init


def make_vec_env(env_kwargs: dict, learner: str, book, seed: int, n_envs: int):
    """Build a vectorised learner environment.

    `SubprocVecEnv` runs each copy in its own process, which is what makes extra cores useful:
    HeMAC stepping is single-threaded pygame work, so one environment leaves a multi-core
    machine mostly idle.

    Args:
    ----
        env_kwargs (dict): Kwargs forwarded to `HeMAC_v0.parallel_env`.
        learner (str): Name of the agent being trained.
        book (PolicyBook): Policies driving the other agents.
        seed (int): Base seed; each copy gets `seed + i`.
        n_envs (int): Number of environment copies. 1 keeps everything in-process.

    Returns:
    -------
        VecEnv: The vectorised environment.

    """
    fns = [make_env_fn(env_kwargs, learner, book, seed + i) for i in range(n_envs)]
    if n_envs > 1:
        return SubprocVecEnv(fns)
    return DummyVecEnv(fns)


def build_groups(possible_agents: list, share_by_type: bool) -> dict:
    """Group agents into policies.

    Args:
    ----
        possible_agents (list): All agent names in the scenario.
        share_by_type (bool): If True, all agents of a type share one policy. If False, every
            agent learns its own policy, which is closer to the paper's IPPO setting but costs
            one training run per agent.

    Returns:
    -------
        dict: Group key to list of agent names.

    """
    groups = {}
    for agent in possible_agents:
        key = agent_type(agent) if share_by_type else agent
        groups.setdefault(key, []).append(agent)
    return groups


def check_algorithm(algorithm: str, space, group: str):
    """Raise if the algorithm cannot handle the group's action space."""
    is_discrete = isinstance(space, gymnasium.spaces.Discrete)
    if algorithm in CONTINUOUS_ONLY and is_discrete:
        raise ValueError(
            f"{algorithm} requires a continuous action space but '{group}' is {space}. "
            f"Use PPO, or set drone_config['discrete_action_space'] = False where applicable."
        )
    if algorithm in DISCRETE_ONLY and not is_discrete:
        raise ValueError(
            f"{algorithm} requires a discrete action space but '{group}' is {space}. "
            f"Use PPO, or set drone_config['discrete_action_space'] = True."
        )


def make_model(algorithm: str, env, seed: int, log_dir: str, n_steps: int):
    """Instantiate an SB3 model for a learner environment."""
    kwargs = dict(policy="MlpPolicy", env=env, verbose=0, seed=seed, tensorboard_log=log_dir)
    if algorithm == "PPO":
        kwargs["n_steps"] = n_steps
    return ALGOS[algorithm](**kwargs)


def train(
    task: str = "simple_fleet",
    algorithm: str = "PPO",
    rounds: int = 4,
    steps_per_round: int = 20000,
    seed: int = 0,
    share_by_type: bool = True,
    n_steps: int = 1024,
    n_envs: int = 1,
    log_dir: str = "./tensorboard_logs",
):
    """Train one policy per group by rotating through the groups for several rounds.

    Args:
    ----
        task (str): Scenario name from `scenarios.SCENARIOS`.
        algorithm (str): One of PPO, SAC, DQN.
        rounds (int): How many times each group is trained.
        steps_per_round (int): Environment steps per group per round.
        seed (int): Base seed.
        share_by_type (bool): Share one policy across all agents of the same type.
        n_steps (int): PPO rollout length per environment.
        n_envs (int): Environment copies run in parallel. >1 uses one process each.
        log_dir (str): Tensorboard directory.

    Returns:
    -------
        dict: Agent name to saved checkpoint path.

    """
    if algorithm not in ALGOS:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Available: {sorted(ALGOS)}")
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}.")
    if algorithm != "PPO" and n_envs > 1:
        raise ValueError(f"{algorithm} does not support n_envs > 1 here; use PPO or set --n-envs 1.")
    # PPO collects n_steps per environment per update, so one update costs n_steps * n_envs.
    if algorithm == "PPO" and steps_per_round < n_steps * n_envs:
        raise ValueError(
            f"steps_per_round ({steps_per_round}) must be >= n_steps * n_envs ({n_steps * n_envs}) for PPO."
        )

    env_kwargs = get_scenario(task)
    env_kwargs["render_mode"] = None

    probe = HeMAC_v0.parallel_env(**env_kwargs)
    possible_agents = list(probe.possible_agents)
    action_spaces = {agent: probe.action_space(agent) for agent in possible_agents}
    probe.close()

    groups = build_groups(possible_agents, share_by_type)
    book = PolicyBook(groups)
    print(f"Training {algorithm} on '{task}': {len(possible_agents)} agents in {len(groups)} policy groups")
    for key, members in groups.items():
        check_algorithm(algorithm, action_spaces[members[0]], key)
        print(f"  {key:<12} -> {members} action_space={action_spaces[members[0]]}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for rnd in range(rounds):
        for key, members in groups.items():
            learner = members[0]  # members share weights, so training one updates them all
            # Workers cannot receive live models, so freeze partners to disk for this round.
            snap_dir = tempfile.mkdtemp(prefix="hemac_partners_") if n_envs > 1 else None
            partners = book.snapshot(snap_dir, algorithm) if n_envs > 1 else book
            env = make_vec_env(env_kwargs, learner, partners, seed + rnd * n_envs, n_envs)
            model = book.models.get(key)
            if model is None:
                model = make_model(algorithm, env, seed + rnd, f"{log_dir}/train_{algorithm}_{timestamp}", n_steps)
                book.models[key] = model
            else:
                model.set_env(env)

            print(f"[round {rnd + 1}/{rounds}] training '{key}' via {learner} for {steps_per_round} steps")
            model.learn(
                total_timesteps=steps_per_round,
                reset_num_timesteps=False,
                tb_log_name=key,
                progress_bar=False,
            )
            env.close()
            if snap_dir:
                shutil.rmtree(snap_dir, ignore_errors=True)

    saved = {}
    for agent in possible_agents:
        path = f"{task}_{algorithm}_{agent}_{timestamp}"
        book.model_for(agent).save(path)
        saved[agent] = f"{path}.zip"
        print(f"saved {saved[agent]}")

    return saved


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="simple_fleet", help="scenario name (see examples/scenarios.py)")
    parser.add_argument("--algorithm", default="PPO", choices=sorted(ALGOS), help="SB3 algorithm")
    parser.add_argument("--rounds", type=int, default=4, help="training rounds per policy group")
    parser.add_argument("--steps-per-round", type=int, default=20000, help="env steps per group per round")
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO rollout length per environment")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="parallel environment copies (PPO only); >1 spawns one process each. "
        "Try roughly the core count, e.g. --n-envs 8",
    )
    parser.add_argument("--seed", type=int, default=0, help="base random seed")
    parser.add_argument(
        "--per-agent-policy",
        action="store_true",
        help="train a separate policy per agent instead of sharing one per agent type",
    )
    args = parser.parse_args()

    # HeMAC always initialises pygame; the dummy driver keeps training headless.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    train(
        task=args.task,
        algorithm=args.algorithm,
        rounds=args.rounds,
        steps_per_round=args.steps_per_round,
        seed=args.seed,
        share_by_type=not args.per_agent_policy,
        n_steps=args.n_steps,
        n_envs=args.n_envs,
    )


if __name__ == "__main__":
    main()
