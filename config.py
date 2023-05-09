from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


# NOTE: When https://github.com/omry/omegaconf/issues/422 is done, we could
#       use typing.Literal instead of this aberration
class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Devices(StrEnum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


class PolicyTypes(StrEnum):
    gaussian = "gaussian"
    deterministic = "deterministic"


class DistOptimizers(StrEnum):
    bo = "bo"
    cma = "cma"
    rs = "rs"
    pso = "pso"


class BOGPMeans(StrEnum):
    Zero = "Zero"
    Constant = "Constant"
    Linear = "Linear"


class PosVelTypes(StrEnum):
    abs = "abs"
    rel = "rel"
    norm = "norm"


class TorsoHeadTypes(StrEnum):
    velocity = "velocity"
    position = "position"


class HeadWrtTypes(StrEnum):
    torso = "torso"
    butt = "butt"


class DualModes(StrEnum):
    q = "q"
    reward = "reward"


class NormalizationTypes(StrEnum):
    range = "range"
    z_score = "z_score"
    none = "none"


class NormalizationModes(StrEnum):
    min = "min"
    mean = "mean"


@dataclass(kw_only=True)
class LoggerConfig:
    """
    Configuration for the logger.
    """

    project_name: str = "Project"
    """Name of the project."""

    group_name: str = "Group"
    """Name of the group."""

    experiment_id: str = "generic-ID"
    """ID of the experiment."""

    loggers: str = "console,file"
    """List of loggers to use. Possible values: console, file, wandb."""


@dataclass(kw_only=True)
class RewarderConfig:
    """
    Configuration for the rewarder.
    """

    name: str = MISSING
    """Name of the rewarder."""

    batch_size: int = 64
    """Batch size for training."""


@dataclass(kw_only=True)
class EnvRewarderConfig(RewarderConfig):
    """
    Configuration for the environment rewarder.
    """

    name: str = "env"
    """Name of the rewarder."""


@dataclass(kw_only=True)
class GAILConfig(RewarderConfig):
    """
    Configuration for the GAIL rewarder.
    """

    name: str = "gail"
    """Name of the rewarder."""

    log_scale_rewards: bool = False
    """Whether to log scale the rewards."""

    disc_weight_decay: float = 1
    """Weight decay for the discriminator."""


@dataclass(kw_only=True)
class SAILConfig(RewarderConfig):
    """
    Configuration for the SAIL rewarder.
    """

    name: str = "sail"
    """Name of the rewarder."""

    lr: float = 0.0003
    """Learning rate for the rewarder."""

    hidden_size: int = 256
    """Hidden size for the rewarder."""

    disc_weight_decay: float = 1e-5
    """Weight decay for the discriminator."""

    g_inv_weight_decay: float = 1e-5
    """Weight decay for the inverse dynamics model."""

    vae_scaler: float = 1.0
    """Scaler for the VAE loss."""


@dataclass(kw_only=True)
class PWILConfig(RewarderConfig):
    """
    Configuration for the PWIL rewarder.
    """

    name: str = "pwil"
    """Name of the rewarder."""


@dataclass(kw_only=True)
class AgentConfig:
    """
    Configuration for the agent.
    """

    name: str = MISSING
    """Name of the agent."""

    gamma: float = 0.99
    """Discount factor."""

    tau: float = 0.005
    """Soft update factor."""

    lr: float = 0.0003
    """Learning rate."""

    q_weight_decay: float = 1e-5
    """Weight decay for the Q networks."""


@dataclass(kw_only=True)
class SACConfig(AgentConfig):
    """
    Configuration for the SAC agent.
    """

    name: str = "sac"
    """Name of the agent."""

    policy_type: PolicyTypes = "gaussian"  # pyright: ignore[reportGeneralTypeIssues]
    """Type of policy."""

    target_entropy: str = "auto"
    """Target entropy."""

    alpha: float = 0.2
    """Entropy regularization factor."""

    automatic_entropy_tuning: bool = True
    """Whether to automatically tune the entropy."""

    hidden_size: int = 256
    """Hidden size for the networks."""

    target_update_interval: int = 1
    """Interval for updating the target networks."""


@dataclass(kw_only=True)
class DualSACConfig(SACConfig):
    """
    Configuration for the Dual SAC agent.
    """

    name: str = "dual_sac"
    """Name of the agent."""


@dataclass(kw_only=True)
class CoAdaptationConfig:
    """
    Configuration for co-adaptation.
    """

    dist_optimizer: DistOptimizers = "bo"  # pyright: ignore[reportGeneralTypeIssues]
    """
    Optimizer for the distributional distance, either:
    - bo -> Bayesian optimization
    - cma -> CMA (Hansen and Ostermeier 2001)
    - rs -> Random search (Bergstra and Bengio 2012)
    - pso -> Particle Swarm Optimization (Eberhart and Kennedy 1995)
    """

    bo_gp_mean: BOGPMeans = "Zero"  # pyright: ignore[reportGeneralTypeIssues]
    """The type of Gaussian process mean for Bayesian optimization."""

    acq_weight: float = 2.0
    """Bayesian optimization LCB acquisition function exploration weight."""


@dataclass(kw_only=True)
class MethodConfig:
    """
    Configuration for the method.
    """

    name: str = MISSING
    """Name of the method."""

    agent: AgentConfig = MISSING
    """Configuration for the agent."""

    rewarder: RewarderConfig = MISSING
    """Configuration for the rewarder."""

    eval: bool = True
    """Whether to evaluate the agent."""

    eval_episodes: int = 10
    """Number of episodes to evaluate the agent."""

    eval_per_episodes: int = 20
    """Number of episodes between evaluations."""

    batch_size: int = 256
    """Batch size for training."""

    num_episodes: int = 500
    """Number of episodes to train the agent."""

    updates_per_step: int = 1
    """Number of updates per environment step."""

    start_steps: int = 10000
    """Number of steps to take before training."""

    replay_capacity: int = 2000000
    """Size of the replay buffer."""

    replay_dim_ratio: float = 1.0
    """The diminishing ratio for the replay buffer."""

    record_test: bool = False
    """Whether to record the test episodes."""

    save_checkpoints: bool = False
    """Whether to save the checkpoints."""

    save_optimal: bool = False
    """Whether to save the optimal policy."""

    save_final: bool = True
    """Whether to save the final policy."""


@dataclass(kw_only=True)
class CoILConfig(MethodConfig):
    """
    Configuration for CoIL.
    """

    defaults: List[Any] = field(
        default_factory=lambda: [
            {
                "agent": "sac",
            },
            {
                "rewarder": "env",
            },
        ]
    )

    name: str = "coil"
    """Name of the method."""

    co_adaptation: CoAdaptationConfig = field(default_factory=CoAdaptationConfig)
    """Configuration for co-adaptation."""

    expert_demos: str = MISSING
    """Path to the expert demonstrations."""

    morpho_warmup: int = 60000
    """Steps before starting to optimize the morphology."""

    episodes_per_morpho: int = 50
    """Number of episodes per morphology."""

    disc_warmup: int = 20000
    """Steps before starting to train the agent."""

    pos_type: Optional[PosVelTypes] = "norm"  # pyright: ignore[reportGeneralTypeIssues]
    """Which position marker coordinate to use."""

    vel_type: Optional[PosVelTypes] = "rel"  # pyright: ignore[reportGeneralTypeIssues]
    """Which velocity marker coordinate to use."""

    expert_legs: List[int] = field(default_factory=lambda: [0, 1])
    """Which legs to use for marker matching on the demonstrator side."""

    policy_legs: List[int] = field(default_factory=lambda: [0, 1])
    """Which legs to use for marker matching on the imitator side."""

    expert_markers: List[int] = field(default_factory=lambda: [1, 2, 3])
    """Which markers to use for matching on the demonstrator side."""

    policy_markers: List[int] = field(default_factory=lambda: [1, 2, 3])
    """Which markers to use for matching on the imitator side."""

    train_distance_value: bool = False
    """Learn a separate distance value which is used to optimize morphology."""

    co_adapt: bool = True
    """Whether to co-adapt the morphology as well as the behavior."""

    subject_id: int = 8
    """Expert subject name when using CMU dataset."""

    torso_type: Optional[
        TorsoHeadTypes  # pyright: ignore[reportGeneralTypeIssues]
    ] = None
    """Use torso velocity, position or None."""

    head_type: Optional[
        TorsoHeadTypes  # pyright: ignore[reportGeneralTypeIssues]
    ] = None
    """Use head velocity, position or None."""

    head_wrt: Optional[HeadWrtTypes] = None  # pyright: ignore[reportGeneralTypeIssues]
    """Use head with respect to body part."""

    omit_done: bool = False
    """Whether to omit the done signal."""

    fixed_morpho: Optional[float] = None
    """Fixed morphology to use."""


@dataclass(kw_only=True)
class CoSILConfig(CoILConfig):
    """
    Configuration for the CoSIL method.
    """

    defaults: List[Any] = field(
        default_factory=lambda: [
            {
                "agent": "dual_sac",
            },
            {
                "rewarder": "gail",
            },
        ]
    )

    name: str = "cosil"
    """Name of the method."""

    omega_init: float = 1.0
    """Initial value for omega."""

    reset_omega: bool = True
    """Whether to reset omega before each morphology change."""

    imitation_capacity: int = 100000
    """Capacity of the imitation buffer."""

    imitation_dim_ratio: float = 0.5
    """The diminishing ratio for the imitation buffer."""

    imitate_prev_morpho: bool = True
    """Whether to imitate the previous morphology by adding observations from its policy to the imitation buffer."""

    clear_imitation: bool = True
    """
    Whether to clear the imitation buffer before adding the observations of a new morphology.
    As a side effect, if this variable is True, all observations in the imitation buffer are used for training.
    Otherwise (if False), an `obs_per_morpho` number of demonstrations are sampled from the imitation buffer.
    """

    obs_per_morpho: int = 10000
    """
    Number of observations per morphology to generate and add to the imitation buffer.
    This variable is also used as the number of demonstrations to sample from the imitation buffer during training when `clear_imitation` is False.
    """

    morpho_policy_warmup: int = 1
    """
    Number of episodes during which we do not train the policy after a morphology change.
    By doing this, we prevent the policy from being updated with an out-of-sync `imitation_critic` which is itself being updated with an out-of-sync `discriminator`.
    This is due to the contents of the `imitation_buffer` being updated with the prev. morphology's demonstrations.
    The warmup is used even when `clear_imitation` is False, since newer demonstrations may still have a higher weight when sampling from the imitation buffer.
    """

    dual_mode: DualModes = "q"  # pyright: ignore[reportGeneralTypeIssues]
    """The dual mode, either a duality of Q-values or of reward signals."""

    normalization_type: NormalizationTypes = (  # pyright: ignore[reportGeneralTypeIssues]
        "range"
    )
    """Normalization type for the reward or Q-values."""

    normalization_mode: NormalizationModes = (  # pyright: ignore[reportGeneralTypeIssues]
        "min"
    )
    """Normalization mode for the reward or Q-values."""

    normalization_gamma: float = 100.0
    """Normalization gamma for the reward or Q-values."""

    normalization_beta: float = 0.0
    """Normalization beta for the reward or Q-values."""

    normalization_low_clip: Optional[float] = None
    """Normalization lower bound for the clipping of the reward or Q-values."""

    normalization_high_clip: Optional[float] = None
    """Normalization upper bound for the clipping of the reward or Q-values."""


@dataclass(kw_only=True)
class Config:
    """
    Base configuration.
    """

    task: str = MISSING
    """Name of the task."""

    logger: LoggerConfig = field(default_factory=LoggerConfig)
    """Configuration for the logger."""

    seed: int = 1
    """Random seed."""

    device: Devices = "cpu"  # pyright: ignore[reportGeneralTypeIssues]
    """Device to use."""

    env_name: str = MISSING
    """Name of the environment."""

    models_dir_path: str = "models"
    """Path to the directory where to save the models."""

    learn_disc_transitions: bool = False  # FIX: We are gonna have a problem with this
    """Learn discriminator using (s, s') transitions."""

    absorbing_state: bool = False  # FIX: We are gonna have a problem with this
    """Whether to use absorbing states."""

    resume: Optional[str] = None
    """Resume from given policy; specify the path + name of the .pt file to resume from."""


@dataclass(kw_only=True)
class TrainConfig(Config):
    """
    Configuration for training.
    """

    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"method": "cosil"},
        ]
    )

    task: str = "train"
    """Name of the task."""

    method: MethodConfig = MISSING
    """Configuration for the method."""

    num_agents: int = 1
    """Number of agents to train."""


@dataclass(kw_only=True)
class GenTrajectoriesConfig(Config):
    """
    Configuration for generating trajectories as demonstrations.
    """

    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"method": "coil"},
        ]
    )

    task: str = "gen_obs"
    """Name of the task."""

    # FIX: This should not be here, we are not using any method nor their config
    #      The only reason we need this is to have the agent's config
    method: MethodConfig = MISSING
    """Configuration for the method."""

    save_path: str = MISSING
    """Path to save the trajectories."""

    num_obs: int = 10000
    """Number of trajectories to generate."""


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_train", node=TrainConfig)
    cs.store(name="base_gen_obs", node=GenTrajectoriesConfig)
    cs.store(name="base_logger", node=LoggerConfig)
    cs.store(name="base_co_adaptation", node=CoAdaptationConfig)
    cs.store(group="method", name="base_coil", node=CoILConfig)
    cs.store(group="method", name="base_cosil", node=CoSILConfig)
    cs.store(group="method/agent", name="base_sac", node=SACConfig)
    cs.store(group="method/agent", name="base_dual_sac", node=DualSACConfig)
    cs.store(group="method/rewarder", name="base_env_rewarder", node=EnvRewarderConfig)
    cs.store(group="method/rewarder", name="base_gail", node=GAILConfig)
    cs.store(group="method/rewarder", name="base_sail", node=SAILConfig)
    cs.store(group="method/rewarder", name="base_pwil", node=PWILConfig)
