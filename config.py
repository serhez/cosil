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
    skip = "skip"


class TorsoHeadTypes(StrEnum):
    velocity = "velocity"
    position = "position"


class HeadWrtTypes(StrEnum):
    torso = "torso"
    butt = "butt"


class DualModes(StrEnum):
    q = "q"
    loss_term = "loss_term"
    reward = "reward"


class DemosStrategies(StrEnum):
    add = "add"
    replace = "replace"
    only_expert = "only_expert"


class NormTypes(StrEnum):
    range = "range"
    z_score = "z_score"
    scale_shift = "scale_shift"
    none = "none"


class NormModes(StrEnum):
    min = "min"
    mean = "mean"


class Schedulers(StrEnum):
    alternating = "alternating"
    binary = "binary"
    constant = "constant"
    cosine_annealing = "cosine_annealing"
    exponential = "exponential"
    step = "step"


@dataclass(kw_only=True)
class LoggerConfig:
    """
    Configuration for the logger.
    """

    project_name: str = "Project"
    """Name of the project."""

    group_name: str = "Group"
    """Name of the group."""

    experiment_name: str = "Name"
    """Name of the group."""

    run_id: str = "generic-ID"
    """ID of the experiment, should be uniquely set at runtime."""

    loggers: str = "console,file"
    """List of loggers to use. Possible values: console, file, wandb."""

    default_mask: str = "wandb"
    """Default mask to be used by the multi-logger."""


@dataclass(kw_only=True)
class RewarderConfig:
    """
    Configuration for the rewarder.
    """

    name: str = MISSING
    """Name of the rewarder."""

    batch_size: int = 64
    """Batch size for training."""

    log_scale_rewards: bool = False
    """Whether to log scale the rewards."""

    norm_type: NormTypes = "z_score"
    """Normalization type for the rewards."""

    norm_mode: NormModes = "min"
    """Normalization mode for the rewards."""

    rl_norm_gamma: float = 1.0
    """Normalization gamma for the reinforcement rewards."""

    rl_norm_beta: float = 0.0
    """Normalization beta for the reinforcement rewards."""

    il_norm_gamma: float = 1.0
    """Normalization gamma for the imitation rewards."""

    il_norm_beta: float = 0.0
    """Normalization beta for the imitation rewards."""

    norm_low_clip: Optional[float] = None
    """Normalization lower bound for the clipping of the rewards."""

    norm_high_clip: Optional[float] = None
    """Normalization upper bound for the clipping of the rewards."""

    pretrain_updates: int = 10000
    """Number of updates for the pretraining of the rewarder after a morphology change."""


@dataclass(kw_only=True)
class EnvRewarderConfig(RewarderConfig):
    """
    Configuration for the environment rewarder.
    """

    name: str = "env"
    """Name of the rewarder."""


@dataclass(kw_only=True)
class MBCConfig(RewarderConfig):
    """
    Configuration for the Morphological Behavioral Cloning rewarder.
    """

    name: str = "mbc"
    """Name of the rewarder."""

    adapt_period: int = 50
    """Number of episodes every which the MBC rewarder adapts its demonstrator morphology."""


@dataclass(kw_only=True)
class GAILConfig(RewarderConfig):
    """
    Configuration for the GAIL rewarder.
    """

    name: str = "gail"
    """Name of the rewarder."""

    lr: float = 1e-4
    """Learning rate for the rewarder."""

    disc_weight_decay: float = 1
    """Weight decay for the discriminator."""


@dataclass(kw_only=True)
class AIRLConfig(RewarderConfig):
    """
    Configuration for the AIRL rewarder.
    """

    name: str = "airl"
    """Name of the rewarder."""

    lr: float = 1e-4
    """Learning rate for the rewarder."""

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

    policy_type: PolicyTypes = "gaussian"
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

    bc_regularization: bool = False
    """
    Whether to use a behavior cloning regularization term in the policy loss.
    This technique has been proposed in the TD3+BC paper: "A minimalist approach to offline reinforcement learning".
    """

    imit_markers: bool = False
    """
    Whether to use imitation markers as input to the imitation critic.
    If false, use the concatenation of the raw state and the morphology parameters instead.
    """

    imit_critic_warmup: int = 0
    """Number of episodes for each morphology during which the imitation critic is not updated."""

    imit_critic_prev_morpho: bool = False
    """
    Whether to use the previous morphology as input to the imitation critic.
    If False, use the current morphology parameters instead.
    """

    norm_type: NormTypes = "z_score"
    """Normalization type for the policy loss (RL part)."""

    norm_mode: NormModes = "min"
    """Normalization mode for the policy loss (RL part)."""

    norm_gamma: float = 1.0
    """Normalization gamma for the policy loss (RL part)."""

    norm_beta: float = 0.0
    """Normalization beta for the policy loss (RL part)."""

    norm_low_clip: Optional[float] = None
    """Normalization lower bound for the clipping of the policy loss (RL part)."""

    norm_high_clip: Optional[float] = None
    """Normalization upper bound for the clipping of the policy loss (RL part)."""


@dataclass(kw_only=True)
class DualSACConfig(SACConfig):
    """
    Configuration for the Dual SAC agent.
    """

    name: str = "dual_sac"
    """Name of the agent."""

    norm_type: NormTypes = "z_score"
    """Normalization type for the Q-values."""

    norm_mode: NormModes = "min"
    """Normalization mode for the Q-values."""

    rl_norm_gamma: float = 1.0
    """Normalization gamma for the reinforcement Q-values."""

    rl_norm_beta: float = 0.0
    """Normalization beta for the reinforcement Q-values."""

    il_norm_gamma: float = 1.0
    """Normalization gamma for the imitation Q-values."""

    il_norm_beta: float = 0.0
    """Normalization beta for the imitation Q-values."""

    norm_low_clip: Optional[float] = None
    """Normalization lower bound for the clipping of the Q-values."""

    norm_high_clip: Optional[float] = None
    """Normalization upper bound for the clipping of the Q-values."""


@dataclass(kw_only=True)
class CoAdaptationConfig:
    """
    Configuration for co-adaptation.
    """

    morphos_path: Optional[str] = None
    """
    Path to a .pt file containing a fixed list of morphologies to use for co-adaptation.
    This can be used for reproducibility purposes.
    Once the list of morphologies is exhausted, co-adaptation will continue with the specified `dist_optimizer`.
    """

    dist_optimizer: DistOptimizers = "bo"
    """
    Optimizer for the distributional distance, either:
    - bo -> Bayesian optimization
    - cma -> CMA (Hansen and Ostermeier 2001)
    - rs -> Random search (Bergstra and Bengio 2012)
    - pso -> Particle Swarm Optimization (Eberhart and Kennedy 1995)
    """

    bo_gp_mean: BOGPMeans = "Zero"
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

    eval: bool = False
    """Whether to evaluate the agent."""

    eval_episodes: int = 10
    """Number of episodes to evaluate the agent."""

    eval_per_episodes: int = 20
    """Number of episodes between evaluations."""

    eval_final: bool = True
    """Whether to evaluate the agent at the end of training."""

    batch_size: int = 256
    """Batch size for training."""

    num_episodes: int = 500
    """Number of episodes to train the agent."""

    updates_per_step: int = 1
    """Number of updates per environment step."""

    pretrain_updates: int = 50000
    """Number of updates per environment step."""

    pretrain_path: Optional[str] = None
    """Path to pre-trained models"""

    start_steps: int = 10000
    """Number of steps to take before training."""

    replay_buffer_path: Optional[str] = None
    """Path to a pre-filled replay buffer."""

    replay_capacity: int = 2000000
    """Size of the replay buffer."""

    replay_dim_ratio: float = 1.0
    """The diminishing ratio for the replay buffer."""

    record_test: bool = True
    """
    Whether to record the test episodes.
    Note that this is only possible if the `eval` or `eval_final` flags are set to True.
    """

    record_path: str = "videos"
    """Path to the directory where to save the videos."""

    save_checkpoints: bool = False
    """Whether to save the checkpoints."""

    save_optimal: bool = False
    """Whether to save the optimal policy."""

    save_final: bool = True
    """Whether to save the final policy."""

    save_buffers: bool = True
    """Whether to save the replay buffers."""

    save_agents: bool = True
    """Whether to save the agents' parameters."""

    save_rewarders: bool = True
    """Whether to save the rewarders' parameters."""

    save_morphos: bool = True
    """Whether to save the morphologies."""

    save_demos: bool = True
    """Whether to save the demonstrations."""

    sparse_mask: Optional[float] = None
    """
    The sparsity mask for the envrionment.
    For example, a mask of 90.0 means that all environment rewards below 90.0 are made 0.0.
    If None, no mask is applied.
    """

    rm_action_penalty: bool = False
    """Whether to remove the action penalty given as part of the reward by the environment."""


@dataclass(kw_only=True)
class RLConfig(MethodConfig):
    """
    Configuration for the RL method.
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

    name: str = "rl"
    """Name of the method."""

    pos_type: Optional[PosVelTypes] = "norm"
    """Which position marker coordinate to use."""

    vel_type: Optional[PosVelTypes] = "rel"
    """Which velocity marker coordinate to use."""

    expert_legs: List[int] = field(default_factory=lambda: [0, 1])
    """Which legs to use for marker matching on the demonstrator side."""

    policy_legs: List[int] = field(default_factory=lambda: [0, 1])
    """Which legs to use for marker matching on the imitator side."""

    expert_markers: List[int] = field(default_factory=lambda: [1, 2, 3])
    """Which markers to use for matching on the demonstrator side."""

    policy_markers: List[int] = field(default_factory=lambda: [1, 2, 3])
    """Which markers to use for matching on the imitator side."""

    torso_type: Optional[TorsoHeadTypes] = None
    """Use torso velocity, position or None."""

    head_type: Optional[TorsoHeadTypes] = None
    """Use head velocity, position or None."""

    head_wrt: Optional[HeadWrtTypes] = None
    """Use head with respect to body part."""

    omit_done: bool = False
    """Whether to omit the done signal."""


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
                "rewarder": "airl",
            },
        ]
    )

    name: str = "coil"
    """Name of the method."""

    co_adaptation: CoAdaptationConfig = field(default_factory=CoAdaptationConfig)
    """Configuration for co-adaptation."""

    expert_demos: Optional[str] = None
    """Path to the expert demonstrations."""

    morpho_warmup: int = 50000
    """Steps before starting to optimize the morphology."""

    episodes_per_morpho: int = 50
    """Number of episodes per morphology."""

    disc_warmup: int = 20000
    """Steps before starting to train the agent."""

    pos_type: Optional[PosVelTypes] = "norm"
    """Which position marker coordinate to use."""

    vel_type: Optional[PosVelTypes] = "rel"
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

    torso_type: Optional[TorsoHeadTypes] = None
    """Use torso velocity, position or None."""

    head_type: Optional[TorsoHeadTypes] = None
    """Use head velocity, position or None."""

    head_wrt: Optional[HeadWrtTypes] = None
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
                "agent": "sac",
            },
            {
                "rewarder": "mbc",
            },
        ]
    )

    name: str = "cosil"
    """Name of the method."""

    transfer: bool = True
    """Whether to perform transfer learning."""

    random_morphos: bool = False
    """Whether to sample random morphologies when adapting."""

    replay_weight: float = 0.0
    """
    Ratio of observations in each batch coming from the replay buffer when training the individual agent.
    The ratio of observations coming from the buffer contaning only obs. from the current morphology is consequently (1 - ind_replay_weight).
    """

    demos_n_ep: float = 10
    """Number of episodes for each morphology to consider as demonstrations, starting from the last episode."""

    add_new_demos: bool = True
    """Whether to add new demonstrations to the demos buffer when changing morphology."""

    optimized_demonstrator: bool = True
    """Whether to use an optimized demonstrator (True) or a previously seen morphology, when using MBC."""

    dual_mode: DualModes = "loss_term"
    """The dual mode, either a duality of Q-values or an IL loss term."""

    pop_omega_init: float = 0.0
    """Initial value for omega (population agent)."""

    omega_init: float = 0.2
    """Initial value for omega."""

    omega_scheduler: Schedulers = "constant"
    """The type of scheduler for omega."""

    omega_init_ep: int = 5
    """Number of episodes before starting to change omega."""

    demos_strategy: DemosStrategies = "add"
    """The strategy to use when adding new demonstrations."""

    pretrain_morpho: bool = False
    """Whether to pretrain offline the agent after each morphology change."""

    pretrain_morpho_omega: float = 0.5
    """The value of omega to use when pretraining the agent."""

    pretrain_morpho_updates: int = 3000
    """The number of updates to perform when pretraining the agent."""

    pretrain_morpho_ind_demonstrators: bool = False
    """
    Whether to optimize the chosen demonstrators for every observation in a batch (True),
    or to choose the best demonstrators for the batch holistically (False) when pretraining the agent.
    """

    adapt_morpho_omega: float = 0.0
    """The value of omega to use when adapting the morphology (PSO)."""

    pretrain_il_rewarder: bool = True
    """Whether to pretrain the IL rewarder after each morphology change using observations from the replay buffer."""


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

    device: Devices = "cuda"
    """Device to use."""

    env_name: str = MISSING
    """Name of the environment."""

    models_dir_path: str = "models"
    """Path to the directory where to save the models."""

    learn_disc_transitions: bool = False  # FIX: We are gonna have a problem with this
    """Learn discriminator using (s, s') transitions."""

    morpho_in_state: bool = True
    """Whether to include the morphology parameters in the state provided to the agent."""

    absorbing_state: bool = False  # FIX: We are gonna have a problem with this
    """Whether to use absorbing states."""

    resume: Optional[str] = None
    """Resume from given policy; specify the path + name of the .pt file to resume from."""

    storage_path: str = "./"
    """Path to the directory where to save large files."""


@dataclass(kw_only=True)
class TrainConfig(Config):
    """
    Configuration for training.
    """

    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"method": "cosil"},
            {"method/agent": "dual_sac"},
            {"method/rewarder": "sail"},
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

    method: MethodConfig = MISSING
    """Configuration for the method."""

    save_path: str = MISSING
    """Path to save the trajectories."""

    saved_agent_name: str = "agent"
    """Name of the agent to load in the saved dictionary model."""

    saved_morpho_name: str = "morphos"
    """Name of the morpho to load in the saved dictionary model."""

    num_trajectories: int = 10
    """Number of trajectories to generate."""


@dataclass(kw_only=True)
class GenBufferConfig(Config):
    """
    Configuration for generating observation buffers with a single morphology.
    """

    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {
                "method": "coil",
            },
        ]
    )

    task: str = "gen_buffer"
    """Name of the task."""

    method: MethodConfig = MISSING
    """Configuration for the method."""

    save_path: str = MISSING
    """Path to save the buffer."""

    num_agents: int = 1
    """Number of agents to train."""


@dataclass(kw_only=True)
class PretrainConfig(Config):
    """
    Configuration for generating observation buffers with a single morphology.
    """

    defaults: List[Any] = field(
        default_factory=lambda: [
            {
                "method/agent": "sac",
            },
            {
                "method/rewarder": "env",
            },
        ]
    )

    method: MethodConfig = CoILConfig()
    """Configuration for the method."""

    task: str = "pretrain"
    """Name of the task."""

    models_dir_path: str = "models/pretrained"
    """Path to the directory where to save the models."""

    updates: int = 50000
    """Number of updates to perform."""

    rewarder_batch_size: int = 256
    """Batch size for the rewarder."""

    batch_size: int = 256
    """Batch size for the agent."""


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_train", node=TrainConfig)
    cs.store(name="base_gen_obs", node=GenTrajectoriesConfig)
    cs.store(name="base_gen_buffer", node=GenBufferConfig)
    cs.store(name="base_halfcheetah_pretrain", node=PretrainConfig)
    cs.store(name="base_logger", node=LoggerConfig)
    cs.store(name="base_co_adaptation", node=CoAdaptationConfig)
    cs.store(group="method", name="base_rl", node=RLConfig)
    cs.store(group="method", name="base_coil", node=CoILConfig)
    cs.store(group="method", name="base_cosil", node=CoSILConfig)
    cs.store(group="method/agent", name="base_sac", node=SACConfig)
    cs.store(group="method/agent", name="base_dual_sac", node=DualSACConfig)
    cs.store(group="method/rewarder", name="base_env_rewarder", node=EnvRewarderConfig)
    cs.store(group="method/rewarder", name="base_mbc", node=MBCConfig)
    cs.store(group="method/rewarder", name="base_gail", node=GAILConfig)
    cs.store(group="method/rewarder", name="base_airl", node=AIRLConfig)
    cs.store(group="method/rewarder", name="base_sail", node=SAILConfig)
