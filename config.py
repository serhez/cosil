import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="(Dual) Co-Imitation Learning")

    parser.add_argument(
        "--num-agents",
        type=int,
        default=1,
        help="Number of agents to train",
    )
    parser.add_argument(
        "--env-name",
        default="GaitTrackHalfCheetah-v0",  # NOTE: Testing, original value was "GaitTrackHalfCheetahOriginal-v0"
        help="The Gym environment",
    )
    parser.add_argument(
        "--method",
        default="CoSIL",
        choices=["CoIL", "CoSIL"],
        help="Learning method",
    )
    parser.add_argument(
        "--agent",
        default="SAC",
        choices=["SAC"],
        help="Reinforcement learning agent algorithm",
    )
    parser.add_argument(
        "--rewarder",
        default="GAIL",
        choices=["env", "GAIL", "SAIL", "PWIL"],
        help="Reward function; if using a dual learner, this will be the imitation reward function",
    )
    parser.add_argument(
        "--rewarder-batch-size",
        type=int,
        default=64,
        help="The batch size used to train the rewarders",
    )
    parser.add_argument(
        "--expert-demos",
        type=str,
        # default="data/demonstrator/GaitTrackHalfCheetahOriginal-v0/expert.pt",
        default="data/expert_demos_sampled_GaitTrackHalfCheetah-v0.pt",
        help="Path to the expert demonstration file",
    )
    parser.add_argument(
        "--policy",
        default="Gaussian",
        choices=["Gaussian", "Deterministic"],
        help="Policy type",
    )
    parser.add_argument(
        "--eval",
        type=bool,
        default=True,
        help="Evaluates a policy every eval_per_episodes",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for reward",
    )
    parser.add_argument(
        "--target-entropy",
        type=str,
        default="auto",
        help="Target value for entropy",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Target smoothing coefficient(τ)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        help="Learning rate",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Temperature parameter α determines the relative importance of the entropy\
                                term against the reward",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1.0,
        help="The reward / Q-value weighting parameter, e.g.: omega * Q_value_1 + (1 - omega) * Q_value_2",
    )
    parser.add_argument(
        "--dual-mode",
        type=str,
        default="q",
        choices=["q", "reward"],
        help="Mode for dual learning",
    )
    parser.add_argument(
        "--dual-normalization",
        type=str,
        default="range",
        choices=["range", "z-score", "none"],
        help="Normalization for dual learning",
    )
    parser.add_argument(
        "--dual-normalization-mode",
        type=str,
        default="min",
        choices=["min", "mean"],
        help="Normalization mode for dual learning",
    )
    parser.add_argument(
        "--dual-normalization-gamma",
        type=float,
        default=100.0,
        help="The gamma parameter for the dual normalization.",
    )
    parser.add_argument(
        "--dual-normalization-beta",
        type=float,
        default=0.0,
        help="The beta parameter for the dual normalization.",
    )
    parser.add_argument(
        "--dual-normalization-low-clip",
        type=float,
        default=None,
        help="The clipping lower bound used when normalizing in dual learning",
    )
    parser.add_argument(
        "--dual-normalization-high-clip",
        type=float,
        default=None,
        help="The clipping higher bound used when normalizing in dual learning",
    )
    parser.add_argument(
        "--automatic-entropy-tuning",
        type=bool,
        default=True,
        help="Automaically adjust α",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
        help="Random seed",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=501,
        help="Maximum number of episodes",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="hidden size",
    )
    parser.add_argument(
        "--updates-per-step",
        type=int,
        default=1,
        help="model updates per simulator step",
    )
    parser.add_argument(
        "--start-steps",
        type=int,
        default=10000,
        help="Steps sampling random actions",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1,
        help="Value target update per no. of updates per step",
    )
    parser.add_argument(
        "--replay-size",
        type=int,
        default=2000000,
        help="size of replay buffer",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device used for PyTorch",
    )
    parser.add_argument(
        "--experiment-id",
        default="generic-ID",
        help="Replace this ID by an auto-generated one in your code (logging only)",
    )
    parser.add_argument(
        "--group-name", default="Group", help="Group name (logging only)"
    )
    parser.add_argument(
        "--project-name", default="CoSIL", help="Project name (logging only)"
    )
    parser.add_argument(
        "--log-scale-rewards",
        type=bool,
        default=False,
        help="Use sigmoid directly as reward or log of sigmoid",
    )
    parser.add_argument(
        "--reward-style",
        type=str,
        default="GAIL",
        choices=["GAIL", "AIRL"],
        help="Reward style",
    )
    parser.add_argument(
        "--train-every", type=int, default=1, help="Train every N timesteps"
    )
    parser.add_argument(
        "--explore-morpho-episodes",
        type=int,
        default=800,
        help="Episodes to run morphology exploration for",
    )
    parser.add_argument(
        "--morpho-warmup",
        type=int,
        default=60000,
        help="Steps before starting to optimize for morphology",
    )
    parser.add_argument(
        "--episodes-per-morpho",
        type=int,
        default=50,
        help="Episodes to run of each morphology",
    )
    parser.add_argument(
        "--disc-warmup",
        type=int,
        default=20000,
        help="Steps before starting to train SAC",
    )
    parser.add_argument(
        "--record-test", type=bool, default=False, help="Record tests (may be slow)"
    )
    parser.add_argument(
        "--load-warmup",
        type=bool,
        default=False,
        help="Load previously saved warmup data",
    )
    parser.add_argument(
        "--q-weight-decay", type=float, default=1e-5, help="Q-function weight decay"
    )
    parser.add_argument(
        "--disc-weight-decay",
        type=float,
        default=1e-5,
        help="Discriminator weight decay",
    )
    parser.add_argument(
        "--vae-scaler",
        type=float,
        default=1.0,
        help="Scaling term for VAE loss in SAIL",
    )
    parser.add_argument(
        "--pos-type",
        type=str,
        default="norm",
        choices=["abs", "rel", "norm", "skip"],
        help="Which position marker coordinate to use (absolute, relative, normalized-relative, or skip to omit it)",
    )
    parser.add_argument(
        "--vel-type",
        type=str,
        default="rel",
        choices=["abs", "rel", "norm", "skip"],
        help="Which velocity marker coordinate to use (absolute, relative, normalized-relative, or skip to omit it)",
    )
    parser.add_argument(
        "--expert-legs",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Which legs to use for marker matching on the demonstrator side",
    )
    parser.add_argument(
        "--policy-legs",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Which legs to use for marker matching on the imitator side",
    )
    parser.add_argument(
        "--expert-markers",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Which markers to use for matching on the demonstrator side",
    )
    parser.add_argument(
        "--policy-markers",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Which markers to use for matching on the imitator side",
    )
    parser.add_argument(
        "--learn-disc-transitions",
        type=bool,
        default=False,
        help="Learn discriminator using s, s' transitions",
    )
    parser.add_argument(
        "--train-distance-value",
        type=bool,
        default=False,
        help="Learn a separate distance value which is used to optimize morphology",
    )
    parser.add_argument(
        "--co-adapt",
        type=bool,
        default=True,  # NOTE: Testing; original value was False
        help="Adapt morphology as well as behaviour",
    )
    parser.add_argument(
        "--expert-env-name", type=str, default=None, help="Expert env name"
    )
    parser.add_argument(
        "--subject-id",
        type=int,
        default=8,
        help="Expert subject name when using CMU dataset",
    )
    parser.add_argument(
        "--expert-episode-length",
        type=int,
        default=300,
        help="Episode length for non-mocap expert data",
    )
    parser.add_argument(
        "--obs-save-path",
        type=str,
        default=None,
        help="Path to which to save the observations (gen_obs)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from given policy; specify the path + name of the .pt file to resume from",
    )
    parser.add_argument(
        "--torso-type",
        type=str,
        default=None,
        nargs="+",
        help="Use torso velocity, position or skip",
    )
    parser.add_argument(
        "--head-type",
        type=str,
        default=None,
        nargs="+",
        help="Use head velocity, position or skip",
    )
    parser.add_argument(
        "--head-wrt",
        type=str,
        default=None,
        nargs="+",
        help="Use head with respect to body part (torso, butt)",
    )
    parser.add_argument(
        "--absorbing-state",
        type=bool,
        default=False,
        help="Replace terminal states with special absorbing states",
    )
    parser.add_argument(
        "--omit-done",
        type=bool,
        default=False,
        help="Simply set done=False always for learning purposes. Alternative to absorbing states.",
    )
    parser.add_argument(
        "--save-morphos",
        type=bool,
        default=False,
        help="Save morphology parameters and corresponding Wasserstein distances for later",
    )
    parser.add_argument(
        "--dist-optimizer",
        default="PSO",
        choices=["BO", "CMA", "RS", "PSO"],
        help="Co-adapt for Wasserstein distance, and optimize using algo.",
    )
    parser.add_argument(
        "--bo-gp-mean", choices=["Zero", "Constant", "Linear"], default="Zero"
    )
    parser.add_argument(
        "--acq-weight",
        type=float,
        default=2.0,
        help="BO LCB acquisition function exploration weight",
    )
    parser.add_argument("--fixed-morpho", nargs="+", default=None, type=float)
    parser.add_argument(
        "--normalize-obs",
        type=bool,
        default=False,
        help="Normalize observations for critic",
    )
    parser.add_argument(
        "--save-checkpoints",
        type=bool,
        default=False,
        help="Save checkpoints for buffer and models",
    )
    parser.add_argument(
        "--save-optimal",
        type=bool,
        default=False,
        help="Save optimal buffer and models",
    )
    parser.add_argument(
        "--save-final",
        type=bool,
        default=True,
        help="Save the final buffer and models",
    )
    parser.add_argument(
        "--loggers",
        type=str,
        default="console,file",
        help="Loggers to report to, separated by commas",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate on",
    )
    parser.add_argument(
        "--eval-per-episodes",
        type=int,
        default=20,
        help="Number of episodes until a round of evaluation happens",
    )

    return parser.parse_args()


# NOTE: These are the args used originally by CoIL
# def parse_args():
#     parser = argparse.ArgumentParser(description="GAIL + SAC + co-adaptation")
#
#     parser.add_argument(
#         "--num-agents",
#         type=int,
#         default=1,
#         metavar="N",
#         help="number of agents to train",
#     )
#     parser.add_argument(
#         "--env-name",
#         default="GaitTrackHalfCheetahOriginal-v0",
#         help="Mujoco Gym environment",
#     )
#     parser.add_argument(
#         "--method",
#         default="CoIL",
#         help="Algorithm: CoIL or CoSIL or RL",
#     )
#     parser.add_argument("--agent", default="SAC", help="Algorithm: SAC")
#     parser.add_argument(
#         "--rewarder",
#         default="GAIL",
#         help="Reward function, either env or GAIL or SAIL or PWIL",
#     )
#     parser.add_argument(
#         "--rewarder-batch-size",
#         type=int,
#         default=64,
#         help="The batch size used to train the rewarders",
#     )
#     parser.add_argument(
#         "--expert-demos",
#         type=str,
#         # default="data/demonstrator/GaitTrackHalfCheetahOriginal-v0/expert.pt",
#         default="data/expert_demos_sampled_GaitTrackHalfCheetah-v0.pt",
#         help="Path to the expert demonstration file",
#     )
#     parser.add_argument(
#         "--policy",
#         default="Gaussian",
#         help="Policy Type: Gaussian | Deterministic",
#     )
#     parser.add_argument(
#         "--eval",
#         type=bool,
#         default=True,
#         help="Evaluates a policy every eval_per_episodes",
#     )
#     parser.add_argument(
#         "--gamma",
#         type=float,
#         default=0.99,
#         metavar="G",
#         help="Discount factor for reward",
#     )
#     parser.add_argument(
#         "--target_entropy",
#         type=str,
#         default="auto",
#         metavar="G",
#         help="Target value for entropy",
#     )
#     parser.add_argument(
#         "--tau",
#         type=float,
#         default=0.005,
#         metavar="G",
#         help="Target smoothing coefficient(τ)",
#     )
#     parser.add_argument(
#         "--lr",
#         type=float,
#         default=0.0003,
#         metavar="G",
#         help="Learning rate",
#     )
#     parser.add_argument(
#         "--alpha",
#         type=float,
#         default=0.2,
#         metavar="G",
#         help="Temperature parameter α determines the relative importance of the entropy\
#                                 term against the reward",
#     )
#     parser.add_argument(
#         "--omega",
#         type=float,
#         default=1.0,
#         metavar="G",
#         help="The reward / Q-value weighting parameter, e.g.: omega * Q_value_1 + (1 - omega) * Q_value_2",
#     )
#     parser.add_argument(
#         "--dual-mode",
#         type=str,
#         default="q",
#         metavar="G",
#         help="Mode for dual learning: 'q' (for q-value) or 'reward'",
#     )
#     parser.add_argument(
#         "--dual-normalization",
#         type=str,
#         default="range",
#         metavar="G",
#         help="Normalization for dual learning: 'range', 'z-score', or 'none'",
#     )
#     parser.add_argument(
#         "--dual-normalization-mode",
#         type=str,
#         default="min",
#         metavar="G",
#         help="Normalization mode for dual learning: 'min' or 'mean'",
#     )
#     parser.add_argument(
#         "--dual-normalization-low-clip",
#         type=float,
#         default=None,
#         metavar="G",
#         help="The clipping lower bound used when normalizing in dual learning",
#     )
#     parser.add_argument(
#         "--dual-normalization-high-clip",
#         type=float,
#         default=None,
#         metavar="G",
#         help="The clipping higher bound used when normalizing in dual learning",
#     )
#     parser.add_argument(
#         "--automatic_entropy_tuning",
#         type=bool,
#         default=True,
#         metavar="G",
#         help="Automaically adjust α",
#     )
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=123456,
#         metavar="N",
#         help="Random seed",
#     )
#     parser.add_argument(
#         "--batch-size",
#         type=int,
#         default=256,
#         metavar="N",
#         help="Batch size",
#     )
#     parser.add_argument(
#         "--num_steps",
#         type=int,
#         default=1000001,
#         metavar="N",
#         help="Maximum number of steps",
#     )
#     parser.add_argument(
#         "--hidden_size",
#         type=int,
#         default=256,
#         metavar="N",
#         help="hidden size",
#     )
#     parser.add_argument(
#         "--updates_per_step",
#         type=int,
#         default=1,
#         metavar="N",
#         help="model updates per simulator step",
#     )
#     parser.add_argument(
#         "--start_steps",
#         type=int,
#         default=10000,
#         metavar="N",
#         help="Steps sampling random actions",
#     )
#     parser.add_argument(
#         "--target_update_interval",
#         type=int,
#         default=1,
#         metavar="N",
#         help="Value target update per no. of updates per step",
#     )
#     parser.add_argument(
#         "--replay_size",
#         type=int,
#         default=2000000,
#         metavar="N",
#         help="size of replay buffer",
#     )
#     parser.add_argument(
#         "--device",
#         type=str,
#         choices=["cpu", "cuda", "mps"],
#         default="cpu",
#         help="Device used for PyTorch: cpu, cuda (CUDA-based GPU acceleration) or mps (Apple's Metal GPU acceleration)",
#     )
#     parser.add_argument(
#         "--experiment-name", default="Experiment", help="Experiment name (logging only)"
#     )
#     parser.add_argument(
#         "--group-name", default="Group", help="Group name (logging only)"
#     )
#     parser.add_argument(
#         "--project-name", default="CoSIL", help="Project name (logging only)"
#     )
#     parser.add_argument(
#         "--log-scale-rewards",
#         type=bool,
#         default=False,
#         help="Use sigmoid directly as reward or log of sigmoid",
#     )
#     parser.add_argument(
#         "--train-every", type=int, default=1, help="Train every N timesteps"
#     )
#     parser.add_argument(
#         "--explore-morpho-episodes",
#         type=int,
#         default=800,
#         help="Episodes to run morphology exploration for",
#     )
#     parser.add_argument(
#         "--morpho-warmup",
#         type=int,
#         default=60000,
#         help="Steps before starting to optimize for morphology",
#     )
#     parser.add_argument(
#         "--episodes-per-morpho",
#         type=int,
#         default=5,
#         help="Episodes to run of each morphology",
#     )
#     parser.add_argument(
#         "--disc-warmup",
#         type=int,
#         default=20000,
#         help="Steps before starting to train SAC",
#     )
#     parser.add_argument(
#         "--record-test", type=bool, default=False, help="Record tests (may be slow)"
#     )
#     parser.add_argument(
#         "--load-warmup",
#         type=bool,
#         default=False,
#         help="Load previously saved warmup data",
#     )
#     parser.add_argument(
#         "--q-weight-decay", type=float, default=1e-5, help="Q-function weight decay"
#     )
#     parser.add_argument(
#         "--disc-weight-decay",
#         type=float,
#         default=1e-5,
#         help="Discriminator weight decay",
#     )
#     parser.add_argument(
#         "--vae-scaler",
#         type=float,
#         default=1.0,
#         help="Scaling term for VAE loss in SAIL",
#     )
#     parser.add_argument(
#         "--pos-type",
#         type=str,
#         default="norm",
#         choices=["abs", "rel", "norm", "skip"],
#         help="Which position marker coordinate to use (absolute, relative, normalized-relative, or skip to omit it)",
#     )
#     parser.add_argument(
#         "--vel-type",
#         type=str,
#         default="rel",
#         choices=["abs", "rel", "norm", "skip"],
#         help="Which velocity marker coordinate to use (absolute, relative, normalized-relative, or skip to omit it)",
#     )
#     parser.add_argument(
#         "--expert-legs",
#         type=int,
#         nargs="+",
#         default=[0, 1],
#         help="Which legs to use for marker matching on the demonstrator side",
#     )
#     parser.add_argument(
#         "--policy-legs",
#         type=int,
#         nargs="+",
#         default=[0, 1],
#         help="Which legs to use for marker matching on the imitator side",
#     )
#     parser.add_argument(
#         "--expert-markers",
#         type=int,
#         nargs="+",
#         default=[1, 2, 3],
#         help="Which markers to use for matching on the demonstrator side",
#     )
#     parser.add_argument(
#         "--policy-markers",
#         type=int,
#         nargs="+",
#         default=[1, 2, 3],
#         help="Which markers to use for matching on the imitator side",
#     )
#     parser.add_argument(
#         "--learn-disc-transitions",
#         type=bool,
#         default=False,
#         help="Learn discriminator using s, s' transitions",
#     )
#     parser.add_argument(
#         "--train-distance-value",
#         type=bool,
#         default=False,
#         help="Learn a separate distance value which is used to optimize morphology",
#     )
#     parser.add_argument(
#         "--co-adapt",
#         type=bool,
#         default=False,
#         help="Adapt morphology as well as behaviour",
#     )
#     parser.add_argument(
#         "--expert-env-name", type=str, default=None, help="Expert env name"
#     )
#     parser.add_argument(
#         "--subject-id",
#         type=int,
#         default=8,
#         help="Expert subject name when using CMU dataset",
#     )
#     parser.add_argument(
#         "--expert-episode-length",
#         type=int,
#         default=300,
#         help="Episode length for non-mocap expert data",
#     )
#     parser.add_argument(
#         "--obs-save-path",
#         type=str,
#         default=None,
#         help="Path to which to save the observations (gen_obs)",
#     )
#     parser.add_argument(
#         "--resume",
#         type=str,
#         default=None,
#         help="Resume from given policy; specify the path + name of the .pt file to resume from",
#     )
#     parser.add_argument(
#         "--torso-type",
#         type=str,
#         default=None,
#         nargs="+",
#         help="Use torso velocity, position or skip",
#     )
#     parser.add_argument(
#         "--head-type",
#         type=str,
#         default=None,
#         nargs="+",
#         help="Use head velocity, position or skip",
#     )
#     parser.add_argument(
#         "--head-wrt",
#         type=str,
#         default=None,
#         nargs="+",
#         help="Use head with respect to body part (torso, butt)",
#     )
#     parser.add_argument(
#         "--absorbing-state",
#         type=bool,
#         default=False,
#         help="Replace terminal states with special absorbing states",
#     )
#     parser.add_argument(
#         "--omit-done",
#         type=bool,
#         default=False,
#         help="Simply set done=False always for learning purposes. Alternative to absorbing states.",
#     )
#     parser.add_argument(
#         "--save-morphos",
#         type=bool,
#         default=False,
#         help="Save morphology parameters and corresponding Wasserstein distances for later",
#     )
#     parser.add_argument(
#         "--dist-optimizer",
#         default="PSO",
#         choices=["BO", "CMA", "RS", "PSO"],
#         help="Co-adapt for Wasserstein distance, and optimize using algo.",
#     )
#     parser.add_argument(
#         "--bo-gp-mean", choices=["Zero", "Constant", "Linear"], default="Zero"
#     )
#     parser.add_argument(
#         "--acq-weight",
#         type=float,
#         default=2.0,
#         help="BO LCB acquisition function exploration weight",
#     )
#     parser.add_argument("--fixed-morpho", nargs="+", default=None, type=float)
#     parser.add_argument(
#         "--normalize-obs",
#         type=bool,
#         default=False,
#         help="Normalize observations for critic",
#     )
#     parser.add_argument(
#         "--save-checkpoints",
#         type=bool,
#         default=False,
#         help="Save checkpoints for buffer and models",
#     )
#     parser.add_argument(
#         "--save-optimal",
#         type=bool,
#         default=False,
#         help="Save optimal buffer and models",
#     )
#     parser.add_argument(
#         "--save-final",
#         type=bool,
#         default=True,
#         help="Save the final buffer and models",
#     )
#     parser.add_argument(
#         "--loggers",
#         type=str,
#         default="console",
#         help="Loggers to report to, separated by commas",
#     )
#     parser.add_argument(
#         "--eval-episodes",
#         type=int,
#         default=10,
#         help="Number of episodes to evaluate on",
#     )
#     parser.add_argument(
#         "--eval-per-episodes",
#         type=int,
#         default=20,
#         help="Number of episodes until a round of evaluation happens",
#     )
#
#     return parser.parse_args()
