import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="GAIL + SAC + co-adaptation")

    parser.add_argument(
        "--num-agents",
        type=int,
        default=1,
        metavar="N",
        help="number of agents to train (default: 1)",
    )
    parser.add_argument(
        "--env-name",
        default="GaitTrackHalfCheetahOriginal-v0",
        help="Mujoco Gym environment",
    )
    parser.add_argument(
        "--method",
        default="CoIL",
        help="Algorithm: CoIL or CoSIL or RL (default: CoIL)",
    )
    parser.add_argument("--agent", default="SAC", help="Algorithm: SAC")
    parser.add_argument(
        "--rewarder",
        default="GAIL",
        help="Reward function, either env or GAIL or SAIL or PWIL (default: GAIL)",
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
        help="Policy Type: Gaussian | Deterministic (default: Gaussian)",
    )
    parser.add_argument(
        "--eval",
        type=bool,
        default=True,
        help="Evaluates a policy every eval_per_episodes (default: True)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="Discount factor for reward (default: 0.99)",
    )
    parser.add_argument(
        "--target_entropy",
        type=str,
        default="auto",
        metavar="G",
        help="Target value for entropy",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        metavar="G",
        help="Target smoothing coefficient(τ) (default: 0.005)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        metavar="G",
        help="Learning rate (default: 0.0003)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        metavar="G",
        help="Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)",
    )
    parser.add_argument(
        "--automatic_entropy_tuning",
        type=bool,
        default=True,
        metavar="G",
        help="Automaically adjust α (default: False)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
        metavar="N",
        help="Random seed (default: 123456)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000001,
        metavar="N",
        help="Maximum number of steps (default: 1000000)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        metavar="N",
        help="hidden size (default: 256)",
    )
    parser.add_argument(
        "--updates_per_step",
        type=int,
        default=1,
        metavar="N",
        help="model updates per simulator step (default: 1)",
    )
    parser.add_argument(
        "--start_steps",
        type=int,
        default=10000,
        metavar="N",
        help="Steps sampling random actions (default: 10000)",
    )
    parser.add_argument(
        "--target_update_interval",
        type=int,
        default=1,
        metavar="N",
        help="Value target update per no. of updates per step (default: 1)",
    )
    parser.add_argument(
        "--replay_size",
        type=int,
        default=2000000,
        metavar="N",
        help="size of replay buffer (default: 20000000)",
    )
    parser.add_argument(
        "--cuda", type=bool, default=False, help="run on CUDA (default: False)"
    )
    parser.add_argument("--run-name", default="test", help="Run name (logging only)")
    parser.add_argument(
        "--project-name", default="cosil", help="Run name (logging only)"
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
        help="AIRL-style or GAIL-style reward",
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
        default=5,
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
        default=False,
        help="Adapt morphology as well as behaviour (default: False)",
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
        default=None,
        choices=["BO", "CMA", "RS"],
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
        help="Save checkpoints for buffer and models (default: false)",
    )
    parser.add_argument(
        "--save-optimal",
        type=bool,
        default=False,
        help="Save optimal buffer and models (default: true)",
    )
    parser.add_argument(
        "--save-final",
        type=bool,
        default=True,
        help="Save the final buffer and models (default: true)",
    )
    parser.add_argument(
        "--use-wandb",
        type=bool,
        default=True,
        help="Record logs to Weights & Biases (default: true)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate on (default: 10)",
    )
    parser.add_argument(
        "--eval-per-episodes",
        type=int,
        default=20,
        help="Number of episodes until a round of evaluation happens (default: 20)",
    )

    return parser.parse_args()
