def main()
    args = parse_args()
    
    if args.policy == "Gaussian":
        policy = GaussianPolicy(
            state_dim, action_dim, action_space, hidden_dim=256, init_w=3e-3
        )
    else: # Deterministic
        policy = DeterministicPolicy(
            state_dim, action_dim, action_space, hidden_dim=256, init_w=3e-3
        )

    if args.policy == "Gaussian":
        _, _, action, _ = self.policy.sample(state)
    else: # Deterministic
        _, _, action = self.policy.sample(state)
    


 def parse_args():
    parser = argparse.ArgumentParser(description="GAIL + SAC + co-adaptation")

    parser.add_argument(
        "--policy",
        default="Gaussian",
        help="Policy Type: Gaussian | Deterministic (default: Gaussian)",
    )

    return parser.parse_args()
   

if __name__ == "__main__":
    main()
