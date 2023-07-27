import torch


def main():
    model = torch.load(
        "models/final/CoSIL/pretrain/GaitTrackHalfCheetah-v0/seed-123456/dualsac-gail-pretrain_1689060342.pt",
        map_location=torch.device("cpu"),
    )

    buffer = model["replay_buffer"]
    morphos = model["morphos"]
    data = {
        "buffer": buffer,
        "morphos": morphos,
    }

    torch.save(
        data,
        "data/replay_buffers/pretrain/GaitTrackHalfCheetah-v0/seed-123456/dualsac-gail-pretrain_1689060342.pt",
    )


if __name__ == "__main__":
    main()
