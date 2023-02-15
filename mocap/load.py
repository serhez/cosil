# Adapted from https://github.com/uscresl/humanoid-gail
from load_mocap import load_features
import numpy as np
from argparse import ArgumentParser
import torch
from scipy.interpolate import PchipInterpolator

parser = ArgumentParser()
parser.add_argument("dir", type=str)
parser.add_argument("--sid", type=int, help="Subject/skeleton id")
parser.add_argument("--mid", type=int, help="Motion sequence id")
parser.add_argument("--freq", type=float, default=66.67, help="Data sampling frequency")
args = parser.parse_args()

asf_file = f"{args.dir}/{args.sid:02d}.asx"
amc_file = f"{args.dir}/{args.sid:02d}_{args.mid:02d}.amc"


def calculate_velocity(data, freq):
    vel_dict = {}
    for key, values in data.items():
        if "norm" in key:
            continue
        vel_key = key.replace("pos", "vel")
        ts = np.arange(values.shape[0])/freq
        res = np.zeros_like(values)
        for d in range(values.shape[-1]):
            pc = PchipInterpolator(ts, values[:, d])
            vels = pc(ts, 1)
            res[:, d] = vels
        vel_dict[vel_key] = res
    return data | vel_dict



#all_bones = [
#             {"base": "rfemur", "bones": ["rfemur", "rtibia", "rfoot",]},
#             {"base": "lfemur", "bones": ["lfemur", "ltibia", "lfoot",]},
#             {"base": "rhumerus", "bones": ["rhumerus", "rradius", "rhand", ]},
#             {"base": "lhumerus", "bones": ["lhumerus", "lradius", "lhand", ]},
#             # Head wrt torso
#             {"base": "thorax", "bones": ["head"]},
#             # Head wrt butt
#             {"base": "root", "bones": ["head"]},
#        ]
all_bones = [
             # Legs
             {"base": "rhipjoint", "bones": ["rhipjoint", "rfemur", "rtibia", "rfoot"]},
             {"base": "lhipjoint", "bones": ["lhipjoint", "lfemur", "ltibia", "lfoot"]},
             # Hands
             {"base": "rclavicle", "bones": ["rclavicle", "rhumerus", "rhand"]},
             {"base": "lclavicle", "bones": ["lclavicle", "lhumerus", "lhand"]},
             # Head wrt torso
             {"base": "thorax", "bones": ["head"]},
             # Head wrt butt
             {"base": "root", "bones": ["head"]},
        ]

struct = {# Hand structure
          "clavicle": [],
          "humerus": ["humerus", ],
          "radius": ["humerus", "radius"],
          "hand": ["humerus", "radius", "wrist", "hand"],

          # Leg structure
          "hipjoint": [],
          "femur": ["femur"],
          "tibia": ["femur", "tibia"],
          "foot": ["femur", "tibia", "foot"],

          # Torso and thorax to head
          "root_head": ["lowerback", "upperback", "thorax", "lowerneck",
              "upperneck", "head"],
          "thorax_head": ["thorax", "lowerneck", "upperneck", "head"]}

res = load_features(asf_file, amc_file, struct=struct, all_bones=all_bones)

# Add velocity data
res = calculate_velocity(res, args.freq)

fname = f"expert_cmu_{args.sid}_{args.mid}.pt"
torch.save(res, fname)

