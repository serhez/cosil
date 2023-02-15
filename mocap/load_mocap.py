# Adapted from https://github.com/uscresl/humanoid-gail
from asf_parser import AsfParser
from amc_parser import AmcParser
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import csv, os
from tqdm import tqdm


def normalize(vector, value=None):
    if value is None:
        value = norm(vector) + 1e-10
    return vector / value


def compute_features(positions, bones=["head", "lhand", "rhand", "lfoot", "rfoot"], base="root", norm_lengths=None):
    features = np.array([
        positions[b] - positions[base] for b in bones
    ])

    norm_features = np.array([
        normalize(features[i], norm_lengths[i]) for i in range(len(bones))
    ])

    return norm_features.flatten(), features.flatten()


def load_positions(asf_file=os.path.join(os.path.dirname(__file__), "examples/12.asf"),
                  amc_file=os.path.join(os.path.dirname(__file__), "examples/02_01.amc")):
    """Computes the end-effector positions over all animation frames of the AMC file."""
    parser = AsfParser()
    parser.parse(asf_file)
    amc = AmcParser()
    amc.parse(amc_file)
    skeleton = parser.skeleton
    positions = []
    for frame in tqdm(amc.frames):
        positions.append(skeleton.compute_motion(frame))
    return positions


def load_features(asf_file=os.path.join(os.path.dirname(__file__), "examples/12.asf"),
                  amc_file=os.path.join(os.path.dirname(__file__), "examples/02_01.amc"),
                  forward_vector_frames=0, frames_per_feature=1,
                  all_bones={}, struct=None, scale=0.076191007977626):
    """Computes the 5 3D vectors over all animation frames of the AMC file. Returns a Tx15 matrix."""
    parser = AsfParser()
    parser.parse(asf_file)
    amc = AmcParser()
    amc.parse(amc_file)
    skeleton = parser.skeleton
    features = [np.zeros(18)] * (frames_per_feature-1)
    norm_features = [np.zeros(18)] * (frames_per_feature-1)
    output_features, output_norm_features = [], []
    positions = [{"root": np.zeros((1, 3))}] * forward_vector_frames

    print("There are %i frames" % len(amc.frames))
    for i, frame in tqdm(enumerate(amc.frames)):
        # Compute forward-facing unit vector from the average of previous frames
        positions.append(skeleton.compute_motion(frame))

        #forward_vector = np.zeros((1, 3))
        #previous_pos = positions[-forward_vector_frames]["root"]
        #for j in range(-forward_vector_frames+1, 0):
        #    next_pos = positions[j]["root"]
        #    forward_vector += next_pos-previous_pos
        #    previous_pos = next_pos.copy()
        #forward_vector /= forward_vector_frames

        #pos = positions[i + forward_vector_frames]
        #norm_feat, feat = compute_features(pos, bones, base, norm_lengths)

        ##feat = np.hstack((feat, forward_vector.flatten()))
        #norm_features.append(norm_feat)
        #features.append(feat)
        #output_norm_features.append(np.array(norm_features[-frames_per_feature:]).flatten())
        #output_features.append(np.array(features[-frames_per_feature:]).flatten())

    nice_names = {"thorax": "torso", "root": "butt"}

    res = {}
    for il, markers in enumerate(all_bones):
        is_head = False
        base = markers["base"]
        bones = markers["bones"]
        # Get lengths needed for normalization
        norm_lengths = np.zeros(len(bones))
        for i, b in enumerate(bones):
            # The CMU demonstrators have only one head (no left/right)
            # so it requires some special treatment
            if "head" in b:
                side, qname = "", b
                path = struct[f"{base}_{qname}"]
                is_head = True
            else:
                side, qname = b[0], b[1:]
                path = struct[qname]
            lengths = [skeleton.bones[f"{side}{b}"]["length"] for b in path]
            norm_lengths[i] = np.sum(lengths)
        print(f"{norm_lengths=}")

        base_pos = np.stack([p[base] for p in positions])
        if is_head:
            assert len(bones) == 1
            norm_len = norm_lengths[0]
            marker_pos = np.stack([p["head"] for p in positions])
            rel_marker_pos = marker_pos - base_pos
            norm_marker_pos = rel_marker_pos/norm_len

            base_name = nice_names[base]
            res[f"track/abs/pos/head"] = marker_pos * scale
            res[f"track/rel/pos/head_wrt_{base_name}"] = rel_marker_pos * scale
            res[f"track/norm/pos/head_wrt_{base_name}"] = norm_marker_pos
        else:
            for im, m in enumerate(bones):
                norm_len = norm_lengths[im]+1e-8
                marker_pos = np.stack([p[m] for p in positions])
                rel_marker_pos = marker_pos - base_pos
                norm_marker_pos = rel_marker_pos/norm_len
                res[f"track/abs/pos/l{il}/m{im}"] = marker_pos * scale
                res[f"track/rel/pos/l{il}/m{im}"] = rel_marker_pos * scale
                res[f"track/norm/pos/l{il}/m{im}"] = norm_marker_pos

    # Add the root positions
    root_pos = np.stack([p["thorax"] for p in positions])
    res["track/abs/pos/torso_not_shifted"] = root_pos * scale
    # Shift to zero out the first one to start at origin
    #print(np.max(root_pos[:, 2]))
    #print(np.max(np.stack([p["head"] for p in positions])[:, 2]))
    print(root_pos[0]*scale)
    root_pos -= root_pos[0]
    res["track/abs/pos/torso"] = root_pos * scale

    # Make a 3d plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    bons = ["root", "lfemur", "rfemur", "rtibia", "ltibia", "lfoot", "rfoot",
            "lhumerus", "rhumerus", "lradius", "rradius", "lhand", "rhand"]
    cs = ["r", "g", "g", "b", "b", "y", "y", "magenta", "magenta",
          "black", "black", "cyan", "cyan"]

    #for b, c in zip(bons, cs):
    #    xdata = [p[b][0] for p in positions]
    #    ydata = [p[b][1] for p in positions]
    #    zdata = [p[b][2] for p in positions]
    #    ax.scatter3D(xdata, ydata, zdata, c=c)
    #plt.suptitle("Absolute poses")
    #plt.show()

    #clr = "rgb"

    #fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection='3d'))
    #for il, limb in enumerate(all_bones):
    #    for im, bn in enumerate(limb["bones"]):
    #        ax = axs[il//2, il%2]
    #        data = res[f"track/rel/pos/l{il}/m{im}"]
    #        dists = np.sqrt(np.sum(data**2, axis=1))
    #        print("Avg rel dist for", bn, "is", np.mean(dists), "with std of", np.std(dists))
    #        ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=clr[im])
    #        ax.set_title(bn)
    #plt.suptitle("Relative poses")
    #plt.show()

    #fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection='3d'))
    #for il, limb in enumerate(all_bones):
    #    for im, bn in enumerate(limb["bones"]):
    #        ax = axs[il//2, il%2]
    #        data = res[f"track/norm/pos/l{il}/m{im}"]
    #        dists = np.sqrt(np.sum(data**2, axis=1))
    #        print("Avg norm dist for", bn, "is", np.mean(dists), "with std of", np.std(dists))
    #        ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=clr[im])
    #        ax.set_title(bn)
    #plt.suptitle("Normalized poses")
    #plt.show()

    return res


def main():
    plt.figure(1)
    features = []
    with open('examples/web_features.csv') as csvfile:
        featreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for idx, frame in enumerate(featreader):
            features.append(frame)
        features = np.array(features)
    for i in range(15):
        plt.plot(list(float(x) for x in features[:, i]))

    plt.figure(2)
    parser = AsfParser()
    parser.parse("examples/12.asf")
    amc = AmcParser()
    amc.parse("examples/02_01.amc")
    skeleton = parser.skeleton
    features = []
    for frame in amc.frames:
        positions = skeleton.compute_motion(frame)
        features.append(compute_features(positions))
    features = np.array(features)
    for i in range(15):
        plt.plot(features[:, i])
    plt.show()


if __name__ == "__main__":
    main()
