import json, os, sys
import numpy as np

N, FPS, SPEED = 481, 16, 0.05
K = np.array([502.9, 503.1, 416.0, 240.0], np.float32)


def ease(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def yaw_mat(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def path(yaw, move):
    c2w = np.tile(np.eye(4), (N, 1, 1))
    pos = np.zeros(3)
    for i in range(N):
        r = yaw_mat(yaw[i])
        if i:
            pos = pos + r @ move[i]
        c2w[i, :3, :3], c2w[i, :3, 3] = r, pos
    return c2w.astype(np.float32)


def scripts():
    f = np.arange(N, dtype=np.float64)
    fwd = np.tile([0, 0, SPEED], (N, 1))
    t = f / FPS
    turn = np.radians(180) * (ease((t - 3.0) / 3.0) - ease((t - 12.0) / 3.0))
    side = np.tile([SPEED, 0, 0], (N, 1))
    return {
        "A_push": (np.zeros(N), fwd, "straight forward, constant speed"),
        "B_arc": (-np.radians(6.0) * t, fwd, "forward with a slow left curve (6 deg/s)"),
        "C_lookback": (-turn, fwd * 0.6, "forward 0-3s, ease-turn 180 deg left 3-6s, hold 6-12s (6s > 4.5s window), turn back 12-15s, forward"),
        "D_strafe": (np.zeros(N), side, "sideways to the right, camera facing forward, constant speed"),
    }


def main(out):
    for name, (yaw, move, desc) in scripts().items():
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "poses.npy"), path(yaw, move))
        np.save(os.path.join(d, "intrinsics.npy"), np.tile(K, (N, 1)))
        json.dump({"script": name, "movement": desc, "frames": N, "fps": FPS, "speed_per_frame": SPEED,
                   "convention": "opencv c2w, x right, y down, z forward; yaw<0 turns left"}, open(os.path.join(d, "script.json"), "w"), indent=1)
        print(name, desc)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cams")
