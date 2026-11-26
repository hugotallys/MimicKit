"""
Play one Go2 reference clip in the viewer, chosen by name.

The clip is set through a copy of data/envs/view_motion_go2_env.yaml with motion_file replaced,
written to output/view_clip_env.yaml, so MimicKit's code and configs stay untouched.

Usage (from MimicKit/, or through `make view CLIP=trot`):
  python tools/view_clip.py --clip trot
  python tools/view_clip.py --clip go2_walk2 --engine_config data/engines/newton_engine.yaml --devices cpu

--clip takes a short name (walk0..walk3, pace, trot, run), the file name (go2_trot) or a .pkl path.
Any argument it does not know is passed on to mimickit/run.py.
"""

import argparse
import glob
import os
import re
import subprocess
import sys

MOTION_DIR = "data/motions/go2"
BASE_ENV = "data/envs/view_motion_go2_env.yaml"
ARG_FILE = "args/view_motion_go2_args.txt"
TMP_ENV = "output/view_clip_env.yaml"


def clip_path(clip):
    if clip.endswith(".pkl") or "/" in clip or "\\" in clip:
        return clip
    name = clip if clip.startswith("go2_") else "go2_" + clip
    return "{}/{}.pkl".format(MOTION_DIR, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="pace")
    ap.add_argument("--engine_config", default="data/engines/newton_engine.yaml")
    args, extra = ap.parse_known_args()

    path = clip_path(args.clip)
    if not os.path.exists(path):
        clips = sorted(os.path.splitext(os.path.basename(p))[0].replace("go2_", "")
                       for p in glob.glob(os.path.join(MOTION_DIR, "*.pkl")))
        print("Clip not found: {}\nAvailable: {}".format(path, ", ".join(clips)))
        sys.exit(1)

    with open(BASE_ENV) as f:
        cfg = f.read()
    cfg = re.sub(r"^motion_file:.*$", 'motion_file: "{}"'.format(path.replace("\\", "/")), cfg, flags=re.M)
    os.makedirs(os.path.dirname(TMP_ENV), exist_ok=True)
    with open(TMP_ENV, "w") as f:
        f.write(cfg)

    print("Playing", path)
    cmd = [sys.executable, "mimickit/run.py", "--arg_file", ARG_FILE, "--env_config", TMP_ENV,
           "--engine_config", args.engine_config, "--visualize", "true"] + extra
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
