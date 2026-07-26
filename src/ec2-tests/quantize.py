#!/usr/bin/env python3
"""JADE OS — Run the model quantization pipeline on the EC2 GPU box, by hand.

This is the manual driver for src/scripts/quantize-model.py (download Qwen3-Coder,
convert -> f16 GGUF, code-calibrated imatrix, quantize Q4_K_M) and publish to
the jade-os S3 contract:
  s3://jade-os-release-bucket/models/model.gguf (+ .gguf.sha256)

It: starts the box (src/ec2-tests/start.py), rsyncs THIS repo to
~/jade-inference-daemon, then runs src/scripts/quantize-model.py over SSH with the
CUDA toolchain on PATH and your AWS creds (from .env) forwarded so the upload
happens in-region from the box (no 18 GB round-trip through your laptop).

The job is long (60 GB download + convert + imatrix + quantize — hours) and the
box needs ~150 GB free disk; the underlying script preflights and fails fast.

Usage:
  python3 src/ec2-tests/quantize.py                 # start, sync, build, upload
  python3 src/ec2-tests/quantize.py --no-upload     # build + verify only (no S3)
  python3 src/ec2-tests/quantize.py --stop          # also stop the box when done
  python3 src/ec2-tests/quantize.py --no-start      # assume the box is already up
  python3 src/ec2-tests/quantize.py --detached      # run under setsid; survives SSH drop
  python3 src/ec2-tests/quantize.py --follow        # tail a detached run's log
  python3 src/ec2-tests/quantize.py -- --local-dir ~/merged   # pass-through args to
                                                # the pipeline (e.g. merged LoRA
                                                # weights already on the box)
"""
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _ec2 import (info, ok, die, load_env, aws_bin, resolve_host,
                  ssh_key, ssh_cmd, repo_root, env_required)

REMOTE_DIR = "jade-inference-daemon"
REMOTE_RUNNER = "quantize-run.py"
REMOTE_LOG = "quantize.log"


def parse_args(argv):
    opts = {"start": True, "stop": False, "upload": True,
            "detached": False, "follow": False, "pass_args": []}
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--no-start":
            opts["start"] = False
        elif a == "--stop":
            opts["stop"] = True
        elif a == "--no-upload":
            opts["upload"] = False
        elif a == "--detached":
            opts["detached"] = True
        elif a == "--follow":
            opts["follow"] = True
        elif a == "--":
            opts["pass_args"] = argv[i + 1:]
            break
        elif a in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        else:
            die(f"unknown arg: {a} (use -- to pass args through to quantize-model.py)")
        i += 1
    return opts


def build_runner(region, quant_args):
    """Generate the self-contained Python runner staged on the box."""
    aws_id = os.environ.get("AWS_ACCESS_KEY_ID", "")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    header = (
        "#!/usr/bin/env python3\n"
        "import os, sys, glob, subprocess\n"
        f"os.environ['AWS_ACCESS_KEY_ID'] = {aws_id!r}\n"
        f"os.environ['AWS_SECRET_ACCESS_KEY'] = {aws_secret!r}\n"
        f"os.environ['AWS_DEFAULT_REGION'] = {region!r}\n"
        f"REMOTE_DIR = {REMOTE_DIR!r}\n"
        f"QUANT_ARGS = {quant_args!r}\n"
    )
    # CUDA toolchain (box has 13.1) on PATH. We do NOT add the monorepo venv:
    # its python shebang points at a Mac path and would shadow a working aws;
    # quantize-model.py provisions its own aws.
    body = r'''
os.environ["PATH"] = ("/usr/local/cuda/bin:/usr/local/cuda-13.1/bin:"
                      + os.path.expanduser("~/.cargo/bin") + ":"
                      + os.environ.get("PATH", ""))
os.environ["LD_LIBRARY_PATH"] = ("/usr/local/cuda/lib64:/usr/local/cuda-13.1/lib64:"
                                 + os.environ.get("LD_LIBRARY_PATH", ""))

# --- scratch on the fast instance-store NVMe --------------------------------
# This box has only ~16 GB RAM, so the imatrix pass mmap-pages ~55 GB of f16
# expert weights from disk. That MUST be the local NVMe instance store (GB/s),
# not the EBS root (~125-250 MB/s). The instance store is EPHEMERAL (wiped on
# stop/start) — fine, it's pure scratch; we reformat each run. Only a disk whose
# hardware MODEL is "Instance Storage" is ever formatted.
SCRATCH = "/mnt/scratch"
if subprocess.run(["mountpoint", "-q", SCRATCH]).returncode != 0:
    dev = ""
    for d in sorted(glob.glob("/dev/nvme*n1")):
        model = subprocess.run(["lsblk", "-dno", "MODEL", d],
                               capture_output=True, text=True).stdout.strip()
        mnt = subprocess.run(["lsblk", "-no", "MOUNTPOINT", d],
                             capture_output=True, text=True).stdout.strip()
        if "Instance Storage" in model and not mnt:
            dev = d
            break
    if dev:
        print(f"==> formatting + mounting instance store {dev} at {SCRATCH} (ephemeral scratch)")
        subprocess.run(["sudo", "mkfs.ext4", "-F", "-q", dev], check=True)
        subprocess.run(["sudo", "mkdir", "-p", SCRATCH], check=True)
        subprocess.run(["sudo", "mount", dev, SCRATCH], check=True)
        subprocess.run(["sudo", "chown", f"{os.getuid()}:{os.getgid()}", SCRATCH], check=True)
    else:
        print("==> WARNING: no instance-store NVMe found — falling back to $HOME (EBS).")
        print("    The imatrix mmap pass will be slow on EBS; grow the root volume and")
        print("    ensure ~150 GB free, or attach instance storage.")
        SCRATCH = os.path.expanduser("~")

os.environ["JADE_QUANTIZE_WORKDIR"] = SCRATCH + "/jade-quantize"
print("==> workdir (scratch): " + os.environ["JADE_QUANTIZE_WORKDIR"])
subprocess.run(["df", "-h", SCRATCH])

workdir = os.path.expanduser("~/" + REMOTE_DIR)
rc = subprocess.run([sys.executable, "src/scripts/quantize-model.py"] + QUANT_ARGS,
                    cwd=workdir).returncode
print(f"JADE_QUANTIZE_DONE rc={rc}")
sys.exit(rc)
'''
    return header + body


def main(argv):
    opts = parse_args(argv)
    # --stop is meaningless with --detached (the job outlives this script).
    if opts["detached"] and opts["stop"]:
        die("--stop can't be combined with --detached")

    load_env()
    aws = aws_bin()

    # --- bring the box up + resolve host ------------------------------------
    if opts["start"]:
        subprocess.run([sys.executable, str(SCRIPT_DIR / "start.py")], check=True)
    host = resolve_host(aws)
    key = ssh_key()

    # --- sync this repo to the box ------------------------------------------
    root = repo_root()
    user = env_required("SSH_USER")
    info(f"syncing repo to {host}:~/{REMOTE_DIR}")
    subprocess.run(
        ["rsync", "-az", "--delete", "--progress",
         "--exclude", ".git/", "--exclude", "target/", "--exclude", "keys/",
         "--exclude", ".env", "--exclude", ".ec2.env", "--exclude", ".ec2-host",
         "--exclude", "dist/", "--exclude", "build/", "--exclude", "*.gguf",
         "-e", f"ssh -o StrictHostKeyChecking=no -i {key}",
         f"{root}/", f"{user}@{host}:~/{REMOTE_DIR}/"],
        check=True,
    )
    ok("synced")

    # --- credentials forwarded for the in-region S3 upload ------------------
    quant_args = []
    if opts["upload"]:
        if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
            die("AWS creds not set — add AWS_ACCESS_KEY (or AWS_ACCESS_KEY_ID) and "
                "AWS_SECRET_ACCESS_KEY to .ec2.env, or use --no-upload")
        quant_args.append("--upload")
    quant_args += opts["pass_args"]

    # --- stage the remote runner --------------------------------------------
    region = os.environ.get("AWS_REGION", "us-east-1")
    runner = build_runner(region, quant_args)
    info(f"staging remote runner ~/{REMOTE_RUNNER} on {host}")
    subprocess.run(ssh_cmd(host, remote=f"cat > ~/{REMOTE_RUNNER}"),
                   input=runner, text=True, check=True)

    if opts["detached"]:
        info(f"launching detached on {host} (logs to ~/{REMOTE_LOG})")
        # setsid + nohup so the job outlives this SSH session.
        subprocess.run(ssh_cmd(
            host,
            remote=(f"setsid nohup python3 ~/{REMOTE_RUNNER} "
                    f"> ~/{REMOTE_LOG} 2>&1 </dev/null & echo \"started pid $!\"")),
            check=True)
        print()
        ok(f"detached build started on {host}")
        print("    follow:  python3 src/ec2-tests/quantize.py --no-start --follow")
        print(f"    or:      python3 src/ec2-tests/ssh.py 'tail -f ~/{REMOTE_LOG}'")
        print("    done when the log shows: JADE_QUANTIZE_DONE rc=0")
        return 0

    if opts["follow"]:
        info(f"following ~/{REMOTE_LOG} on {host} (Ctrl-C to stop tailing; job keeps running)")
        cmd = ssh_cmd(host, remote=f"tail -n +1 -f ~/{REMOTE_LOG}")
        os.execvp(cmd[0], cmd)

    # --- synchronous run ----------------------------------------------------
    info(f"running quantize-model.py on {host} (this is the long part)")
    subprocess.run(ssh_cmd(host, remote=f"python3 ~/{REMOTE_RUNNER}"), check=True)
    ok("pipeline finished")

    if opts["upload"]:
        bucket = os.environ.get("S3_BUCKET", "jade-os-release-bucket")
        prefix = os.environ.get("S3_PREFIX", "models")
        print()
        ok(f"published s3://{bucket}/{prefix}/model.gguf (+ .sha256)")
        print("    jade-os CI will now bake this model into the next image build.")

    if opts["stop"]:
        info("stopping the box")
        subprocess.run([sys.executable, str(SCRIPT_DIR / "stop.py")], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
