#!/usr/bin/env python3
"""JADE OS — One-time EC2 instance setup.

Run this ONCE after provisioning. Idempotent — safe to re-run.
Installs: CUDA, LLVM 18, Rust toolchain, musl-tools, NVIDIA driver, and
downloads Qwen3-Coder-30B-A3B-Instruct-Q4_K_M to /jade/inference/model.gguf.

Run from your local machine:
  python3 src/ec2-tests/first-run.py

Or directly on the instance:
  python3 ~/jade-os/src/ec2-tests/first-run.py --local
"""
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/"
    "resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
)


def info(msg):
    print(f"==> {msg}")


def ok(msg):
    print(f"    OK: {msg}")


def warn(msg):
    print(f"    WARN: {msg}")


def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def sh(cmd, check=False, **kwargs):
    """Run a shell snippet through bash (for the system-mutation stanzas)."""
    return subprocess.run(cmd, shell=True, executable="/bin/bash", check=check, **kwargs)


def have(tool):
    return shutil.which(tool) is not None


# ── Remote dispatch (runs on your laptop) ────────────────────────────────────
def dispatch_remote():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _ec2 import load_env, aws_bin, resolve_host, ssh_key, ssh_user

    load_env()
    aws = aws_bin()
    host = resolve_host(aws)
    key = ssh_key()
    user = ssh_user()

    info(f"Uploading setup script to {host}...")
    subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no", "-i", key,
         str(REPO_ROOT / "src" / "ec2-tests" / "first-run.py"),
         f"{user}@{host}:~/ec2-first-run.py"],
        check=True,
    )

    info("Running first-run setup on instance (this takes ~10 minutes)...")
    subprocess.run(
        ["ssh", "-t", "-o", "StrictHostKeyChecking=no", "-i", key,
         f"{user}@{host}", "python3 ~/ec2-first-run.py --local"],
        check=True,
    )
    ok("First-run setup complete")


# ── On-box provisioning (runs on the EC2 instance) ───────────────────────────
def system_packages():
    info("Installing system packages...")
    os.environ["DEBIAN_FRONTEND"] = "noninteractive"
    sh("sudo apt-get update -qq")
    sh("sudo apt-get install -y -qq "
       "build-essential cmake curl git pkg-config "
       "musl-tools clang lld llvm "
       "libxml2-dev libzstd-dev "
       "gcc-13 g++-13 "
       f"linux-headers-$(uname -r) 2>/dev/null || true")

    # CUDA runfile expects libxml2.so.2 but recent Ubuntu ships libxml2.so.16
    if not Path("/usr/lib/x86_64-linux-gnu/libxml2.so.2").exists():
        libs = sorted(glob.glob("/usr/lib/x86_64-linux-gnu/libxml2.so.*"))
        if libs:
            sh(f"sudo ln -sf {libs[0]} /usr/lib/x86_64-linux-gnu/libxml2.so.2 && sudo ldconfig")
    ok("System packages installed")


def llvm18():
    if sh("dpkg -s llvm-18-dev >/dev/null 2>&1").returncode != 0:
        info("Installing LLVM 18 dev...")
        sh("sudo apt-get install -y -qq llvm-18-dev")
        ok("LLVM 18 dev installed")
    else:
        ok("LLVM 18 dev already installed")
    # llvm-sys 181 links Polly statically but Ubuntu ships only LLVMPolly.so;
    # create empty stub archives so the linker is satisfied.
    for stub in ("/usr/lib/llvm-18/lib/libPolly.a", "/usr/lib/llvm-18/lib/libPollyISL.a"):
        if not Path(stub).exists():
            sh(f"sudo ar crs {stub}")
    ok("Polly stub archives present")


def cuda():
    if not have("nvcc"):
        info("Installing CUDA 12 toolkit...")
        arch = "x86_64"
        ubuntu_ver = subprocess.run(["lsb_release", "-rs"], capture_output=True,
                                    text=True).stdout.strip().replace(".", "")
        distro = f"ubuntu{ubuntu_ver}"
        keyring_url = (f"https://developer.download.nvidia.com/compute/cuda/repos/"
                       f"{distro}/{arch}/cuda-keyring_1.1-1_all.deb")

        apt_ok = False
        if sh(f"wget -q --spider {keyring_url} 2>/dev/null").returncode == 0:
            sh(f"wget -q {keyring_url}")
            sh("sudo dpkg -i cuda-keyring_1.1-1_all.deb")
            sh("rm -f cuda-keyring_1.1-1_all.deb")
            sh("sudo apt-get update -qq")
            if sh("sudo apt-get install -y -qq cuda-toolkit-13-1 2>/dev/null").returncode == 0:
                apt_ok = True
                ok("CUDA 13.1 installed via apt")
            elif sh("sudo apt-get install -y -qq cuda-toolkit-13 2>/dev/null").returncode == 0:
                apt_ok = True
                ok("CUDA 13 installed via apt")
            else:
                warn(f"No cuda-toolkit in apt repo for {distro} — falling back to runfile")
        if not apt_ok:
            runfile = "cuda_12.6.3_560.35.05_linux.run"
            runfile_url = (f"https://developer.download.nvidia.com/compute/cuda/"
                           f"12.6.3/local_installers/{runfile}")
            # /tmp is tmpfs (RAM-backed); download and extract on disk instead.
            sh("sudo mkdir -p /var/tmp/cuda-install")
            sh(f"wget -q --show-progress -O /var/tmp/{runfile} {runfile_url}")
            sh(f"sudo TMPDIR=/var/tmp/cuda-install sh /var/tmp/{runfile} "
               "--silent --toolkit --override")
            sh(f"sudo rm -rf /var/tmp/{runfile} /var/tmp/cuda-install")
            ok("CUDA 12.6 installed via runfile")
    else:
        ver = subprocess.run("nvcc --version | head -1", shell=True,
                             capture_output=True, text=True).stdout.strip()
        ok(f"CUDA already installed: {ver}")

    # Ensure nvcc is in PATH for this session.
    os.environ["PATH"] = ("/usr/local/cuda-13.1/bin:/usr/local/cuda/bin:"
                          + os.environ.get("PATH", ""))
    os.environ["LD_LIBRARY_PATH"] = ("/usr/local/cuda-13.1/lib64:/usr/local/cuda/lib64:"
                                     + os.environ.get("LD_LIBRARY_PATH", ""))


def patch_cuda_math_headers():
    # CUDA 13.1 declares rsqrt/rsqrtf without noexcept; glibc 2.41 redeclares
    # them with noexcept — nvcc flags the mismatch. Add noexcept to CUDA's side.
    candidates = sorted(glob.glob(
        "/usr/local/cuda-*/targets/x86_64-linux/include/crt/math_functions.h"))
    if not candidates:
        return
    math_h = candidates[-1]
    try:
        text = Path(math_h).read_text()
    except Exception:
        return
    if "rsqrt(double x) noexcept" in text:
        ok("CUDA math_functions.h already patched")
        return
    for sym, repl in (("rsqrt(double x);", "rsqrt(double x) noexcept(true);"),
                      ("rsqrtf(float x);", "rsqrtf(float x) noexcept(true);")):
        # Patch the first non-comment declaration line, matching the old sed.
        for i, line in enumerate(text.splitlines(), start=1):
            if sym in line and "//" not in line:
                sh(f"sudo sed -i '{i}s/{re.escape(sym)}/{repl}/' {math_h}")
                break
    ok("Patched CUDA math_functions.h for glibc 2.41 noexcept compat")


def nouveau_blacklist():
    conf = Path("/etc/modprobe.d/blacklist-nouveau.conf")
    if not (conf.exists() and "blacklist nouveau" in conf.read_text()):
        info("Blacklisting nouveau driver...")
        sh("echo 'blacklist nouveau' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf > /dev/null")
        sh("echo 'options nouveau modeset=0' | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf > /dev/null")
        sh('sudo update-initramfs -u -k "$(uname -r)" 2>/dev/null | tail -1')
        ok("nouveau blacklisted")


def nvidia_driver():
    if not have("nvidia-smi"):
        info("Installing NVIDIA driver...")
        sh("sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-driver")
        ok("NVIDIA driver installed")

    # Unbind nouveau if still attached, then load nvidia.
    if sh("lspci -k 2>/dev/null | grep -q 'Kernel driver in use: nouveau'").returncode == 0:
        sh("PCIID=$(lspci | grep -i nvidia | awk '{print $1}' | head -1); "
           "echo \"0000:${PCIID}\" | sudo tee /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null || true")
        sh("sudo rmmod nouveau 2>/dev/null || true")
    sh("sudo modprobe nvidia 2>/dev/null || true")

    if sh("nvidia-smi >/dev/null 2>&1").returncode == 0:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
        ok(f"NVIDIA driver: {drv}")
    else:
        warn("nvidia-smi failed — NVIDIA driver may not be installed")
        warn("Run: sudo apt-get install -y nvidia-driver-535")
        warn("Then reboot and re-run this script")


def rust():
    if not have("cargo"):
        info("Installing Rust toolchain...")
        sh("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --quiet")
        ok("Rust installed")
    os.environ["PATH"] = str(Path.home() / ".cargo" / "bin") + ":" + os.environ.get("PATH", "")
    sh("rustup target add x86_64-unknown-linux-musl")
    ver = subprocess.run("rustc --version", shell=True, capture_output=True,
                         text=True).stdout.strip()
    ok(f"Rust: {ver}")


def download_model():
    (Path.home() / ".jade").mkdir(parents=True, exist_ok=True)
    ok(f"{Path.home() / '.jade'} directory ready")

    model_dest = Path(os.environ.get("JADE_MODEL_PATH", "/jade/inference/model.gguf"))
    sh(f"sudo mkdir -p {model_dest.parent}")
    if model_dest.is_file():
        sz = subprocess.run(["du", "-sh", str(model_dest)], capture_output=True,
                            text=True).stdout.split()[0]
        ok(f"Model already present: {sz}  {model_dest}")
    else:
        info("Downloading Qwen3-Coder-30B-A3B-Instruct-Q4_K_M (~18.6GB)...")
        sh(f"sudo curl -L --progress-bar -o {model_dest} {MODEL_URL}")
        sz = subprocess.run(["du", "-sh", str(model_dest)], capture_output=True,
                            text=True).stdout.split()[0]
        ok(f"Model downloaded: {sz}  {model_dest}")


def run_local():
    system_packages()
    llvm18()
    cuda()
    patch_cuda_math_headers()
    nouveau_blacklist()
    nvidia_driver()
    rust()
    download_model()
    print()
    info("First-run setup complete")
    info("Next: run the deploy step")


def main(argv):
    if "--local" in argv[1:]:
        run_local()
    else:
        dispatch_remote()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
