"""
  modal run blackwell/modal_run.py --task all
  modal run blackwell/modal_run.py --task check   # for confirming the GPU
  modal shell blackwell/modal_run.py              # live shell, iterate for free-ish
      # inside: cd /root/blackwell && python tma_copy.py
"""

from pathlib import Path

import modal

HERE = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("nvidia-cutlass-dsl==4.5.2", "cuda-python<13", "numpy")
    .add_local_dir(str(HERE), "/root/blackwell")
)

app = modal.App("cuTe")

GPU = "B200"


def _run(name: str):
    import subprocess
    print(f"\n>>> {name}")
    r = subprocess.run(["python", f"/root/blackwell/{name}"],
                       capture_output=True, text=True,check=False)
    out = r.stdout.strip()
    print(out if out else "(no stdout)")
    if r.returncode != 0:
        tail = [l for l in r.stderr.strip().splitlines() if l.strip()][-8:]
        print("FAILED (stderr tail):\n  " + "\n  ".join(tail))
    print("-" * 72)
    return r.returncode


@app.function(gpu=GPU, image=image, timeout=300)
def all_tests():
    """Everything in ONE container: one cold start, one image pull, one bill."""
    import torch
    print(f"device: {torch.cuda.get_device_name(0)}   capability: {torch.cuda.get_device_capability(0)}")
    print("=" * 72)
    _run("tma_copy.py")
    _run("matmul_tcgen05_sm100.py")
    _run("relu_cute.py")


@app.function(gpu=GPU, image=image, timeout=120)
def check():
    import torch
    print(f"device     : {torch.cuda.get_device_name(0)}")
    print(f"capability : {torch.cuda.get_device_capability(0)}   (want (10,0) for tcgen05)")
    import cutlass
    print("cutlass DSL:", getattr(cutlass, "__version__", "unknown"))
    print(f"torch      : {torch.__version__}, cuda {torch.version.cuda}")


@app.function(gpu=GPU, image=image, timeout=180)
def tma():
    _run("tma_copy.py")


@app.function(gpu=GPU, image=image, timeout=240)
def gemm():
    _run("matmul_tcgen05_sm100.py")


@app.function(gpu=GPU, image=image, timeout=240)
def gemm_min():
    _run("matmul_tcgen05_minimal.py")


@app.function(gpu=GPU, image=image, timeout=240)
def gemm_min_plainop():
    _run("matmul_tcgen05_minimal_plainop.py")


@app.function(gpu=GPU, image=image, timeout=180)
def relu():
    _run("relu_cute.py")


@app.local_entrypoint()
def main(task: str = "all"):
    tasks = {"all": all_tests, "check": check, "tma": tma, "gemm": gemm, "gemm_min": gemm_min, "gemm_min_plainop": gemm_min_plainop, "relu": relu}
    if task in tasks:
        tasks[task].remote()
    else:
        print(f"unknown task '{task}'. use: {' | '.join(tasks)}")
