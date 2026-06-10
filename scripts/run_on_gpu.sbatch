#!/bin/bash
# ---------------------------------------------------------------------------
# Chạy host-a100 trên một node GPU của cụm SLURM.
#
# Node đăng nhập (hpc-head1) KHÔNG có GPU — GPU nằm ở các compute node và phải
# xin qua SLURM. GPU được gắn nhãn theo kiến trúc, KHÔNG phải tên thương mại:
#     ampere   -> A100 / A30 / A40   (đây là "A100")
#     hopper   -> H100
#     lovelace -> RTX 40xx / L40
#     turing   -> T4 / RTX 20xx
#     blackwell-> B200 / RTX 50xx
#
# Dùng:
#     sbatch scripts/run_on_gpu.sbatch
#     squeue -u $USER                 # xem job, cột NODELIST = node được cấp
#     tail -f host-a100-<jobid>.log   # xem log (in ra node + cổng)
#
# Mở web UI từ laptop (thay <node> bằng NODELIST ở trên):
#     ssh -L 8198:<node>:8198 s3002152@hpc-head1
#     # rồi mở http://localhost:8198
# ---------------------------------------------------------------------------
#SBATCH --job-name=host-a100
#SBATCH --gres=gpu:ampere:1        # 1 GPU A100 (ampere). Đổi nhãn nếu cần.
#SBATCH --time=04:00:00            # thời lượng tối đa; tăng/giảm tuỳ nhu cầu
#SBATCH --output=host-a100-%j.log  # %j = job id

set -e
cd "$HOME/LeeHoang_/ollama/app/host-a100"

echo "=== Node: $(hostname) ==="
nvidia-smi || echo "CẢNH BÁO: nvidia-smi không chạy được trên node này."

# Mở web UI bằng SSH tunnel tới node ở trên (xem hướng dẫn ở đầu file).
python3 app.py
