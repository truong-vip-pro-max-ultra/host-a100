#!/bin/bash
# ---------------------------------------------------------------------------
# Chạy host-a100 trên một node GPU của cụm SLURM.
#
# Node đăng nhập (hpc-head1) KHÔNG có GPU — GPU nằm ở các compute node và phải
# xin qua SLURM. GRES được gắn nhãn theo KIẾN TRÚC (họ chip), KHÔNG phải model:
#     ampere   -> A100 HOẶC A40 (và A4500)   <-- KHÔNG đảm bảo ra A100!
#     hopper   -> H200
#     lovelace -> L40 / L40S
#     turing   -> RTX-6000 / T4 / RTX-2080Ti
#     blackwell-> RTX 6000 Pro
# Model thật nằm ở node FEATURE. Muốn ĐÚNG A100 phải thêm constraint:
#     --gres=gpu:ampere:1 --constraint=a100   (chỉ node ctit089)
# Xem model + số GPU trống mỗi loại:  sinfo -N -o "%n|%G|%f|%T"
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
#SBATCH --gres=gpu:ampere:1        # 1 GPU họ ampere (A100/A40). Thêm dòng dưới để ép A100:
##SBATCH --constraint=a100         # bỏ ## để chỉ chạy trên node A100 (ctit089)
#SBATCH --time=04:00:00            # thời lượng tối đa; tăng/giảm tuỳ nhu cầu
#SBATCH --output=host-a100-%j.log  # %j = job id

set -e
cd "$HOME/LeeHoang_/ollama/app/host-a100"

echo "=== Node: $(hostname) ==="
nvidia-smi || echo "CẢNH BÁO: nvidia-smi không chạy được trên node này."

# Mở web UI bằng SSH tunnel tới node ở trên (xem hướng dẫn ở đầu file).
python3 app.py
