# Tools → Clone giọng nói (OmniVoice TTS)

Đọc một kịch bản thành **một file MP3 liền mạch** bằng **OmniVoice** chạy trên
GPU (khuyến nghị **L40S**), kèm:

- **Clone giọng** từ 5–15 giây ghi âm mẫu (zero-shot voice cloning), hoặc dùng
  giọng mặc định khoá theo `seed`.
- **Chỉnh tốc độ** (0.5–2.0×, giữ nguyên cao độ — không bị méo giọng).
- **Khử nhiễu** (highpass + FFT `afftdn`).
- **Xuất phụ đề `.SRT`** khớp đúng độ dài file MP3.

Kiến trúc **giống hệt API farm**: một server OmniVoice thường trú chạy trên GPU
node qua `sbatch`, ghi `endpoint.json` (`<node>:<port>`) ra FS chung; app login
node gọi sang để tổng hợp **từng đoạn**, rồi **tự ghép/chỉnh/khử nhiễu/xuất MP3
+ SRT bằng ffmpeg ngay trên login node** (không tốn GPU cho khâu này).

```
kịch bản ──(login node)──► chia đoạn ──HTTP──► OmniVoice server (GPU)
                                                   │ render từng đoạn → wav (FS chung)
        ◄── MP3 + SRT ◄── ffmpeg ghép/atempo/afftdn/loudnorm ◄──┘
```

## Thành phần

| Vai trò | File |
|---|---|
| Server TTS chạy trên GPU node | `scripts/omnivoice_server.py` |
| Vòng đời server (sbatch/monitor/auto-resubmit/resume) | `services/voice_service.py` |
| Chia đoạn + ffmpeg + SRT + clone profile + chạy job | `services/voice_pipeline.py` |
| Routes + UI | `app.py` (`/tools…`), `templates/tools.html`, `templates/tool_clone_voice.html` |
| Bảng DB | `voice_servers`, `voice_profiles`, `voice_jobs` (trong `storage_service.py`) |
| Cấu hình đường dẫn | `config.py` (`VOICE_*`, `OMNI_*`, `ffmpeg_path()`/`ffprobe_path()`) |

Mọi đường dẫn nằm trên FS chung dưới `DATA_DIR`:
`voice-servers/`, `voice-outputs/<job_id>/`, `voices/<slug>/`, `hf-cache/`.

## Chuẩn bị (làm 1 lần, trên LOGIN node — nơi có internet)

GPU node **không có internet**, nên model + thư viện phải tải sẵn ở login node
vào FS chung.

### 1. Tạo môi trường + cài thư viện (làm hoàn toàn trong web UI)

Phải tạo env **qua tab Môi trường** để nó được ghi vào DB → mới hiện trong
dropdown chọn server giọng nói. (Tạo venv tay bằng terminal sẽ KHÔNG hiện.)

1. **Tab Môi trường → "Tạo môi trường mới"**, đặt tên ví dụ `env-omnivoice` →
   Tạo. Chờ tới khi xong.
2. Chọn env đó → mục **"…hoặc cài từ file requirements.txt"** → **upload file
   `scripts/requirements-omnivoice.txt`** (đã có sẵn trong repo). Bấm cài và
   theo dõi log — lần đầu tải ~2–3GB (torch + cuda wheels), khá lâu.

> **Vì sao không gõ tên gói vào ô "cài gói"?** Ô đó validate từng tên gói nên
> **không nhận cờ** `--index-url`/`--extra-index-url`. File requirements thì
> được — pip đọc các dòng cờ ngay trong file. File đã ghim sẵn các phiên bản
> đúng (xem chú thích trong file): **torch 2.6.0+cu124** (khớp module
> `nvidia/cuda-12.4` của cluster; L40S = Ada sm_89) và **transformers==5.3.0**
> (ghim cứng: bản mới cần torch≥2.7 → crash khi import omnivoice với torch 2.6).
>
> torch cu124 đi kèm các wheel `nvidia-*-cu12` (libcudart/libcublas…); batch
> script tự thêm chúng vào `LD_LIBRARY_PATH` (`_cuda_lib_dirs`) nên **không cần
> `module load cuda`** (driver `libcuda.so.1` đã có sẵn trên GPU node). Card mới
> hơn (Blackwell 50xx) mới cần cu128 + torch 2.8 — với L40S thì cu124 là an toàn.

### 2. Tải weights OmniVoice vào HF cache chung (web terminal trên login node)

Compute node không có net nên weights phải nằm sẵn trong HF cache **trên FS
chung** đúng đường dẫn `config.OMNI_HF_HOME` (mặc định `DATA_DIR/hf-cache`).
Mở **Terminal** trong app rồi:

```bash
# Web terminal chặn cd/cat… vào thư mục nguồn — gỡ guard cho phiên này trước:
unset -f cd ls kill pkill killall cat tac head tail nl less more view vi vim nano strings od xxd bat

cd ~/LeeHoang_/ollama/app/host-a100
export HF_HOME="$PWD/host-a100-data/hf-cache"

# Dùng CHÍNH python của env vừa tạo (đã có huggingface_hub) để tải:
host-a100-data/envs/env-omnivoice/bin/python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('k2-fsa/OmniVoice')"

# Kiểm tra weights đã về:
ls -la host-a100-data/hf-cache/hub/
```

> `env-omnivoice` là TÊN env bạn đặt ở bước 1 (đường dẫn venv =
> `host-a100-data/envs/<tên>/bin/python`). Đổi model khác: đặt env
> `HOSTA100_OMNI_MODEL=<repo_id>` (hoặc sửa `config.OMNI_MODEL_ID`) rồi tải đúng
> repo đó vào cache này.

### 3. ffmpeg trên login node

Khâu ghép/chỉnh/khử nhiễu/xuất MP3 chạy bằng `ffmpeg`/`ffprobe` trên **login
node**. Kiểm tra:

```bash
which ffmpeg ffprobe   # có đường dẫn = xong, bỏ qua phần dưới
```

Nếu **chưa có**, cách chắc ăn nhất (không phụ thuộc `module`) là tải bản tĩnh về
thư mục `ffmpeg/` cạnh `app.py` — `config._find_bin` tự dò ở đó:

```bash
cd ~/LeeHoang_/ollama/app/host-a100
mkdir -p ffmpeg && cd ffmpeg
curl -L -o ff.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xf ff.tar.xz --strip-components=1 --wildcards '*/ffmpeg' '*/ffprobe'
chmod +x ffmpeg ffprobe && rm ff.tar.xz
./ffmpeg -version | head -1
```

(Hoặc nếu cluster có module: `module load ffmpeg` — nhưng phải load TRƯỚC khi
khởi động app thì tiến trình app mới thấy; bản tĩnh trong `ffmpeg/` thì luôn
thấy bất kể module.) Có thể đặt `HOSTA100_FFMPEG`/`HOSTA100_FFPROBE` trỏ đường
dẫn tuỳ ý. Trang tool sẽ hiện cảnh báo vàng nếu vẫn không tìm thấy. ffmpeg được
gọi với `-threads 0 -filter_complex_threads 0` để tận dụng **toàn bộ CPU**.

## Dùng

1. Vào tab **Công cụ → Clone giọng nói**.
2. Mục **Server giọng nói (GPU)** → chọn môi trường + Loại GPU (**L40S**) →
   **Khởi động**. Chờ trạng thái `chờ cấp GPU → đang nạp model → sẵn sàng`
   (lần đầu nạp vài GB weights).
3. (Tuỳ chọn) **Clone giọng mới**: tải file ghi âm mẫu 5–15s → đặt tên → Tạo
   giọng. File được chuẩn hoá (mono/24k/highpass/loudnorm) và lưu trên FS chung.
4. **Tạo giọng đọc**: dán kịch bản, chọn giọng (mặc định hoặc giọng clone),
   `num_step` (8 nhanh ↔ 32 nét), `seed` (khoá chất giọng zero-shot), tốc độ,
   bật/tắt khử nhiễu → **Tạo MP3**. Theo dõi tiến trình ở bảng **Tác vụ**, xong
   thì tải **MP3** và **SRT**.

## Hiệu năng

- **GPU (OmniVoice / L40S):** server bật `fp16` + `TF32` (`allow_tf32`,
  `set_float32_matmul_precision("high")`) + `cudnn.benchmark`. App gửi **cả
  kịch bản trong một request `/synthesize_batch`** nên GPU không nghỉ giữa các
  đoạn và prompt clone chỉ tính 1 lần (Whisper ASR + trích đặc trưng giọng được
  cache).
- **CPU (ffmpeg / login node):** `-threads 0 -filter_complex_threads 0` =
  dùng hết nhân.
- `num_step` là cần gạt tốc/độ nét rõ nhất: 8 cho nhanh, 16 cân bằng, 24–32 nét.

## Vận hành / lưu ý

- **Triển khai thay đổi code** = push → `git pull` trên server → **restart
  DETACHED app** (giống mọi thay đổi khác). `omnivoice_server.py` đi theo git
  nên `git pull` là đủ; **không cần** recreate server cho thay đổi phía login
  (pipeline/UI). Recreate server chỉ khi đổi model/env.
- **Auto-resubmit + circuit breaker**: hệt API farm — server chết liên tục
  <60s sau khi ready 3 lần → ngắt, báo `config.ALERT_WEBHOOK` (nếu đặt). Thường
  do thiếu weights trong HF cache hoặc OOM.
- **Health = `ready`**: `/health` chỉ báo `ready:true` sau khi model nạp xong
  (weights nạp SAU khi HTTP bind), nên request đầu không bị đua với lúc đang nạp.
- **Lỗi một đoạn không làm hỏng cả file**: đoạn nào server trả lỗi sẽ bị bỏ qua
  (ghi vào job.log), phần còn lại vẫn ghép thành MP3.

## Test

Không cần GPU/HPC để test phần login (chunk/SRT/atempo/ffmpeg), chỉ cần ffmpeg:

```bash
python3 tests/test_voice_pipeline.py
```
