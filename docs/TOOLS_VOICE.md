# Tools → Clone giọng nói (OmniVoice TTS)

Đọc một kịch bản thành **một file MP3 liền mạch** bằng **OmniVoice** chạy trên
GPU (khuyến nghị **L40S**), kèm:

- **Clone giọng** từ 5–15 giây ghi âm mẫu (zero-shot voice cloning), hoặc dùng
  giọng mặc định khoá theo `seed`.
- **Chỉnh tốc độ** (0.5–2.0×, giữ nguyên cao độ — không bị méo giọng).
- **Tự khử nhiễu "tạch" đầu/cuối mỗi đoạn** (strip onset blip + trim mép).
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
> script tự dò chúng (`nvidia/*/lib`) rồi thêm vào `LD_LIBRARY_PATH` ngay trên
> node nên **không cần `module load cuda`** (driver `libcuda.so.1` đã có sẵn trên
> GPU node). Card mới hơn (Blackwell 50xx) mới cần cu128 + torch 2.8 — với L40S
> thì cu124 là an toàn.

### Tăng tốc nạp model: stage venv về ổ local của node (QUAN TRỌNG)

**Triệu chứng "server nạp model mãi không xong / cứ ở loading":** không phải treo,
mà là `import torch/omnivoice` **từ venv nằm trên FS chung (NFS) quá chậm**. Đo
thực tế trên cluster này: import từ NFS mất **~9 phút** (torch ~2 phút + omnivoice
~9 phút do "bão `stat()`" hàng nghìn file nhỏ), trong khi chạy ĐÚNG venv đó từ ổ
**local của node (`/tmp`)** chỉ mất **~4 giây** — nhanh hơn ~100 lần. Node A40 NFS
còn chậm hơn nên dễ tưởng là treo.

`run.sh` (do `voice_service` sinh ra) vì vậy **giải nén một tarball của venv ra ổ
local của node rồi chạy python từ đó** (`config.OMNI_STAGE_LOCAL`, mặc định bật):

- Đọc **1 file nén tuần tự** → né được "bão stat" của NFS. Tarball nằm **cạnh thư
  mục env**: `host-a100-data/envs/<env>.tar.gz` (ưu tiên `.tar.zst` > `.tar.gz`/
  `.tgz` > `.tar`). Phải nén để giảm số byte đọc qua NFS (NFS ở đây ~12 MB/s) —
  bản chưa nén 5.7GB đọc mất ~8 phút, gần như không lợi.
- Bản đã stage được **tái sử dụng** (khoá theo size+mtime của tarball) nên
  restart/auto-resubmit trên cùng node là **tức thì**; các bản cũ (khác chữ ký)
  bị dọn trước để chỉ giữ 1 bản/env/node, không đầy `/tmp`.
- **Tự fallback về python trên NFS** (chạy được, chỉ chậm) nếu thiếu tarball,
  thiếu `zstd`, hay giải nén lỗi → không bao giờ vỡ.

**Tạo tarball (làm 1 lần, trên LOGIN node):** gói + nén thư mục env. `tar` thuần
chịu "bão stat" 1 lần ở nơi nhanh (login), `gzip`/`pigz` đọc tuần tự để nén:

```bash
cd ~/LeeHoang_/ollama/app/host-a100/host-a100-data/envs
tar -cf env-omnivoice.tar env-omnivoice        # (chậm 1 lần — đọc nguội cả env)
command -v pigz >/dev/null && pigz -p4 env-omnivoice.tar || gzip env-omnivoice.tar
ls -lh env-omnivoice.tar.gz                     # ~2–3GB
```

Sau khi pip cài thêm gói vào env, app tự nhận tarball **cũ hơn env** và **build lại
ở nền** (gzip/pigz); lần khởi động đó dùng tạm bản cũ (hoặc NFS nếu chưa có), lần
sau dùng bản mới. Muốn tắt hẳn: `HOSTA100_OMNI_STAGE_LOCAL=0`.

### 2. Cache weights OmniVoice (chạy script warmup trên login node)

Compute node không có net và server chạy `HF_HUB_OFFLINE=1`, nên **mọi** repo
phải nằm sẵn trong HF cache trên FS chung (`config.OMNI_HF_HOME`, mặc định
`DATA_DIR/hf-cache`). **`snapshot_download('k2-fsa/OmniVoice')` là CHƯA ĐỦ** —
OmniVoice còn tải lười (lazy) thêm repo lúc chạy: một vocoder/aux ở lần generate
đầu, **và một model Whisper ASR khi bạn clone giọng mà KHÔNG điền lời thoại mẫu**
(nó tự nhận dạng). Mấy repo đó không vào cache qua snapshot_download → server
offline lỗi đúng các đường đó (`Cannot find an appropriate cached snapshot … offline`).

Dùng script `scripts/warmup_omnivoice.py` (đi theo git) — nó chạy lần lượt cả 3
đường để kéo đủ mọi repo. Mở **Terminal** trong app:

```bash
# Web terminal chặn cd/cat… vào thư mục nguồn — gỡ guard cho phiên này trước:
unset -f cd ls kill pkill killall cat tac head tail nl less more view vi vim nano strings od xxd bat

cd ~/LeeHoang_/ollama/app/host-a100
HF_HOME="$PWD/host-a100-data/hf-cache" HF_HUB_OFFLINE=0 \
    host-a100-data/envs/env-omnivoice/bin/python scripts/warmup_omnivoice.py

# Kiểm tra cache (phải thấy cả 1 thư mục models--…whisper… cho clone):
ls -la host-a100-data/hf-cache/hub/
```

> Script chạy trên CPU nên chậm và **có thể báo lỗi ở cuối** — không sao, miễn là
> phần TẢI đã xong (mỗi bước được chạy theo thứ tự để kéo đủ repo). `env-omnivoice`
> là tên env bạn đặt ở bước 1. Đổi model khác: đặt `HOSTA100_OMNI_MODEL=<repo_id>`
> (hoặc sửa `config.OMNI_MODEL_ID`) rồi chạy lại.
>
> **Cách né ASR hoàn toàn:** khi clone giọng, ĐIỀN ô "lời thoại mẫu" đúng nội dung
> file ghi âm → OmniVoice không cần Whisper, chạy offline được ngay cả khi chưa
> cache ASR.

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
2. Mục **Server giọng nói (GPU)** → chọn môi trường + Loại GPU. **Để "Bất kỳ GPU
   rảnh" (mặc định)** để được cấp nhanh nhất — chỉ ghim L40S nếu cần tốc độ tối đa
   và chấp nhận chờ. Bấm **Khởi động**, chờ `chờ cấp GPU → đang nạp model → sẵn
   sàng` (lần đầu nạp vài GB weights).
3. (Tuỳ chọn) **Clone giọng mới**: tải file ghi âm mẫu 5–15s → đặt tên → **nên
   ĐIỀN ô "lời thoại mẫu"** (đúng nội dung file) để khỏi cần Whisper ASR (chạy
   offline ngay) → Tạo giọng. File được chuẩn hoá (mono/24k/highpass/loudnorm).
4. **Tạo giọng đọc**: dán kịch bản (văn bản CÓ NGHĨA — chuỗi vô nghĩa kiểu "Yyyy
   Yyyy…" sẽ ra audio rỗng), chọn giọng, `num_step` (8 nhanh ↔ 32 nét), `seed`
   (khoá chất giọng zero-shot), tốc độ, **"Số đoạn xử lý cùng lúc (batch GPU)"**
   (tăng để vắt VRAM L40S, đọc nhanh hơn) → **Tạo MP3**. Nhiễu "tạch" đầu/cuối
   mỗi đoạn **đã tự khử** (không có nút khử nhiễu — audio TTS vốn sạch). Theo dõi
   ở bảng **Tác vụ**, xong thì **▶ nghe thử** / tải **MP3** / tải **SRT**.

### Kho theo người dùng (username)

Bảng **Tác vụ giọng đọc** có ô **"Tên người dùng"** chia tác vụ theo *namespace*
(giống tab Gen video — xem `docs/TOOLS_VIDEO.md`):

- **Để trống** → xem & tạo vào **kho công khai** (`owner=''`).
- **Nhập tên** → chỉ thấy và tạo vào kho riêng của tên đó; **chỉ ai nhập đúng tên
  mới thấy**. Đây là namespace để tổ chức, **không phải xác thực** (cả app đã nằm
  sau login chủ sở hữu).

Cột `voice_jobs.owner` (mặc định `''`); `voice_pipeline.list_jobs(owner)` lọc
`owner=?` (hoặc `COALESCE(owner,'')=''` cho public), `start_job(owner=…)` đóng dấu
khi tạo. Tên trim + cap 64 ký tự (`normalize_owner`), lưu localStorage, và prefill
được từ `?u=<tên>` (link từ trang ẩn `/users`). Bảng tác vụ tự reconcile theo
`jobs.json?username=` nên đổi tên là đổi danh sách ngay; "Xoá tất cả" chỉ xoá trong
kho đang xem.

**Trang ẩn `/users`:** liệt kê mọi username đang có ở **cả 2 tab** (giọng + video)
kèm số lượng mỗi loại, có ô lọc tên và nút mở nhanh sang đúng kho. **Không** xuất
hiện trong menu — chỉ ai biết URL mới vào (vẫn sau login chủ sở hữu). Route
`users_page` trong `app.py`, gộp `voice_pipeline.list_owners()` +
`video_pipeline.list_owners()`; template `templates/users.html`.

### Kiểu giọng (thiết kế giọng — voice design)

OmniVoice có chế độ **voice design**: thay vì clone từ file mẫu, model tự "thiết
kế" một giọng theo **bộ thuộc tính cố định**. Trên form Tạo giọng đọc, mở mục
**"Kiểu giọng (thiết kế giọng)"** và chọn:

- **Giới tính:** Nữ / Nam
- **Độ tuổi:** trẻ em / thiếu niên / thanh niên / trung niên / lớn tuổi
- **Cao độ:** rất trầm → rất cao
- **Thì thầm (whisper)**

> ⚠️ Đây **không** phải prompt tự do. OmniVoice chỉ nhận đúng bộ từ vựng trên
> (gender / age / pitch / whisper); mô tả kiểu "đọc vui vẻ, hào hứng, dồn dập" sẽ
> bị model từ chối. (Accent american/british… cũng có nhưng **ép sang tiếng Anh**
> nên không hợp narration tiếng Việt → không đưa vào UI.)

Cách hoạt động: instruct chỉ dùng để **đúc MỘT clip giọng mẫu** (seed-locked), sau
đó mọi đoạn đều **clone** từ clip đó → chất giọng + kiểu đọc nhất quán xuyên suốt
như giọng mặc định bình thường. **Chỉ áp dụng cho giọng mặc định**; khi chọn một
giọng đã clone thì instruct **bị bỏ qua** (chất giọng lấy từ file mẫu) — UI tự làm
mờ mục này. Trên server đi qua `instruct` của `OmniVoice.generate()`; cache giọng
mặc định khoá theo `(seed, instruct)`.

## Hiệu năng

- **GPU (OmniVoice / L40S):** server bật `fp16` + `TF32` (`allow_tf32`,
  `set_float32_matmul_precision("high")`) + `cudnn.benchmark`. Mặc định chỉ xin
  **`2 CPU + 16G`** (`_VOICE_DEFAULT_CPUS`/`_MEM`) — sinh giọng/ảnh là GPU-bound
  nên ít CPU gần như không chậm, mà các node GPU ở cluster này thường **cạn CPU**
  (GPU trống nhưng cạnh chỉ 0–2 CPU rảnh), nên xin 8 CPU làm job treo `Priority`
  hàng giờ dù còn GPU. `OMP/MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK`. Cần nhiều CPU
  hơn: đặt `HOSTA100_SLURM_CPUS`/`_MEM`.
- **Batch GPU thật (ăn VRAM tối đa) — chỉnh ngay trên form, KHÔNG cần recreate:**
  ô **"Số đoạn xử lý cùng lúc (batch GPU)"** ở form Tạo giọng đọc quyết định số
  đoạn nạp vào GPU trong MỘT lần generate. Tăng (12–24) = ăn nhiều VRAM L40S +
  đọc nhanh hơn; OOM thì giảm. Mặc định lấy từ `config.OMNI_MAX_BATCH` (8). Vì
  gửi theo từng request nên đổi tức thì mỗi lần tạo, không phải tạo lại server.
  **Phòng thủ:** nếu OmniVoice không nhận list text (lỗi/sai số lượng), server
  **tự lùi về đọc từng đoạn** nên không bao giờ méo/lệch tiếng. `1` = tắt batch.
  Tắt hẳn ở server: `HOSTA100_OMNI_BATCH=0`. Kiểm: `GET /health` trả
  `batch`/`max_batch`; `server.log` in `batched N utterances in one generate()`.
- **CPU (ffmpeg / login node):** `-threads 0 -filter_complex_threads 0` =
  dùng hết nhân.
- `num_step` là cần gạt tốc/độ nét rõ nhất: 8 cho nhanh, 16 cân bằng, 24–32 nét.

> **Lưu ý so với clone-voice:** bản desktop của bạn cũng đọc **tuần tự từng đoạn**
> (narrator.py), KHÔNG batch. Batch GPU ở đây là tính năng MỚI, vượt clone-voice.

## Vận hành / lưu ý

- **Triển khai thay đổi code** = push → `git pull` trên server → **restart
  DETACHED app**. QUY TẮC quan trọng về việc có phải **tạo lại server (recreate)**
  hay không:
  - Sửa phía **login** (`app.py`, `services/*`, `templates/*`, ffmpeg) → chỉ cần
    `git pull` + **restart app**. KHÔNG cần recreate.
  - Sửa **`scripts/omnivoice_server.py`** (code chạy TRÊN node GPU), HOẶC đổi
    model / env / quant → **PHẢI recreate server**. Server là một tiến trình
    SLURM đã nạp sẵn model vào VRAM; `git pull` KHÔNG cập nhật tiến trình đang
    chạy — phải Dừng server cũ rồi tạo server mới thì nó mới đọc code mới + nạp
    lại model. (Đây là lý do bản vá engine ảnh chỉ ăn sau khi recreate.)
- **Auto-resubmit + circuit breaker**: hệt API farm — server chết liên tục
  <60s sau khi ready 3 lần → ngắt, báo `config.ALERT_WEBHOOK` (nếu đặt). Thường
  do thiếu weights trong HF cache hoặc OOM.
- **Health = `ready`**: `/health` chỉ báo `ready:true` sau khi model nạp xong
  (weights nạp SAU khi HTTP bind), nên request đầu không bị đua với lúc đang nạp.
- **Lỗi một đoạn không làm hỏng cả file**: đoạn nào server trả lỗi sẽ bị bỏ qua
  (ghi vào job.log), phần còn lại vẫn ghép thành MP3.

## Settings (biến môi trường / config)

Tất cả có default hợp lý — chỉ đặt khi cần đổi. Sửa thẳng trong `config.py` hoặc
đặt biến môi trường tương ứng.

| Biến môi trường | config | Mặc định | Ý nghĩa |
|---|---|---|---|
| `HOSTA100_OMNI_MODEL` | `OMNI_MODEL_ID` | `k2-fsa/OmniVoice` | Model HF (nạp từ HF cache chung) |
| `HOSTA100_HF_HOME` | `OMNI_HF_HOME` | `DATA_DIR/hf-cache` | HF cache chung (warmup tải vào đây) |
| `HOSTA100_OMNI_MAX_BATCH` | `OMNI_MAX_BATCH` | `8` | **Giá trị MẶC ĐỊNH** của ô batch trên form (mỗi job tự chỉnh được, không cần recreate). ↑ = ăn VRAM nhiều hơn, OOM thì ↓ |
| `HOSTA100_OMNI_BATCH` | `OMNI_BATCH_ENABLED` | `1` | `0` = tắt hẳn batch GPU ở server (cần recreate) |
| `HOSTA100_OMNI_STAGE_LOCAL` | `OMNI_STAGE_LOCAL` | `1` | `0` = tắt stage venv về ổ local (chạy thẳng NFS, chậm). Cần tarball `envs/<env>.tar.gz` |
| `HOSTA100_FFMPEG` / `HOSTA100_FFPROBE` | — | dò PATH / `ffmpeg/` | Đường dẫn ffmpeg/ffprobe trên login node |
| `HOSTA100_SLURM_CPUS` / `_MEM` | `SLURM_CPUS`/`_MEM` | `2` / `16G` (riêng voice) | CPU/RAM xin cho server giọng nói. Để thấp để chen được vào node GPU đang cạn CPU; tăng nếu cần CPU nhiều hơn (vd FLUX ~32–48G) |

Tham số **mỗi lần tạo** (trên UI, không phải env): giọng (mặc định/clone),
`num_step` 8–32, `seed`, tốc độ 0.5–2.0×, số đoạn batch GPU. (Không có nút khử
nhiễu — audio TTS vốn sạch; nhiễu mép đoạn đã tự khử. Pipeline vẫn còn tham số
`denoise=afftdn` nhưng mặc định tắt, không lộ ra UI.)

**Chỗ nào đổi thì cần gì:**
- Đổi `OMNI_MAX_BATCH` / model / env, hoặc sửa **code trong
  `scripts/omnivoice_server.py`** (engine ảnh/giọng) → **recreate server giọng nói**
  (tiến trình SLURM đang chạy không tự cập nhật khi git pull).
- Đổi pipeline/UI/ffmpeg/CPU-mem-default (login-side) → chỉ cần `git pull` +
  **restart app**.

## Troubleshooting

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| **Server nạp model mãi không xong / cứ ở loading** (nhất là A40) | `import torch/omnivoice` từ venv trên NFS quá chậm (~9 phút, "bão stat") | Tạo tarball venv (mục "Tăng tốc nạp model") để stage về ổ local → import ~4s. Kiểm `server.log`: `dùng venv local` = đã stage; `staging thất bại`/`zstd thiếu` = đang chạy NFS chậm |
| Đã tạo tarball nhưng vẫn chậm | Server cũ chạy bằng `run.sh` cũ; hoặc tarball chưa nén (5.7GB đọc 8 phút) | **Dừng + Khởi động lại** server (run.sh sinh lại lúc submit); nén tarball (`pigz`/`gzip`) |
| Server bind xong, `/health` mãi `ready:false`, log có `UnicodeEncodeError: 'latin-1'` | Console node latin-1 → một `print()` có ký tự non-ASCII làm chết luồng nạp model TRƯỚC khi nạp xong | Đã vá: run.sh đặt `PYTHONUTF8=1` + server reconfigure stdout utf-8 (commit). **Recreate server**; giữ print trong `omnivoice_server.py` ASCII |
| Treo `Priority` dù dashboard báo còn GPU trống | Node GPU trống nhưng **cạn CPU** (chỉ 0–2 CPU rảnh) → job 8-CPU không nhét vừa | Hạ CPU/RAM (`HOSTA100_SLURM_CPUS=2`/`_MEM=16G`, đã là mặc định mới); kiểm `sinfo -p main-gpu -N -o "%n %t %G %C"` |
| Job treo `PD (QOSMaxGRESPerUser)` | Quota GPU/user (vd 2) đã hết — server khác đang giữ | Dừng bớt 1 server GPU (API farm/voice). Bạn chạy tối đa N GPU job cùng lúc |
| Treo dù còn GPU trống | Ghim loại GPU (`--constraint`) mà loại đó bận; hoặc `--time` vượt giới hạn node | Chọn "Bất kỳ GPU rảnh"; hạ "Thời lượng tối đa" |
| `Cannot find … cached snapshot … offline` (giọng **mặc định**) | Weights chính chưa cache | Chạy `scripts/warmup_omnivoice.py` (Bước 2) |
| Lỗi offline chỉ khi dùng **giọng clone** | Clone bỏ trống lời thoại → cần Whisper ASR chưa cache | Chạy warmup (cache cả ASR), **hoặc** điền ô "lời thoại mẫu" |
| `OmniVoice produced empty audio` | Văn bản vô nghĩa (vd "Yyyy Yyyy…") → không có âm tiết | Dùng văn bản có nghĩa |
| Cảnh báo vàng "không tìm thấy ffmpeg" | ffmpeg chưa có trên login node | Bước 3 (tải bản tĩnh vào `ffmpeg/`) |
| `server.log`: `batch of N failed … falling back` | Bản omnivoice này không nhận list text | Vẫn chạy đúng (từng đoạn). Gửi log để chỉnh cách gọi batch, hoặc đặt `HOSTA100_OMNI_BATCH=0` |
| Server `error` sau khi chết <60s nhiều lần | OOM hoặc thiếu env/weights | Xem `server.log`; hạ `OMNI_MAX_BATCH`/đổi GPU; kiểm tra env + warmup |

## Test

Không cần GPU/HPC (chỉ stdlib + numpy + ffmpeg):

```bash
python3 tests/test_voice_pipeline.py    # chunk/SRT/atempo/ffmpeg round-trip
python3 tests/test_omnivoice_batch.py   # batch GPU + fallback (fake model)
```
