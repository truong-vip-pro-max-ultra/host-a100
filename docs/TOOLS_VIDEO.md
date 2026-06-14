# Tools → Gen video từ kịch bản

Biến một **kịch bản text** thành **video kể chuyện** (ảnh minh hoạ AI + giọng đọc
OmniVoice + phụ đề + slow-zoom Ken Burns) ngay trong host-a100, tab **Công cụ →
Gen video từ kịch bản** (`/tools/gen-video`).

Đây là tool thứ hai của tab Công cụ, port từ desktop *AI Story Video Studio*. Nó
**dùng CHUNG đúng một GPU server với Clone giọng nói** — không tốn thêm slot GPU.

---

## Kiến trúc (vì sao như vậy)

```
  Login node (Flask, có internet, nhiều CPU)        GPU compute node (sbatch)
  ┌─────────────────────────────────────────┐      ┌──────────────────────────┐
  │ video_pipeline._run_job                  │      │ scripts/omnivoice_server │
  │  1. chia cảnh (voice_pipeline.parse…)    │      │  • OmniEngine  (TTS)     │
  │  2. video_prompts.build_prompts          │ HTTP │  • ImageEngine (SDXL)    │
  │       └ LLM API farm  hoặc  rule-based   │─────▶│    lazy-load, fp16,      │
  │  3. POST /generate_image_batch  ────────────────▶    resident pipe        │
  │  4. POST /synthesize_batch     ─────────────────▶    (vắt kiệt VRAM)       │
  │  5. video_render.render (ffmpeg) → MP4   │      └──────────────────────────┘
  └─────────────────────────────────────────┘
```

- **GPU server gộp chung:** `scripts/omnivoice_server.py` giờ chạy CẢ HAI engine
  trong một tiến trình. `ImageEngine` (diffusers SDXL) **lazy-load** — chỉ nạp model
  vào VRAM khi có request ảnh đầu tiên, nên server dùng-chỉ-giọng không tốn gì thêm.
  Endpoints mới: `POST /generate_image`, `POST /generate_image_batch`; `/health`
  báo thêm `image_model` / `image_ready` / `image_error`.
- **Tối ưu GPU (port từ bản chỉnh cho RTX 4090):** card ≥16GB VRAM → giữ pipe
  **thường trú** (KHÔNG `model_cpu_offload`, KHÔNG attention-slicing) → ~6× nhanh hơn
  so với đường tiết kiệm RAM. fp16, native 1024px cho SDXL.
- **Login node lo phần còn lại:** chia cảnh, viết prompt, ghép video bằng ffmpeg
  (Ken Burns zoompan + fade, ghép audio gapless theo đúng thời lượng cảnh, phụ đề
  `.ass` burn-in + `.srt`, nhạc nền + loudnorm). Gọi GPU server qua HTTP NỘI BỘ
  (login→compute, KHÔNG qua Cloudflare tunnel → không dính Error-524).
- **Prompt ảnh:** ưu tiên **LLM của API farm** (`serve_service.resolve_endpoint()` →
  `/v1/chat/completions`, Qwen3-Coder) để viết mỗi câu tiếng Việt thành 1 cảnh hình
  cụ thể bằng tiếng Anh. Không có LLM nào ready → **fallback rule-based** (detect
  mood + anchor + style) + dịch VN→EN bằng `deep-translator` (chạy ở login node).

**File mới:** `services/video_pipeline.py` (orchestrator + jobs),
`services/video_prompts.py` (prompt), `services/video_render.py` (ffmpeg + phụ đề),
routes `/tools/video/...` trong `app.py`, `templates/tool_gen_video.html`. DB:
bảng `video_jobs`. config: `VIDEO_OUTPUTS_DIR`, `IMAGE_MODEL_ID`, `IMAGE_MAX_BATCH`.

---

## Cài đặt (một lần)

### 1. Thêm `diffusers` vào MÔI TRƯỜNG của server (env-omnivoice)

Tab **Môi trường → "…hoặc cài từ file requirements.txt"** → upload
`scripts/requirements-video-image.txt` (KHÔNG dùng ô "cài gói" — nó từ chối flag
`--index-url`). File này:

- cài `diffusers` (gói thuần Python, dùng torch sẵn có),
- **GHIM lại `transformers==5.3.0`** để pip không tự nâng (bản mới ref
  `torch.float8_e8m0fnu`, cần torch≥2.7 → crash import omnivoice trên torch 2.6),
- **KHÔNG** đụng tới torch/torchaudio (đã có 2.6.0+cu124).

`deep-translator` trong file đó là để **fallback dịch khi không có LLM** — gói này
cần nằm trong **môi trường chạy `app.py` (login node)**, không phải env-omnivoice
(việc dịch chạy trong tiến trình Flask, không trên GPU). Nếu luôn bật prompt-LLM thì
không bắt buộc.

### 2. Tải weights model ảnh về HF cache CHUNG (login node có internet)

Compute node không có internet, nên phải tải sẵn về `OMNI_HF_HOME`
(`host-a100-data/hf-cache`) trên login node:

```bash
cd <app>            # ~/LeeHoang_/ollama/app/host-a100
HF_HOME="$PWD/host-a100-data/hf-cache" \
  host-a100-data/envs/<env-omnivoice>/bin/python -c \
  "from huggingface_hub import snapshot_download; \
   snapshot_download('stabilityai/sdxl-turbo', \
   allow_patterns=['*.json','*.txt','*fp16*','*.safetensors'])"
```

> Nếu model bạn chọn không có biến thể `fp16` riêng, `ImageEngine` tự thử lại không
> `variant="fp16"`. Đổi model qua env `HOSTA100_IMAGE_MODEL` (vd
> `stabilityai/stable-diffusion-xl-base-1.0`).

### 3. Recreate server voice

Nhánh `--image-model` được thêm vào `_build_cmd`/batch script → **server voice cũ
phải được tạo lại** mới có khả năng gen ảnh. Vào tab **Clone giọng nói**: dừng + xoá
server cũ, tạo server mới (cùng env-omnivoice). `/health` lúc này có
`image_model` = `stabilityai/sdxl-turbo`.

### 4. ffmpeg trên login node

Như tool voice: bỏ static build vào `ffmpeg/` cạnh `app.py`, hoặc PATH, hoặc
`HOSTA100_FFMPEG`/`HOSTA100_FFPROBE`.

---

## Dùng

1. Mở **Công cụ → Gen video từ kịch bản**. Banner trên cùng cho biết server GPU đã
   sẵn sàng chưa + LLM prompt bật/tắt. Chưa có server → bấm link sang tab Clone giọng
   nói để khởi động (server đó lo cả ảnh lẫn giọng).
2. Dán kịch bản, chọn giọng (mặc định/clone), phong cách ảnh, tỉ lệ/độ phân giải,
   FPS, batch ảnh, số bước ảnh, bật/tắt Ken Burns & LLM-prompt → **Dựng video**.
3. Theo dõi tiến trình ở bảng bên phải (chia cảnh → prompt → ảnh GPU → giọng GPU →
   ffmpeg). Xong: xem thử trong modal, tải **MP4** + **SRT**.

---

## Tham số / env knobs

| Env | Mặc định | Ý nghĩa | Đổi cần gì |
|---|---|---|---|
| `HOSTA100_IMAGE_MODEL` | `stabilityai/sdxl-turbo` | Model text-to-image GPU lazy-load | recreate server |
| `HOSTA100_IMAGE_MAX_BATCH` | `4` | Số ảnh / request `/generate_image_batch` | restart app |
| (UI) Batch ảnh | = IMAGE_MAX_BATCH | Số ảnh gửi GPU mỗi lần (mỗi job) | không |
| (UI) Số bước ảnh | `4` | Steps SDXL (Turbo: 1–4) | không |

Login-side (pipeline/render/UI/routes/prompt) đổi gì cũng chỉ cần **git pull +
DETACHED restart app**. Chỉ khi đổi `--image-model`/model/quant/env của GPU mới cần
**recreate server**.

---

## Khắc phục sự cố

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| "Chưa có server GPU nào sẵn sàng" | Khởi động server ở tab Clone giọng nói trước. |
| Job lỗi `image engine không bật` | Server tạo trước khi có nhánh ảnh → recreate server. |
| Ảnh ra nền phẳng (màu tối) | Request ảnh lỗi (xem job.log). Thường: thiếu `diffusers` trong env, hoặc chưa tải weights về HF cache, hoặc OOM → giảm batch ảnh. |
| `Cannot find … offline` trong server.log | Chưa tải weights model ảnh về `OMNI_HF_HOME` trên login node. |
| Cảnh đầu rất chậm | Bình thường — GPU đang lazy-load SDXL vào VRAM. Các cảnh sau nhanh. |
| Prompt ra tiếng Anh chung chung | Không có LLM API farm → rule-based. Bật một API-farm server để có prompt cụ thể hơn. |
| OOM khi vừa chạy ảnh vừa chạy giọng | SDXL + OmniVoice cùng card. Giảm batch ảnh/giọng, hoặc dùng card ≥40GB. |
| "ffmpeg dựng video thất bại" | Thiếu ffmpeg trên login node (xem banner), hoặc audio toàn im lặng (mọi cảnh đều fail TTS → loudnorm NaN). |

---

## Test & Deploy

- Test (local, không cần GPU/HPC — chỉ cần ffmpeg): `python3 tests/test_video_pipeline.py`
  (chia/prompt rule-based, filter graph Ken Burns + audio, SRT/ASS, render MP4 thật,
  fallback ảnh thiếu).
- Deploy: sửa trên Mac → commit `main` → push → `git pull` trên server → **DETACHED
  restart** app (xem quy trình ở README / memory). Nhánh ảnh trong
  `omnivoice_server.py` là file repo → cần git pull RỒI recreate server voice.
