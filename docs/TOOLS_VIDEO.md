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

Model ảnh **mặc định = `SG161222/RealVisXL_V5.0_Lightning`** (SDXL realism, ít lỗi
nhân đôi/tay hơn nhiều SDXL-Turbo). Tải **toàn bộ layout diffusers** (không lọc
pattern — repo có subfolder unet/vae/text_encoder):

```bash
cd ~/LeeHoang_/ollama/app/host-a100
# auto-pick the env that has huggingface_hub (no placeholder to fill in)
for p in host-a100-data/envs/*/bin/python; do
  if "$p" -c "import huggingface_hub" 2>/dev/null; then PY="$p"; break; fi
done
echo "Dùng python: $PY"
# PYTHONIOENCODING=utf-8 is REQUIRED: the login node runs a C/POSIX locale, so a
# plain print() of any Vietnamese/non-latin-1 text crashes with
# 'UnicodeEncodeError: latin-1 codec can't encode…' BEFORE the download starts.
HF_HOME="$PWD/host-a100-data/hf-cache" PYTHONIOENCODING=utf-8 \
"$PY" - <<'PY'
import os
os.environ.pop("HF_HUB_OFFLINE", None)          # ensure NOT offline while fetching
from huggingface_hub import snapshot_download
print(">> Downloading SG161222/RealVisXL_V5.0_Lightning ...", flush=True)
print(">> DONE. Saved at:", snapshot_download("SG161222/RealVisXL_V5.0_Lightning"))
PY
```

> **Đừng** thêm `!` ở đầu lệnh (đó là cú pháp ô chat, không phải shell server) và
> **đừng** dán placeholder `<env-omnivoice>` — dấu `<`/`>` bị shell hiểu là chuyển
> hướng file nên lệnh không chạy gì cả.

> `ImageEngine` thử `variant='fp16'` trước, không có thì tự lùi về bản thường.
> Đổi model qua env `HOSTA100_IMAGE_MODEL`. Engine tự nhận diện loại model:
> **turbo/sdxs** (1–4 step, CFG 0, **bỏ qua negative**); **lightning/lcm/hyper**
> (4–8 step, CFG ~1.5, **DÙNG negative**, scheduler Euler-trailing); còn lại = full
> SDXL (≈24 step, CFG 7, negative). Model SDXL gen ở **bucket tỉ lệ chuẩn**
> (1344×768 cho 16:9, 768×1344 cho 9:16) thay vì 1024×576 — đây là cách hết lỗi
> **nhân đôi nhân vật** (twin) trên khung ngang.

### 3. Recreate server voice

Nhánh `--image-model` được thêm vào `_build_cmd`/batch script → **server voice cũ
phải được tạo lại** mới có khả năng gen ảnh. Vào tab **Clone giọng nói**: dừng + xoá
server cũ, tạo server mới (cùng env-omnivoice). `/health` lúc này có
`image_model` = `SG161222/RealVisXL_V5.0_Lightning` (hoặc model bạn đặt qua env).
**Đổi model cũng phải tạo lại server** (model id truyền lúc khởi động).

### 4. ffmpeg trên login node

Như tool voice: bỏ static build vào `ffmpeg/` cạnh `app.py`, hoặc PATH, hoặc
`HOSTA100_FFMPEG`/`HOSTA100_FFPROBE`.

### 5. (Tuỳ chọn) yt-dlp để lấy kịch bản từ YouTube

Cài vào **môi trường chạy `app.py` (login node, có internet)**:
`pip install -U yt-dlp`. Khi có, form Gen video hiện ô "Lấy kịch bản từ YouTube":
dán link → `services/youtube_transcript.fetch_transcript` shell ra `yt-dlp` tải
phụ đề (ưu tiên vi rồi en, kể cả phụ đề tự động), flatten + gộp các dòng caption
lặp → đổ vào ô kịch bản. Không cài thì nút bị mờ, tự nhập tay vẫn chạy.

---

## Dùng

1. Mở **Công cụ → Gen video từ kịch bản**. Banner trên cùng cho biết server GPU đã
   sẵn sàng chưa + LLM prompt bật/tắt. Chưa có server → bấm link sang tab Clone giọng
   nói để khởi động (server đó lo cả ảnh lẫn giọng).
2. Dán kịch bản, chọn giọng (mặc định/clone), phong cách ảnh, tỉ lệ/độ phân giải,
   FPS, **tốc độ đọc**, batch ảnh, số bước ảnh, bật/tắt Ken Burns & LLM-prompt →
   **Dựng video**.
   - **Tốc độ đọc (0.5–2.0×):** time-stretch giọng bằng `atempo` (giữ cao độ).
     **Thời lượng MỖI cảnh và cả video tự co/giãn theo** — pipeline đặt
     `duration = độ-dài-giọng-thô / speed`, renderer atempo audio đúng `speed` rồi
     khoá vào duration đó (`apad,atrim=0:dur`), nên hình + phụ đề luôn khớp tiếng.
     Đọc nhanh → video ngắn lại, đọc chậm → dài ra. Cảnh không có giọng giữ thời
     lượng ước lượng theo độ dài chữ.
3. Theo dõi tiến trình ở bảng bên phải (chia cảnh → prompt → ảnh GPU → giọng GPU →
   ffmpeg). Xong: xem thử trong modal, tải **MP4** + **SRT**.
4. **Ảnh các cảnh (storyboard):** nút 🖼 (`bi-images`) ở mỗi tác vụ mở lưới ảnh từng
   cảnh — kèm số cảnh, lời thoại và prompt ảnh; bấm 1 ảnh để phóng to (lightbox).
   Lưới **tự cập nhật** trong lúc đang chạy: ảnh hiện dần theo từng nhóm GPU (cảnh
   chưa tạo xong hiện ô "đang tạo…"). Dùng được cả khi job đang chạy lẫn đã xong.
   Cơ chế: pipeline ghi `scenes.json` (text+prompt+cờ ảnh) trong job dir, cập nhật
   sau bước prompt và sau mỗi nhóm ảnh; route `…/scenes.json` + `…/img/<idx>` phục vụ.

---

## Prompt ảnh & model LLM "thinking"

Tool gọi LLM của API farm để viết prompt ảnh. Lưu ý với model **Qwen3.x** (vd
`qwen3.6-35b-a3b`): chúng mặc định "thinking" (sinh `<think>…</think>`) và sẽ **đốt
hết token budget vào phần nghĩ → trả về rỗng**, nhất là khi server `n_ctx` nhỏ (mặc
định API farm = 8192). `video_prompts` xử lý: chèn `/no_think` vào prompt + **strip
`<think>` block** (kể cả khi cụt token), gọi LLM theo **lô 16 cảnh** (vừa ctx 8192),
và **cắt mỗi dòng còn ~45 từ** trước khi gửi (prompt ảnh chỉ cần ý chính). Nếu vẫn
rỗng, job log in dòng `[LLM] …` với status/finish_reason/body để chẩn đoán. Không có
LLM hợp lệ → tự lùi rule-based + dịch (vẫn ra video). **Job log in prompt thật của
TỪNG cảnh** kèm nguồn `[LLM]`/`[luật]` + anchor xuyên suốt → bấm nút Log để xem ảnh
mỗi cảnh được gen từ gì.

## Kịch bản dạng phim (tự lọc)

Nếu dán kịch bản kiểu phim/storyboard, `video_pipeline.clean_screenplay` tự **lọc
sạch trước khi chia cảnh** — chỉ giữ lời narrator/thoại để đọc + dựng cảnh. Bỏ:
tiêu đề mục `[MỞ ĐẦU – 0:00-0:10]`, dòng tiêu đề/chữ-trên-màn-hình bọc `*…*`,
đường kẻ `---`, chỉ dẫn cảnh trong `(ngoặc)`, nhãn người nói (`Narrator:`,
`Caster:`, `Khán giả hô vang:`…), dấu nhấn `*_\``, và dấu nháy bao quanh. An toàn
với văn xuôi thường (không khớp luật nào → giữ nguyên). Job log ghi "Đã lọc kịch
bản…" khi có cắt.

## Phong cách ảnh

Dropdown "Phong cách ảnh" có đủ nhóm: **Ảnh/điện ảnh** (cinematic, realistic,
documentary), **Hoạt hình/3D** (3d/Pixar, anime, comic, cartoon doodle), **Tranh vẽ
tay** (sơn dầu, màu nước, bút chì, bút chì màu, bút mực), **Tối giản** (người que /
stick figure). Style được áp cho TOÀN video để đồng nhất. Các style vẽ tay tự bỏ
"cinematic still / sharp focus" (khỏi cãi nhau với nét vẽ) và bỏ tông màu mood ở
sketch; **người que** đưa style lên đầu prompt + ép cảnh đơn giản (LLM cũng được yêu
cầu viết cảnh tối giản cho các style này). Logic ở `services/video_prompts.py`
(`_DRAWING_PRESETS`, `_NO_MOOD_PRESETS`, `_NO_ANCHOR_PRESETS`, `_STYLE_FIRST_PRESETS`).

**Khoá phong cách (vì sao "điện ảnh" có lúc ra hoạt hình).** CLIP của SDXL chỉ đọc
~77 token đầu; prompt subject 40–60 từ đã lấp gần hết → token style ở CUỐI bị cắt,
model mất hướng và tự ý vẽ hoạt hình. Cách khắc phục, áp cho MỌI style:
- **Đưa nhãn medium ngắn lên ĐẦU prompt** (`_STYLE_LEAD`), vd cinematic =
  `"cinematic film still, photorealistic, photographic, …"` → cái nhìn sống sót qua
  cắt token và lấn át toàn ảnh. Subject hạ cap còn ~320 ký tự để vừa khung.
- **Negative chống lạc medium** (`_STYLE_NEGATIVE`): nhóm ảnh thật cấm
  `cartoon, anime, illustration, drawing, 3d render, cgi, …`; nhóm vẽ/anime/3D cấm
  `photorealistic, realistic photo, …`.
- **Gợi ý medium cho LLM** (`_style_hint`): style ảnh thật → "describe REAL,
  physically-plausible scenes — never cartoon characters" (truyền vào prompt batch).

## Sửa ảnh từng cảnh + Dựng lại video (không chạy lại GPU)

Mở **Ảnh các cảnh** (icon 🖼) của một job đã xong. Mỗi thẻ cảnh có 2 nút:
- **Đổi ảnh** — tải lên ảnh thay thế. Ảnh được ffmpeg chuẩn hoá về đúng độ phân giải
  job (scale-increase + crop center → `img_XXXX.png`), nên không cần đúng tỉ lệ.
- **Tạo lại** — gen lại ảnh cảnh đó trên GPU với **seed ngẫu nhiên mới** (giữ prompt
  cũ → reroll). Chạy nền; lưới hiển thị `đang tạo lại…` → ảnh mới (tự cache-bust theo
  `ver` = mtime). Cần một server GPU đang sẵn sàng.

Sau khi đổ/tạo lại ảnh, bấm **Dựng lại video** → chỉ chạy **ffmpeg trên login node**
(không tạo lại ảnh/giọng), tái dùng `img_XXXX.png` + `voice_XXXX.wav` đã có. Tiến
trình theo dõi ngay ở bảng tác vụ như render thường.

Cơ chế: lúc render lần đầu, pipeline ghi `manifest.json` (text + basename ảnh/giọng +
duration mỗi cảnh + render cfg + tên MP4). `rerender_job` đọc manifest và dựng lại
(job cũ không có manifest → dựng lại bằng cách đo độ dài wav qua ffprobe +
scenes.json). Vì manifest lưu **basename**, ảnh tải lên (cùng tên file) được dùng tự
động. Hàm: `replace_scene_image`, `regenerate_scene_image`, `rerender_job`,
`_write_manifest`/`_read_manifest`/`_build_render_scenes` trong
`services/video_pipeline.py`; route `/tools/video/jobs/<id>/scene/<idx>/image`,
`…/regenerate`, `…/rerender`.

## Vắt GPU + dựng nhanh

- **True GPU batching cho ảnh:** `ImageEngine.generate_batch` chạy NHIỀU prompt trong
  **một lần gọi pipe** (mỗi ảnh một generator riêng để giữ seed) → lấp VRAM, nhanh hơn
  hẳn trên card 40–48GB. Defensive: gom theo cùng kích thước/steps, lỗi/**OOM** ở
  một nhóm → tự lùi về từng ảnh + `empty_cache` (không bao giờ làm hỏng cả job). Chỉnh
  bằng field **Batch ảnh** trên form (vd 6–12 trên A100/L40S).
- **Batch ảnh & batch giọng RIÊNG:** hai field độc lập trên form (ảnh dùng VRAM khác
  giọng). Mặc định = `HOSTA100_IMAGE_MAX_BATCH` / `HOSTA100_OMNI_MAX_BATCH`.
- **TF32 + cudnn.benchmark** bật cho cả image pipe lẫn OmniVoice.
- **ffmpeg nhanh nhất:** clip mỗi cảnh render **song song = số core** (cap 16), preset
  `veryfast`, nội dung still + slow-zoom vẫn nét. Đổi qua
  `render={clip_preset, clip_workers}` trong `video_render.DEFAULT_RENDER`.
- **Pass cuối KHÔNG mã hoá lại video:** phụ đề được **đốt thẳng vào từng clip** (song
  song, dùng hết core) qua `build_scene_ass` (cue 0..dur mỗi cảnh), nên "Ghép hoàn
  thiện" chỉ **nối copy video (`-c:v copy`)** + ghép audio gapless + loudnorm + mux →
  nhanh gần như tức thì thay vì giải mã + đốt sub + mã hoá lại cả phim (khâu này
  trước đây nghẽn vì libass chạy ~1 luồng). Trước kia: video dài 5–10' → pass cuối
  vài phút; giờ chỉ còn vài giây.

## Tham số / env knobs

| Env / field | Mặc định | Ý nghĩa | Đổi cần gì |
|---|---|---|---|
| `HOSTA100_IMAGE_MODEL` | `stabilityai/sdxl-turbo` | Model text-to-image GPU lazy-load | recreate server |
| `HOSTA100_IMAGE_MAX_BATCH` | `4` | Mặc định cho field "Batch ảnh" | restart app |
| (UI) Batch ảnh | = IMAGE_MAX_BATCH | Số ảnh gen **song song** trên GPU mỗi lần | không |
| (UI) Batch giọng | = OMNI_MAX_BATCH | Số đoạn giọng nạp GPU mỗi lần | không |
| (UI) Số bước ảnh | `4` | Steps SDXL (Turbo: 1–4; style vẽ tăng nhẹ) | không |
| (UI) Số bước giọng | `16` | num_step OmniVoice | không |

Login-side (pipeline/render/UI/routes/prompt) đổi gì cũng chỉ cần **git pull +
DETACHED restart app**. Phải **recreate server** khi: đổi `--image-model`/model/
quant/env GPU, **HOẶC sửa code trong `scripts/omnivoice_server.py`** (engine ảnh/
giọng chạy trên node GPU — tiến trình SLURM đang chạy đã nạp model vào VRAM, `git
pull` không cập nhật nó; phải Dừng + tạo server mới).

---

## Khắc phục sự cố

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| "Chưa có server GPU nào sẵn sàng" | Khởi động server ở tab Clone giọng nói trước. |
| Job lỗi `image engine không bật` | Server tạo trước khi có nhánh ảnh → recreate server. |
| Ảnh ra nền phẳng (màu tối) | Request ảnh lỗi (xem job.log). Thường: thiếu `diffusers` trong env, hoặc chưa tải weights về HF cache, hoặc OOM → giảm batch ảnh. |
| `Cannot find … offline` trong server.log | Chưa tải weights model ảnh về `OMNI_HF_HOME` trên login node. |
| `UnicodeEncodeError: 'latin-1' codec` khi tải model | Login node chạy locale C → `print()` chữ có dấu crash *trước khi* tải. Thêm `PYTHONIOENCODING=utf-8` và dùng thông báo không dấu (xem mục Cài đặt 2). |
| Lệnh tải "không hiện gì" | Đã dán nguyên placeholder `<env-omnivoice>` (dấu `<`/`>` bị shell hiểu là redirect) hoặc thêm `!` ở đầu. Dùng khối tự-dò-env ở mục Cài đặt 2. |
| Nhân vật bị nhân đôi (twin) / lỗi tay / nhoè | Triệu chứng của SDXL-Turbo (bỏ qua negative + gen 1024×576). Đảm bảo `image_model` là **RealVisXL Lightning** (xem `/health`), gen ở 1344×768. Cảnh lỗi lẻ → bấm **Tạo lại** (seed mới). |
| `Input type (c10::Half) and bias type (float) should be the same` → mọi ảnh ra nền phẳng | Pipe lẫn dtype: `variant="fp16"` để lại một thành phần (hay VAE) ở fp32. `ImageEngine.load()` đã **ép cả pipe về fp16** sau khi load (`pipe.to(dtype)`). Cần `git pull` + **tạo lại server**. |
| Cảnh đầu rất chậm | Bình thường — GPU đang lazy-load SDXL vào VRAM. Các cảnh sau nhanh. |
| Prompt ra tiếng Anh chung chung | Không có LLM API farm → rule-based. Bật một API-farm server để có prompt cụ thể hơn. |
| OOM khi vừa chạy ảnh vừa chạy giọng | SDXL + OmniVoice cùng card. Giảm batch ảnh/giọng, hoặc dùng card ≥40GB. |
| "ffmpeg dựng video thất bại" | Thiếu ffmpeg trên login node (xem banner), hoặc audio toàn im lặng (mọi cảnh đều fail TTS → loudnorm NaN). |
| Nút "Lấy phụ đề" mờ / "Không tìm thấy yt-dlp" | Chưa cài yt-dlp ở login node: `pip install -U yt-dlp`. |
| "Video này không có phụ đề" | Video không có phụ đề vi/en (kể cả tự động) — thử video khác hoặc nhập tay. |

---

## Test & Deploy

- Test (local, không cần GPU/HPC — chỉ cần ffmpeg): `python3 tests/test_video_pipeline.py`
  (chia/prompt rule-based, filter graph Ken Burns + audio, SRT/ASS, render MP4 thật,
  fallback ảnh thiếu).
- Deploy: sửa trên Mac → commit `main` → push → `git pull` trên server → **DETACHED
  restart** app (xem quy trình ở README / memory). **Nếu lần sửa đụng tới
  `scripts/omnivoice_server.py`** (engine ảnh/giọng) thì sau git pull phải **recreate
  server voice** — tiến trình GPU đang chạy không tự đọc code mới (đã nạp model vào
  VRAM). Sửa thuần login-side thì chỉ restart app là đủ.
