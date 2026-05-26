"""
VS Code (로컬) 버전 생성 스크립트
변경 사항:
  [환경 차이로 인한 필수 변경]
  - Cell 10 : Google Drive mount 제거 → 로컬 경로 안내
  - Cell 12 : CLIPFeatureExtractor import 제거 (May 16th 업데이트)
  - Cell 15 : images_path 로컬 경로로 변경, input() 루프 제거
  - Cell 26 : !mkdir -p → os.makedirs (Windows 호환)
  - Cell 29 : notebook_launcher → 직접 함수 호출
  - Cell 30 : .to("cuda") → device 자동 감지
  - Cell 31 : callback= → callback_on_step_end=, device 자동 감지
  [그 외 모든 셀은 _1 노트북과 동일하게 유지]
"""
import json, copy

with open('assignment3_Textual_Inversion _1 .ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

vscode_nb = copy.deepcopy(nb)
cells = vscode_nb['cells']

def set_source(cell, lines):
    cell['source'] = lines
    cell['outputs'] = []
    cell['execution_count'] = None

# ── Cell 10: Google Drive mount → 로컬 안내 ─────────────────────
set_source(cells[10], [
    "# [VS Code] Google Drive mount 불필요 - 로컬 파일 시스템 사용\n",
    "import os\n",
    "print('Working directory:', os.getcwd())\n",
    "# reference_images/ 폴더가 이 노트북 파일과 같은 위치에 있어야 합니다.\n",
])

# ── Cell 12: CLIPFeatureExtractor 라인만 제거 ───────────────────
old_src = ''.join(cells[12]['source'])
new_src = old_src.replace(
    'from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer',
    'from transformers import CLIPTextModel, CLIPTokenizer'
)
# device 자동 감지 변수 추가 (Cell 30, 31에서 사용)
new_src += "\nDEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
new_src += "print(f'Using device: {DEVICE}')\n"
cells[12]['source'] = [new_src]
cells[12]['outputs'] = []
cells[12]['execution_count'] = None

# ── Cell 15: 로컬 경로 + input() 루프 제거 ──────────────────────
set_source(cells[15], [
    "#@title Configure subject information\n",
    "#################################################\n",
    "# Prepare 4 to 6 images for fine-tuning and place them in images_path.\n",
    "# Set the placeholder_token and the initializer_token to initialize it.\n",
    "\n",
    "# Modify this section of the code as needed\n",
    "images_path = \"reference_images/cat2\"  # [VS Code] 로컬 상대 경로\n",
    "placeholder_token = \"<cat2>\"\n",
    "initializer_token = \"cat\"\n",
    "what_to_teach = \"object\"\n",
    "\n",
    "#################################################\n",
    "\n",
    "if not os.path.exists(str(images_path)):\n",
    "    raise FileNotFoundError(f'경로를 찾을 수 없습니다: {os.path.abspath(images_path)}')\n",
    "save_path = images_path\n",
])

# ── Cell 26: !mkdir -p → os.makedirs ────────────────────────────
old_src26 = ''.join(cells[26]['source'])
new_src26 = old_src26.replace(
    '!mkdir -p sd-concept-output',
    'os.makedirs("sd-concept-output", exist_ok=True)'
)
cells[26]['source'] = [new_src26]
cells[26]['outputs'] = []
cells[26]['execution_count'] = None

# ── Cell 29: notebook_launcher → 직접 호출 ──────────────────────
set_source(cells[29], [
    "#@title Start Training Loop\n",
    "# [VS Code] notebook_launcher 대신 직접 호출\n",
    "training_function(text_encoder, vae, unet)\n",
    "\n",
    "for param in itertools.chain(unet.parameters(), text_encoder.parameters()):\n",
    "  if param.grad is not None:\n",
    "    del param.grad\n",
    "if torch.cuda.is_available():\n",
    "  torch.cuda.empty_cache()\n",
])

# ── Cell 30: .to(\"cuda\") → .to(DEVICE) ──────────────────────────
old_src30 = ''.join(cells[30]['source'])
new_src30 = old_src30.replace(
    'torch_dtype=torch.float16,\n).to("cuda")',
    'torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,\n).to(DEVICE)'
)
cells[30]['source'] = [new_src30]
cells[30]['outputs'] = []
cells[30]['execution_count'] = None

# ── Cell 31: callback= → callback_on_step_end= + device 적용 ────
set_source(cells[31], [
    "#@title Observe the denoising process\n",
    "pipe = StableDiffusionPipeline.from_pretrained(\n",
    "    hyperparameters[\"output_dir\"],\n",
    "    scheduler=DPMSolverMultistepScheduler.from_pretrained(\n",
    "        hyperparameters[\"output_dir\"], subfolder=\"scheduler\"\n",
    "    ),\n",
    "    torch_dtype=torch.float16 if DEVICE == \"cuda\" else torch.float32,\n",
    ").to(DEVICE)\n",
    "\n",
    "#################################################\n",
    "# Modify this section of the code as needed.\n",
    "\n",
    "learned_embeds_step = 1000\n",
    "prompt = \"a photo of a <cat2> on a cobblestone street\"\n",
    "pipe.unload_textual_inversion(tokens=\"<cat2>\")\n",
    "\n",
    "#################################################\n",
    "\n",
    "embed_path = f\"{hyperparameters['output_dir']}/learned_embeds-step-{learned_embeds_step}.bin\"\n",
    "pipe.load_textual_inversion(embed_path)\n",
    "\n",
    "num_samples = 1\n",
    "num_inference_steps = 30\n",
    "num_intermediate = 15\n",
    "target_steps = set(np.linspace(0, num_inference_steps - 1, num_intermediate, dtype=int))\n",
    "\n",
    "intermediates = []\n",
    "\n",
    "def save_intermediate(pipe, step: int, timestep: int, callback_kwargs: dict) -> dict:\n",
    "    #################################################\n",
    "    # TODO\n",
    "    # Implement the save_intermediate function.\n",
    "    # Every step you want to save, decodes latents into an image.\n",
    "    # Add to the list of intermediates.\n",
    "    #################################################\n",
    "    # [START]\n",
    "    if step in target_steps:\n",
    "        latents = callback_kwargs[\"latents\"]\n",
    "        latents_scaled = latents / pipe.vae.config.scaling_factor\n",
    "        with torch.no_grad():\n",
    "            image_tensor = pipe.vae.decode(latents_scaled.to(pipe.vae.dtype)).sample\n",
    "        image = (image_tensor / 2 + 0.5).clamp(0, 1)\n",
    "        image_np = image.cpu().permute(0, 2, 3, 1).float().numpy()\n",
    "        pil_images = pipe.numpy_to_pil(image_np)\n",
    "        intermediates.append(pil_images[0])\n",
    "    return callback_kwargs\n",
    "    # [END]\n",
    "\n",
    "_ = pipe(\n",
    "    [prompt] * num_samples,\n",
    "    num_inference_steps=num_inference_steps,\n",
    "    guidance_scale=7.5,\n",
    "    callback_on_step_end=save_intermediate,\n",
    "    callback_on_step_end_tensor_inputs=[\"latents\"],\n",
    ")\n",
    "\n",
    "def image_grid(images: list, rows: int, cols: int):\n",
    "    w, h = images[0].size\n",
    "    grid = Image.new('RGB', (cols * w, rows * h))\n",
    "    for idx, img in enumerate(images):\n",
    "        grid.paste(img, (idx % cols * w, idx // cols * h))\n",
    "    return grid\n",
    "\n",
    "grid = image_grid(intermediates, rows=1, cols=num_intermediate)\n",
    "grid\n",
])

with open('assignment3_Textual_Inversion_vscode.ipynb', 'w', encoding='utf-8') as f:
    json.dump(vscode_nb, f, ensure_ascii=False, indent=1)
print('VS Code notebook written.')
