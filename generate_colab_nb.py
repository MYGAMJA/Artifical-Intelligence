"""
Colab 버전 생성 스크립트
변경 사항 (최소한):
1. Cell 12 : CLIPFeatureExtractor import 제거 (May 16th 업데이트)
2. Cell 31 : callback= → callback_on_step_end= (diffusers 0.27+ 필수)
그 외 모든 셀은 _1 노트북과 동일하게 유지.
"""
import json, copy

with open('assignment3_Textual_Inversion _1 .ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

colab_nb = copy.deepcopy(nb)
cells = colab_nb['cells']

def set_source(cell, lines):
    cell['source'] = lines
    cell['outputs'] = []
    cell['execution_count'] = None

# ── Cell 12: CLIPFeatureExtractor 라인만 제거 ───────────────────
# 원본: from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
# 수정: CLIPFeatureExtractor 제거, CLIPTextModel·CLIPTokenizer 유지
old_src = ''.join(cells[12]['source'])
new_src = old_src.replace(
    'from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer',
    'from transformers import CLIPTextModel, CLIPTokenizer'
)
cells[12]['source'] = [new_src]
cells[12]['outputs'] = []
cells[12]['execution_count'] = None

# ── Cell 31: callback= → callback_on_step_end= ──────────────────
# diffusers 0.27 이후 callback/callback_steps 파라미터가 제거됨
set_source(cells[31], [
    "#@title Observe the denoising process\n",
    "pipe = StableDiffusionPipeline.from_pretrained(\n",
    "    hyperparameters[\"output_dir\"],\n",
    "    scheduler=DPMSolverMultistepScheduler.from_pretrained(\n",
    "        hyperparameters[\"output_dir\"], subfolder=\"scheduler\"\n",
    "    ),\n",
    "    torch_dtype=torch.float16,\n",
    ").to(\"cuda\")\n",
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

with open('assignment3_Textual_Inversion_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(colab_nb, f, ensure_ascii=False, indent=1)
print('Colab notebook written.')
