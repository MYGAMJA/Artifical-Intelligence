# Assignment 3 — Textual Inversion 과제 가이드

## 0. 한눈에 보기

- **주제**: Diffusion 모델 개인화(Personalization) 기법인 **Textual Inversion** 직접 구현 및 분석
- **기반 논문**: *An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion* (ICLR 2023, Gal et al.)
- **제출 마감**: 2026-05-29 (금) 23:59
- **제출물**:
  1. Colab 링크 (full-access 권한 부여)
  2. `.py` 파일
  3. 결과/답변이 담긴 **리포트**
  4. 위 셋을 모두 묶은 `.zip` 아카이브
- **주의**: ChatGPT 등 외부 도구 무단 사용·표절은 즉시 0점 처리
- **러닝 환경**: Colab 무료 T4 GPU 기준 학습 30~40분 / 추론 수초

---

## 1. Textual Inversion 핵심 개념 (먼저 이해할 것)

Diffusion personalization은 **사전학습된 text-to-image diffusion 모델**에게 사용자가 정의한 *새로운 시각 개념(예: 내 강아지, 내 머그컵)*을 단 3~5장의 참조 이미지만으로 학습시키는 기법입니다.

Textual Inversion의 핵심 아이디어:

- **모델 전체를 fine-tune하지 않는다.** Text Encoder · U-Net · VAE 는 **모두 frozen**.
- 대신 **placeholder token**(예: `<cat2>`)을 토크나이저 vocab에 새로 추가하고, 그 토큰의 **embedding 벡터 하나만** 학습한다.
- `initializer_token`(예: `"cat"`)의 임베딩을 시작점으로 사용해 학습을 안정화.
- 학습 후에는 자연어 프롬프트 안에 placeholder token을 자유롭게 끼워 넣어(`"a photo of <cat2> on the beach"`) 새로운 장면을 생성할 수 있다.

### 학습 루프의 흐름
1. 참조 이미지 → VAE encoder → latent $z$
2. latent에 노이즈 추가 → $z_t$
3. 텍스트 프롬프트("a photo of `<placeholder>`") → Tokenizer → Embedding Lookup → Text Encoder → conditioning $c_\theta(y)$
4. U-Net이 $z_t$, $t$, $c_\theta(y)$ 받아 노이즈 예측
5. **LDM loss** = $\| \epsilon - \epsilon_\theta(z_t, t, c_\theta(y)) \|_2^2$
6. 역전파 → **`<placeholder>` 임베딩 벡터만** 업데이트

즉, "학습"이라기보다 **하나의 임베딩 벡터를 최적화(inversion)** 하는 작업.

---

## 2. 폴더 구성

```
Assignment 3/
├── CSE4007_Artificial_Intelligence_2026___Assignment_3.pdf   # 과제 명세
├── an_image_is_worth_one_word_per.pdf                        # 원논문 (참고)
├── assignment3_Textual_Inversion.ipynb                       # 구현해야 할 노트북
└── reference_images/                                         # 학습용 참조 이미지 (10개 컨셉)
    ├── backpack_dog/
    ├── berry_bowl/
    ├── cat2/
    ├── clock/
    ├── grey_sloth_plushie/
    ├── monster_toy/
    ├── poop_emoji/
    ├── rc_car/
    ├── robot_toy/
    └── teapot/
```

각 `reference_images/<concept>/` 폴더에는 한 가지 개념에 대한 3~5장의 학습용 이미지가 들어 있습니다. 이 중 원하는 컨셉을 선택해 placeholder token으로 학습시킵니다.

---

## 3. 문제별 상세 분해

### Problem 1 — Learning the New Token (구현)

노트북의 TODO 섹션을 채우는 단계입니다. 총 **3개의 핵심 TODO**가 있습니다.

#### (a) Tokenizer 셋업 & initializer 토큰 처리 (노트북 Cell 19 근처)
- `tokenizer.add_tokens(placeholder_token)` 으로 placeholder token을 vocab에 추가
- `num_added_tokens` 가 0이면 이미 존재한다는 뜻 → 에러
- `tokenizer.encode(initializer_token, add_special_tokens=False)` 로 token id 변환
- initializer token은 **반드시 단일 토큰**이어야 함 (sub-word로 쪼개지면 에러)
- `text_encoder.resize_token_embeddings(len(tokenizer))` 로 임베딩 레이어 확장
- `token_embeds[placeholder_token_id] = token_embeds[initializer_token_id].clone()` 로 초기값 복사

#### (b) 모델 freeze (Cell 23)
- `vae`, `unet` → 모든 파라미터 `requires_grad=False`
- `text_encoder` → **token embedding layer만 학습 가능**, 나머지(transformer 블록, position embedding 등)는 freeze
- 구현 힌트: `text_encoder.text_model.encoder`, `text_encoder.text_model.final_layer_norm`, `text_encoder.text_model.embeddings.position_embedding` 의 파라미터를 freeze 처리

#### (c) `training_function` 내부 완성 (Cell 27)
구현해야 할 핵심 부분:
1. **Optimizer에 어떤 파라미터를 줄지 결정**
   - 힌트: `text_encoder.get_input_embeddings().parameters()` 만 optimizer에 넘긴다 (다른 파라미터는 freeze되어 있으므로)
2. **학습 스텝**
   - 이미지 → VAE encoder → latents (× `vae.config.scaling_factor`)
   - 노이즈 샘플 + timestep 샘플 → `scheduler.add_noise`
   - 텍스트 → text_encoder → encoder_hidden_states
   - `unet(noisy_latents, timesteps, encoder_hidden_states).sample` → 예측 노이즈
   - MSE loss = `F.mse_loss(model_pred, noise)`
   - `accelerator.backward(loss)`
3. **학습 대상 외 임베딩이 업데이트되지 않도록 grad mask 처리**
   - placeholder가 아닌 모든 토큰의 embedding gradient를 0으로 만들어 줘야 함
   - 일반적인 패턴:
     ```
     grads = text_encoder.get_input_embeddings().weight.grad
     index_no_updates = torch.arange(len(tokenizer)) != placeholder_token_id
     grads.data[index_no_updates, :] = 0
     ```
4. 일정 step마다 `save_progress(...)` 호출해 학습된 임베딩 저장

---

### Problem 2 — Use & Evaluate (실험·관찰·분석)

학습이 끝난 임베딩을 로드해 다양한 조건에서 생성해 보며 Textual Inversion의 동작을 분석합니다.

#### (a) Denoising 과정 시각화 (Cell 31)
- `pipe(...)` 호출 시 `callback=save_intermediate` 옵션으로 중간 step latents를 받아서 이미지로 디코딩
- 핵심 구현:
  ```
  latents_scaled = latents / vae.config.scaling_factor
  image_tensor = vae.decode(latents_scaled).sample
  image = (image_tensor / 2 + 0.5).clamp(0, 1)
  pil_images = [...]  # tensor → numpy → PIL 변환
  intermediates.append(pil_images[0])
  ```
- `target_steps` 에 포함된 step에서만 저장 (예: 30 step 중 균등 간격 15장)

#### (b) Timestep별 관찰 (정성 분석)
리포트에 다음 관점으로 서술:
- **초기 step (높은 timestep, 노이즈 많음)**: 색·구도 같은 **저주파 정보**가 먼저 잡힌다
- **후기 step (낮은 timestep)**: 텍스처·디테일 같은 **고주파 정보**가 나타난다
- 전체 timestep에 걸친 패턴 / 어떤 step부터 컨셉의 identity가 보이기 시작하는지

#### (c) Failure case 탐색 실험 설계
다음 변수를 바꿔가며 비교 실험을 설계:
- **다른 참조 이미지 컨셉** (cat2 vs teapot vs poop_emoji 등)
- **다양한 프롬프트 변형**: 스타일 변환 ("oil painting of `<X>`"), 장면 변경 ("`<X>` on the beach"), 합성 ("`<X>` and a dog")
- **learning rate** 변경 (너무 크면 발산/overfitting, 너무 작으면 미학습)
- **max_train_steps** 변경 (under-trained vs over-trained)
- **before vs after**: 학습 *전*에 initializer token으로 생성한 결과 vs 학습 *후*에 placeholder로 생성한 결과 비교

#### (d) 종합 토론 — token embedding만 최적화하는 방식의 장단점
- **장점**: 매우 적은 파라미터(단일 벡터 ~768d), 모델 본체 손상 없음, 다른 토큰과 자연스럽게 합성 가능, 저장 용량 작음
- **한계**: 복잡한 형태/디테일은 못 잡음, 프롬프트 의존도 높음, 학습 step에 민감, overfitting 시 다양한 장면 생성 능력 저하

#### (e) 근거 제시
- 정성적: 생성 이미지 격자 비교 (before/after, 조건별)
- 정량적: **CLIP-T** (이미지-프롬프트 의미 일치도), **CLIP-I** (이미지-참조 이미지 유사도) 점수 표/그래프

---

### Problem 3 — Additional Experiments (자유 실험·개선 제안)

가장 자유도가 높은 문제. 정성+정량 분석 모두 요구됨.

#### (a) Success/Failure 심층 분석
Problem 2에서 발견한 케이스를 더 깊이 파고든다:
- **언제 잘 되나?** 단순하고 강한 색·형태 특징(예: monster_toy), 자연어로 묘사 가능한 컨셉
- **언제 실패하나?** 글자/로고/세밀한 텍스처, 사람 얼굴, 특정 포즈, 여러 개체가 섞인 참조 이미지
- 옵션: U-Net의 **cross-attention map** 추출해서 placeholder token이 이미지의 어디에 주목하는지 시각화

#### (b) 개선 전략 **구체적으로** 제안
예시 아이디어:
- **데이터 증강**: random crop, color jitter, horizontal flip 등을 학습 이미지에 적용
- **다중 토큰 임베딩**: 단일 vector가 아닌 N개 vector로 컨셉 표현
- **Loss 수정**: 임베딩이 initializer token에서 너무 멀어지지 않도록 regularization 항 추가, 또는 prior preservation loss
- **학습 스케줄**: learning rate scheduling, EMA 사용, curriculum learning
- **프롬프트 템플릿 다양화**: CLIP ImageNet 템플릿 외 도메인 특화 템플릿 추가

#### (c) 실험 코드 구현 + 결과 보고
- 제안 중 1~2개를 골라 실제로 구현
- baseline vs improved 정량 비교 (CLIP-T, CLIP-I)
- 정성 비교 (이미지 격자)

#### (d) 본인 관찰·결론
- 실험에서 본인이 깨달은 점
- 어떤 개선이 효과적이었는지, 왜 그랬다고 생각하는지
- Textual Inversion이라는 패러다임의 한계와 후속 연구 방향(예: DreamBooth, LoRA)

---

## 4. 진행 순서 추천

```
[1단계] 환경 설정
  └─ Colab 노트북 열기 → T4 GPU 런타임 → Cell 0~12 (Prepare) 실행
       · diffusers / transformers / accelerate 설치
       · reference_images 업로드 또는 마운트

[2단계] Problem 1 구현 (Cell 13~)
  ├─ Cell 19: Tokenizer + initializer 처리 TODO
  ├─ Cell 23: Freeze TODO
  └─ Cell 27: training_function TODO
       · 작은 max_train_steps (예: 200)로 먼저 동작 확인
       · 정상 동작하면 1000~2000 step으로 본 학습 (30~40분)

[3단계] Problem 2 실험
  ├─ Cell 31: 중간 step 시각화 TODO 구현
  ├─ 다양한 프롬프트로 생성 → 결과 모으기
  └─ CLIP-T / CLIP-I 점수 계산 스크립트 작성

[4단계] Problem 3 추가 실험
  ├─ 개선 아이디어 1~2개 선정
  ├─ 코드 작성 → 재학습 → 비교 평가
  └─ 결과 정리

[5단계] 산출물 정리
  ├─ Colab 공유 링크 권한 확인 (anyone with link → editor 또는 view)
  ├─ .py로 export (File → Download → .py)
  ├─ 리포트 작성 (PDF 추천): 각 문제별 답변·이미지·표
  └─ 전체 .zip 압축 후 제출
```

---

## 5. 리포트 작성 체크리스트

- [ ] Problem 1: 세 가지 TODO 각각에 대해 **왜 그렇게 구현했는지** 짧게 설명
- [ ] Problem 2-a/b: 중간 step 이미지 격자 + timestep별 관찰 서술
- [ ] Problem 2-c: 실험 설계 표(조건 / 변수 / 기대 결과)
- [ ] Problem 2-d/e: 장단점 토론 + 정량/정성 근거
- [ ] Problem 3-a: 성공/실패 케이스 예시 이미지
- [ ] Problem 3-b: 개선 전략 제안 (불릿 + 짧은 근거)
- [ ] Problem 3-c: 구현 코드 설명 + 베이스라인 대비 결과 비교
- [ ] Problem 3-d: 본인의 결론과 향후 방향

---

## 6. 자주 빠지는 함정

1. **placeholder token이 sub-word로 쪼개지는 경우** — `<token>` 처럼 angle bracket으로 감싸 단일 토큰화되도록 한다
2. **embedding gradient mask를 빼먹어 다른 토큰까지 학습되는 문제** — Problem 1-(c)에서 grad zero 처리 필수
3. **VAE scaling factor 누락** — encode 후 곱하기, decode 전 나누기 (`0.18215` 또는 `vae.config.scaling_factor`)
4. **mixed precision + accelerator** — `accelerator.prepare(...)`로 모델·옵티마이저·데이터로더 감싸기
5. **저장된 임베딩 로드** — `pipe.load_textual_inversion(embed_path)` 호출 전에 `pipe.unload_textual_inversion(...)`로 이전 토큰 정리 필요(같은 토큰명 재사용 시)
6. **CLIP-T/I 평가 시 동일한 CLIP 모델 사용** — `openai/clip-vit-base-patch32` 또는 `large-patch14` 등 통일

---

## 7. 핵심 참고 자료

- 원논문: `an_image_is_worth_one_word_per.pdf` (Section 3 Method, Section 4 Qualitative Comparisons)
- HuggingFace diffusers Textual Inversion 공식 예시 (구조 참고용, **복붙 금지**)
- 노트북 markdown 셀에 이미 핵심 개념 설명이 잘 정리되어 있음 → 먼저 정독
