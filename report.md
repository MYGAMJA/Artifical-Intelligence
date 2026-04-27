# CSE4007 Artificial Intelligence — Assignment #2 Report

---

## Problem 1: Building DEtection TRansformer (DETR)

### a) Positional Encoding (Sinusoidal)

`DetrSinePositionEmbedding`은 이미지의 2D 공간 정보를 Transformer가 이해할 수 있는 형태로 인코딩합니다.

**구현 핵심:**

```python
dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=feature_map.device)
dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.embedding_dim)

pos_x = x_embed[:, :, :, None] / dim_t   # (B, H, W, embedding_dim)
pos_y = y_embed[:, :, :, None] / dim_t

pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (B, d_model, H, W)
```

**원리:**
- 각 픽셀 위치의 x, y 누적합(`cumsum`)을 기반으로 연속 위치값을 계산합니다.
- `normalize=True`일 때 전체 이미지 크기로 정규화해 `[0, 2π]` 범위로 스케일합니다.
- `embedding_dim`개의 주파수 성분 `dim_t`를 생성하고, 짝수 인덱스에 sin, 홀수 인덱스에 cos을 적용합니다.
- x 방향과 y 방향 인코딩을 concat해 `d_model` 차원의 위치 벡터를 만듭니다.

**검증 근거:**
- 출력 shape이 `(B, d_model, H, W)`임을 확인할 수 있습니다. `d_model=256`이면 x방향 128차원 + y방향 128차원 = 256으로 정확히 일치합니다.
- 동일한 위치는 항상 동일한 인코딩을 가지므로 결정론적(deterministic)입니다.
- 인접한 위치일수록 인코딩 벡터 간의 거리가 작고, 멀수록 커지는 연속성(continuity)이 성립합니다.
- 이 구현은 DETR 논문 원저자의 공식 PyTorch 구현체(`models/position_encoding.py`)와 동일한 방식입니다.

---

### b) Attention Module

`DetrAttention`은 Multi-Head Self-Attention 및 Cross-Attention을 모두 처리합니다.

**구현 핵심:**

```python
# __init__
self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

# forward - position embedding 더하기
hidden_states_original = hidden_states
hidden_states = self.with_pos_embed(hidden_states, object_queries)        # Q, K에만 더함

key_value_states_original = key_value_states
key_value_states = self.with_pos_embed(key_value_states, spatial_position_embeddings)

# attention 계산
attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))       # QK^T
attn_weights = nn.functional.softmax(attn_weights, dim=-1)                # softmax
attn_output  = torch.bmm(attn_weights, value_states)                      # weighted sum V
attn_output  = self.out_proj(attn_output)
```

**원리:**
- `with_pos_embed`는 position embedding을 Query와 Key에만 더합니다. Value는 원본(`_original`)을 사용합니다. 이는 DETR 논문의 핵심 설계로, 위치 정보는 어디를 볼지(Q/K)에만 영향을 주고, 무엇을 가져올지(V)는 순수 feature를 사용합니다.
- `scaling = head_dim ** -0.5`를 Q에 미리 곱해 gradient vanishing을 방지합니다.
- `bmm`으로 batch matrix multiplication을 수행해 `(B*num_heads, target_len, source_len)` attention weight를 계산합니다.
- Self-attention이면 K/V도 `hidden_states`에서, Cross-attention이면 `encoder_hidden_states`에서 가져옵니다.

**검증 근거:**
- attention weight의 shape이 `(batch*num_heads, target_len, source_len)`임을 코드 내 assert로 확인합니다.
- softmax를 `dim=-1`에 적용해 각 query에 대한 모든 key의 weight 합이 1이 됩니다.
- output shape `(batch*num_heads, target_len, head_dim)` 역시 assert로 검증됩니다.
- `embed_dim = num_heads * head_dim` 조건을 `__init__`에서 강제합니다.

---

### c) Encoder Layers

`DetrEncoderLayer`는 Self-Attention + Feed-Forward Network(FFN) 구조입니다.

**구현 핵심:**

```python
# __init__
self.self_attn          = DetrAttention(embed_dim=d_model, num_heads=encoder_attention_heads)
self.self_attn_layer_norm = nn.LayerNorm(d_model)
self.fc1                = nn.Linear(d_model, encoder_ffn_dim)
self.fc2                = nn.Linear(encoder_ffn_dim, d_model)
self.final_layer_norm   = nn.LayerNorm(d_model)

# forward
residual     = hidden_states
hidden_states = self.self_attn(hidden_states, attention_mask, object_queries)
hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
hidden_states = residual + hidden_states            # Residual connection
hidden_states = self.self_attn_layer_norm(hidden_states)

residual      = hidden_states
hidden_states = self.activation_fn(self.fc1(hidden_states))
hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
hidden_states = self.fc2(hidden_states)
hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
hidden_states = residual + hidden_states
hidden_states = self.final_layer_norm(hidden_states)
```

`DetrEncoder.forward`:
```python
hidden_states = inputs_embeds
hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
for encoder_layer in self.layers:
    hidden_states = encoder_layer(hidden_states, attention_mask, object_queries)
```

**원리:**
- Pre-LN 구조가 아닌 Post-LN 구조(Attention → Residual → LN → FFN → Residual → LN)를 사용합니다.
- FFN은 `d_model → encoder_ffn_dim(2048) → d_model` 로 차원을 키웠다가 줄이는 병목 구조입니다.
- `object_queries`(positional encoding)을 매 레이어마다 Q/K에 더해 위치 정보를 지속적으로 주입합니다.
- Dropout은 훈련 중에만 적용됩니다.

**검증 근거:**
- Encoder 입력/출력 shape이 동일(`B, H*W, d_model`)하므로, 이후 Decoder와의 연결이 dimension 오류 없이 작동합니다.
- 논문 Fig.1의 인코더 구조(self-attention + FFN, position embedding 추가)와 정확히 일치합니다.
- `torch.isinf`/`torch.isnan` 체크가 포함되어 있어 학습 중 수치 불안정을 감지합니다.

---

### d) Decoder Layers

`DetrDecoderLayer`는 Self-Attention + Cross-Attention + FFN 구조입니다.

**구현 핵심:**

```python
# Self Attention (object queries끼리)
residual      = hidden_states
hidden_states = self.self_attn(hidden_states,
                               object_queries=query_position_embeddings,  # Q, K에 더함
                               attention_mask=attention_mask)
hidden_states = dropout + residual + self.self_attn_layer_norm(...)

# Cross Attention (queries → encoder features)
residual      = hidden_states
hidden_states = self.encoder_attn(hidden_states,
                                  object_queries=query_position_embeddings,     # Q에 더함
                                  key_value_states=encoder_hidden_states,       # K, V는 encoder 출력
                                  spatial_position_embeddings=object_queries)   # K에 더함
hidden_states = dropout + residual + self.encoder_attn_layer_norm(...)

# FFN
residual      = hidden_states
hidden_states = ReLU(fc1) → dropout → fc2 → dropout
hidden_states = residual + final_layer_norm(...)
```

`DetrDecoder.forward`:
```python
self.layernorm = nn.LayerNorm(d_model)   # 마지막에만 LayerNorm 적용

for decoder_layer in self.layers:
    hidden_states = decoder_layer(hidden_states, ..., encoder_hidden_states, ...)
hidden_states = self.layernorm(hidden_states)
```

**원리:**
- Self-Attention: 100개의 object query들이 서로 관계를 파악합니다. `query_position_embeddings`(학습 가능한 임베딩)을 Q/K에 더합니다.
- Cross-Attention: 각 object query가 인코더 출력(이미지 feature)의 어느 위치를 봐야 할지 결정합니다. query의 learnable position → Q, 이미지의 sinusoidal position → K에 더합니다.
- Decoder는 마지막 레이어 후에만 LayerNorm을 적용합니다(encoder는 각 레이어마다 적용). 이는 DETR 원본 코드 구현을 따릅니다.

**검증 근거:**
- 출력 shape `(B, num_queries=100, d_model=256)`이 Class head/Box head 입력과 정확히 일치합니다.
- Cross-Attention의 K, V source가 동일한 `encoder_hidden_states`에서 오되, K에는 spatial position이 더해지고 V는 순수 feature를 유지합니다.

---

### e) DETR Heads & Complete Architecture

**구현 핵심:**

```python
# DetrMLPPredictionHead (Box head)
self.layers = nn.ModuleList(
    nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
)

def forward(self, x):
    for i, layer in enumerate(self.layers):
        x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
    return x

# DetrForObjectDetection
self.class_labels_classifier = nn.Linear(d_model, num_labels + 1)  # +1: "no object"
self.bbox_predictor = DetrMLPPredictionHead(input_dim=d_model, hidden_dim=d_model, output_dim=4, num_layers=3)

def forward(self, images):
    outputs = self.model(images)
    sequence_output = outputs[0]                                      # (B, 100, 256)
    logits    = self.class_labels_classifier(sequence_output)         # (B, 100, 21)
    pred_boxes = self.bbox_predictor(sequence_output).sigmoid()       # (B, 100, 4) in [0,1]
```

**원리:**
- Class head: Linear 하나로 각 query를 20개 클래스 + 1개 "no object" 중 하나로 분류합니다.
- Box head: 3층 MLP(256→256→256→4)로 bounding box를 예측합니다. 마지막에 `sigmoid`를 적용해 `(cx, cy, w, h)`를 `[0,1]` 범위로 정규화합니다.
- MLP의 마지막 레이어에는 ReLU를 적용하지 않습니다(회귀 출력이므로).

**검증 근거:**
- `num_labels + 1` (VOC2007: 20+1=21)이므로 "no object" 클래스가 명시적으로 포함됩니다.
- `sigmoid` 출력이 `[0,1]`이므로 이미지 크기로 스케일하면 절대 좌표로 변환됩니다.
- Pretrained DETR 가중치를 로드할 때 class head/box head를 제외하고 나머지(backbone + transformer)를 초기화하는 코드와 차원이 일치합니다.
- `post_process`에서 `center_to_corners_format`으로 `(cx,cy,w,h) → (x1,y1,x2,y2)` 변환 후 threshold 필터링이 정상 동작합니다.

---

## Problem 2: Building Loss Function for DETR

### a) Hungarian Matcher

**구현 핵심:**

```python
class_cost = -out_prob[:, target_ids]                                       # (B*Q, N_gt)
bbox_cost  = torch.cdist(out_bbox, target_bbox, p=1)                        # L1 pairwise cost
giou_cost  = -generalized_box_iou(center_to_corners_format(out_bbox),
                                   center_to_corners_format(target_bbox))    # GIoU cost
cost_matrix = class_cost * self.class_cost + bbox_cost * self.bbox_cost + giou_cost * self.giou_cost
```

**원리:**
- 100개 query와 N개 GT object 사이의 pairwise cost matrix를 계산합니다.
- `class_cost`: NLL 손실 대신 `-softmax_prob[target_class]`를 사용합니다. (상수 1은 매칭에 영향 없음)
- `bbox_cost`: 정규화된 `(cx,cy,w,h)` 좌표 간 L1 거리입니다.
- `giou_cost`: GIoU를 최대화(=비용 최소화)하기 위해 음수를 취합니다.
- `linear_sum_assignment`(scipy)로 헝가리안 알고리즘을 적용해 최소 비용 이분 매칭을 수행합니다.

**검증 근거:**
- 매칭 결과가 `(index_i, index_j)` 쌍의 리스트로 반환되며 `len(index_i) = min(num_queries, num_gt)`가 보장됩니다.
- `@torch.no_grad()` 데코레이터로 gradient 계산 없이 순수 매칭만 수행합니다.
- Cost matrix를 배치별로 분리(`cost_matrix.split(sizes, -1)`)해 각 이미지별로 독립 매칭을 수행합니다.

---

### b) Object Loss

**구현 핵심:**

```python
card_pred    = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)   # "no object"가 아닌 예측 수
object_error = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
```

**원리:**
- logits의 마지막 클래스(인덱스 `num_classes`)가 "no object"이므로, 그 외 클래스로 예측된 query 수를 셉니다.
- GT object 수(`target_lengths`)와의 L1 오차를 계산합니다.
- `@torch.no_grad()`: 이 loss는 학습에 사용하지 않고 모니터링 목적으로만 사용합니다.

**검증 근거:**
- `logits.shape[-1] - 1`이 "no object" 인덱스임을 `DetrLoss.__init__`에서 `empty_weight[-1] = eos_coef`로 확인할 수 있습니다.
- L1 loss는 예측된 object 수와 실제 object 수의 절대 오차를 직관적으로 나타냅니다.

---

### c) Class Loss

**구현 핵심:**

```python
target_classes = torch.full(source_logits.shape[:2], self.num_classes,
                             dtype=torch.int64, device=source_logits.device)  # 기본값: "no object"
target_classes[idx] = target_classes_o                                         # 매칭된 query에 실제 label

loss_ce = nn.functional.cross_entropy(
    source_logits.transpose(1, 2), target_classes, self.empty_weight
)
```

**원리:**
- 먼저 모든 query를 "no object"(= `num_classes`)로 초기화합니다.
- 헝가리안 매칭 결과(`idx`)로 선택된 query에만 실제 GT class label을 할당합니다.
- `empty_weight`로 클래스 불균형을 보정합니다: "no object" 가중치는 `eos_coef=0.1`로 낮게 설정해 100개 중 대부분이 "no object"인 상황에서 학습이 "no object"만 예측하는 방향으로 치우치지 않도록 합니다.
- `cross_entropy`는 `(B, num_classes+1, num_queries)` 형태를 요구하므로 `transpose(1, 2)`를 적용합니다.

**검증 근거:**
- `empty_weight` shape `(num_classes+1,)` = `(21,)`이 `cross_entropy`의 `weight` 인자와 정확히 일치합니다.
- `self.register_buffer("empty_weight", ...)`로 `empty_weight`가 모델과 함께 device 이동합니다.
- 매칭된 query 수 = GT object 수이므로, 나머지 (100 - N_gt)개는 모두 "no object" label을 가집니다.

---

### d) Bounding Box Loss

**구현 핵심:**

```python
# L1 Loss
loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction='none')
losses["loss_bbox"] = loss_bbox.sum() / num_boxes

# GIoU Loss
loss_giou = 1 - torch.diag(
    generalized_box_iou(center_to_corners_format(source_boxes),
                         center_to_corners_format(target_boxes))
)
losses["loss_giou"] = loss_giou.sum() / num_boxes
```

**원리:**
- 헝가리안 매칭으로 선택된 query(`source_boxes`)와 대응 GT box(`target_boxes`)만 비교합니다.
- **L1 Loss**: 정규화된 `(cx,cy,w,h)` 좌표 간 절대 오차. 크기에 관계없이 동일한 오차를 패널티로 부여합니다.
- **GIoU Loss**: IoU의 일반화 버전. 두 박스가 겹치지 않아도 gradient를 제공합니다. `torch.diag`로 매칭된 쌍(N×N 행렬의 대각선)만 추출합니다.
- `reduction='none'`으로 원소별 loss를 얻은 뒤 sum/num_boxes로 정규화해 배치 크기와 object 수에 독립적인 loss 스케일을 유지합니다.
- 최종 loss 결합: `loss = loss_ce + 5*loss_bbox + 2*loss_giou`

**검증 근거:**
- `source_boxes`와 `target_boxes`의 shape이 동일한 `(N_matched, 4)`임을 `idx` 인덱싱으로 보장합니다.
- GIoU 계산 전 `center_to_corners_format`으로 변환해 `(x1,y1,x2,y2)` 형식을 맞춥니다.
- `generalized_box_iou` 내부에서 `boxes1[:, 2:] >= boxes1[:, :2]` 조건을 assert해 올바른 corner 형식임을 검증합니다.

---

## Problem 3: Achieving Best Performance

본 섹션에서는 기본 DETR 프레임워크에서 성능을 향상시키기 위한 방법들을 설명합니다.

### 현재 기본 설정
- Backbone: ResNet-50 (pretrained ImageNet)
- Encoder/Decoder: 6 layers, 8 heads, d_model=256
- Epochs: 20, Batch size: 2, LR: 1e-5 (Adam)
- Dataset: VOC2007 (trainval 5011장, test 4952장)

---

### 접근 방법 1: 학습률 스케줄링 (Learning Rate Scheduling)

DETR은 학습 초반에 큰 lr이 필요하고 후반에는 작은 lr이 필요합니다.

```python
# Cosine Annealing + Warmup
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)
```

또는 Step decay:
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

- Backbone 파라미터와 Transformer 파라미터에 다른 학습률 적용:
  ```python
  optimizer = optim.Adam([
      {'params': model.model.backbone.parameters(), 'lr': 1e-5},
      {'params': model.model.encoder.parameters(), 'lr': 1e-4},
      {'params': model.model.decoder.parameters(), 'lr': 1e-4},
      {'params': model.class_labels_classifier.parameters(), 'lr': 1e-4},
      {'params': model.bbox_predictor.parameters(), 'lr': 1e-4},
  ])
  ```

---

### 접근 방법 2: Data Augmentation

DETR 논문에서 사용한 augmentation을 추가합니다.

```python
# Random Horizontal Flip
class RandomHorizontalFlip:
    def __call__(self, image, annots):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            w = annots['orig_size'][1]
            boxes = annots['boxes'].copy()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]  # x1, x2 반전
            annots['boxes'] = boxes
        return image, annots

# Random Scale (multi-scale training)
# shortest edge를 [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800] 중 랜덤 선택
```

---

### 접근 방법 3: Backbone 업그레이드

ResNet-50에서 더 강력한 backbone으로 교체합니다.

```python
# ResNet-101
BACKBONE_MODEL_TYPE = 'resnet101'

# 또는 EfficientNet
BACKBONE_MODEL_TYPE = 'tf_efficientnetv2_s'
```

더 강력한 backbone은 더 풍부한 feature를 제공해 detection 성능을 향상시킵니다.

---

### 접근 방법 4: 에폭 수 증가 및 Pretrained 가중치 활용

DETR은 수렴이 매우 느린 모델입니다. 논문에서는 500 epoch를 사용했습니다.

- 에폭 수를 20 → 50~100으로 증가
- HuggingFace의 `facebook/detr-resnet-50`에서 Transformer 전체 가중치를 초기화하면 VOC2007 fine-tuning 시 빠른 수렴이 가능합니다.

---

### 접근 방법 5: Auxiliary Losses (보조 손실)

DETR 논문에서는 각 디코더 레이어의 출력에도 loss를 적용합니다.

```python
# 각 decoder layer 출력을 intermediate_hidden_states에 저장
# 각각에 대해 class head + box head를 통과시켜 loss 계산
for layer_output in intermediate_hidden_states:
    aux_logits = self.class_labels_classifier(layer_output)
    aux_boxes  = self.bbox_predictor(layer_output).sigmoid()
    aux_loss += loss_fn({'logits': aux_logits, 'pred_boxes': aux_boxes}, targets)
```

보조 손실은 decoder 레이어들이 더 빠르게 수렴하도록 도와줍니다.

---

### 접근 방법 6: Inference 최적화 (Threshold 튜닝)

```python
# confidence threshold를 validation set에서 최적화
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    mAP = evaluate(model, dataloader_val, threshold=threshold)
    print(f"Threshold {threshold}: mAP={mAP:.4f}")
```

---

### 결과 요약

| 방법 | 예상 효과 |
|---|---|
| LR Scheduling | 수렴 안정성 향상, 과적합 감소 |
| Data Augmentation (Flip + Scale) | +2~5% mAP |
| Backbone 업그레이드 (ResNet-101) | +3~7% mAP |
| 에폭 수 증가 (×2~5) | +5~15% mAP (DETR은 epoch 민감) |
| Auxiliary Losses | 수렴 속도 개선, +1~3% mAP |
| Threshold 최적화 | 추론 성능 최적화 |

DETR 모델의 특성상 충분한 학습(에폭 증가)이 가장 큰 성능 향상 요인입니다. 제한된 자원 환경에서는 **LR Scheduling + Horizontal Flip Augmentation** 조합이 가장 효과적인 선택입니다.
