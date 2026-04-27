# Assignment 2 — DETR 전체 구조 설명

## 전체 흐름 한눈에 보기

```
[환경 설정] → [데이터셋] → [전처리] → [모델 구현 ★] → [손실 함수 ★] → [학습] → [평가/시각화] → [제출]
```

---

## 섹션별 설명

| 섹션 | 셀 번호 | 내용 |
|------|--------|------|
| **환경 설정** | 2 ~ 5 | 라이브러리 설치(pip), 패키지 import, 랜덤 시드 고정 |
| **VOC2007 데이터셋** | 7 ~ 9 | Pascal VOC 2007 데이터셋 클래스 정의, train/test 로드 |
| **시각화** | 11 ~ 12 | 이미지에 GT 박스를 그려주는 `show_sample` 함수 |
| **이미지 전처리** | 14 | Resize, Normalize 등 transform 클래스 정의 |
| **하이퍼파라미터 & DataLoader** | 16 | 모델/학습에 쓸 숫자값 상수들, DataLoader 생성 |
| **DETR 모델 구현 ★** | 18 ~ 26 | 백본, 포지셔널 인코딩, Attention, Encoder, Decoder, Head 구현 |
| **손실 함수 ★** | 28 ~ 30 | IoU 유틸, Hungarian Matcher, DetrLoss 구현 |
| **학습** | 32 ~ 34 | Pretrained 가중치 로드, 학습 루프 실행 |
| **평가 & 시각화** | 36 ~ 41 | mAP 계산, GT/예측 결과 시각화 |
| **제출** | 43 | Kaggle 제출용 CSV 파일 생성 |

---

## TODO 목록 — 직접 구현해야 하는 부분

### ✅ TODO 1 — `DetrSinePositionEmbedding.forward` (셀 18)

**역할**: 이미지의 2D 위치 정보를 Transformer에 전달하기 위한 **sin/cos 포지셔널 인코딩** 계산

```
x, y 누적합 → 주파수 계산(dim_t) → 각도 계산 → sin/cos 적용 → (B, d_model, H, W) 반환
```

- CNN 백본 출력 feature map의 각 픽셀 위치를 `d_model`차원 벡터로 인코딩한다
- x방향 128차원 + y방향 128차원 = 256차원(`d_model`)
- Transformer는 순서 개념이 없기 때문에 이 인코딩을 더해줘야 공간 위치를 알 수 있다

---

### ✅ TODO 2 — `DetrAttention.__init__` + `forward` (셀 20)

**역할**: DETR의 **Multi-Head Attention** 구현 (Self-Attention과 Cross-Attention 모두 처리)

```
Q, K, V projection 레이어 정의 → Q/K에 position embedding 더하기 → QK^T → softmax → V 가중합 → out_proj
```

- `__init__`: `k_proj`, `v_proj`, `q_proj`, `out_proj` 4개의 Linear 레이어 정의
- `forward`: Q와 K에만 position embedding을 더하고, V는 순수 feature를 그대로 사용
- Encoder에서는 Self-Attention(이미지 feature끼리), Decoder에서는 Cross-Attention(query ↔ image feature)으로 사용

---

### ✅ TODO 3 — `DetrEncoderLayer` + `DetrEncoder` (셀 22)

**역할**: **Transformer Encoder** 구현 (이미지 feature를 문맥에 맞게 변환)

```
Self-Attention → Residual + LayerNorm → FFN(Linear→ReLU→Linear) → Residual + LayerNorm
```

- `DetrEncoderLayer.__init__`: self_attn, LayerNorm 2개, fc1/fc2(FFN) 정의
- `DetrEncoderLayer.forward`: Attention + Dropout + Residual + LN + FFN + Dropout + Residual + LN 순서로 처리
- `DetrEncoder.forward`: 6개 레이어를 순서대로 통과시키는 루프 작성

---

### ✅ TODO 4 — `DetrDecoderLayer` + `DetrDecoder` (셀 24)

**역할**: **Transformer Decoder** 구현 (object query들이 이미지에서 객체를 찾아내는 과정)

```
Self-Attention(query끼리) → Cross-Attention(query ↔ encoder 출력) → FFN
```

- `DetrDecoderLayer.__init__`: self_attn, encoder_attn(cross-attention), LayerNorm 3개, fc1/fc2 정의
- `DetrDecoderLayer.forward`:
  - **Self-Attention**: 100개 object query들이 서로 정보를 교환
  - **Cross-Attention**: 각 query가 인코더 출력(이미지 feature)에서 관련 정보를 가져옴
- `DetrDecoder.__init__`: 마지막에 한 번만 쓸 `LayerNorm` 정의
- `DetrDecoder.forward`: 6개 레이어 통과 후 최종 LayerNorm 적용

---

### ✅ TODO 5 — `DetrModel` + `DetrMLPPredictionHead` + `DetrForObjectDetection` (셀 26)

**역할**: **완성된 DETR 모델** 조립 + 예측 헤드 구현

```
입력 이미지 → Backbone → Projection(2048→256) → Encoder → Decoder → Class Head / Box Head → 출력
```

- `DetrModel.forward`: 백본 feature에 1×1 conv 투영 → flatten → Encoder → Decoder 순서로 연결
- `DetrMLPPredictionHead.__init__`: 3층 MLP 레이어 목록(`nn.ModuleList`) 정의
- `DetrMLPPredictionHead.forward`: 마지막 레이어 빼고 모두 ReLU 적용
- `DetrForObjectDetection.__init__`: Class head(`Linear(256→21)`) + Box head(`MLP(256→256→256→4)`) 정의
- `DetrForObjectDetection.forward`: Decoder 출력 → logits(클래스) + `sigmoid` 박스 좌표 예측

---

### ✅ TODO 6 — `HungarianMatcher.forward` (셀 29)

**역할**: 예측 100개와 GT N개 사이의 **최적 1:1 매칭** 찾기 (헝가리안 알고리즘)

```
Class cost + BBox L1 cost + GIoU cost → 비용 행렬 → scipy.linear_sum_assignment → 최적 매칭 쌍 반환
```

- `class_cost`: 예측 softmax 확률에서 정답 클래스의 확률을 음수로 취함
- `bbox_cost`: 예측 박스와 GT 박스의 L1 거리
- `giou_cost`: GIoU를 최대화하기 위해 음수로 취함
- 세 cost를 가중 합산해 cost matrix를 만들고 헝가리안 알고리즘으로 최소 비용 매칭을 구함

---

### ✅ TODO 7 — `DetrLoss.compute_loss_labels` + `compute_loss_object` + `compute_loss_boxes` (셀 30)

**역할**: 헝가리안 매칭 결과를 바탕으로 **3가지 손실 계산**

| 손실 함수 | 내용 |
|-----------|------|
| `compute_loss_labels` | 매칭된 query에 GT 클래스 label 배정 → Cross-Entropy Loss (클래스 불균형은 `empty_weight`로 보정) |
| `compute_loss_object` | 예측된 object 수와 실제 GT object 수의 L1 차이 (모니터링용, gradient 없음) |
| `compute_loss_boxes` | 매칭된 박스 간 **L1 Loss** + **GIoU Loss** 계산 |

---

## DETR 전체 데이터 흐름 요약

```
이미지 (B, 3, H, W)
    │
    ▼ [ResNet-50 백본]
Feature Map (B, 2048, H/32, W/32)
    │
    ▼ [1×1 Conv 투영]
Projected (B, 256, H/32, W/32)
    │  + Positional Encoding (TODO 1)
    ▼ [flatten → (B, H/32*W/32, 256)]
    │
    ▼ [Transformer Encoder × 6 (TODO 3)]
Encoder Output (B, seq_len, 256)
    │
    ▼ [Transformer Decoder × 6 (TODO 4)]  ← Object Queries (B, 100, 256)
Decoder Output (B, 100, 256)
    │
    ├─▶ [Class Head: Linear (TODO 5)] → Logits (B, 100, 21)
    └─▶ [Box Head: MLP + sigmoid (TODO 5)] → Boxes (B, 100, 4)
    │
    ▼ [Hungarian Matcher (TODO 6)]
예측 ↔ GT 1:1 매칭
    │
    ▼ [Loss 계산 (TODO 7)]
Class Loss + BBox L1 Loss + GIoU Loss
```
