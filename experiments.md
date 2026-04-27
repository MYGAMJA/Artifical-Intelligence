# DETR VOC2007 하이퍼파라미터 실험 기록

## Competition Rules 대응 요약

| 규칙 | 허용 여부 | 적용 내용 |
|---|---|---|
| DETR + ResNet50만 사용 | 필수 | 모든 실험 동일 |
| Test set 학습 금지 | 필수 | `is_train=True` trainval만 학습 |
| 재현성 | 필수 | `set_random_seed(0)` 고정 |
| 데이터 증강 | **허용** | RandomHFlip, ColorJitter |
| LR 스케줄러 | **허용** | StepLR / CosineAnnealingLR |
| Test-Time Augmentation | **허용** | Exp 3에서 활용 |

## 전체 실험 전략

```
Exp 1 (학습 개선)  →  Exp 2 (증강 + 스케줄러 강화)  →  Exp 3 (최고 모델 + TTA 추론)
   기준 설정             더 좋은 학습                    추론 성능 최대화
```

- **Exp 1, 2**: 서로 다른 학습 전략 비교 → 더 좋은 `.pt` 선택
- **Exp 3**: Exp 1, 2 중 best 체크포인트를 로드 + TTA 적용 → 최종 제출 후보

---

## 공통 고정 사항 (모든 실험 동일)
- 모델: DETR ResNet-50 (사전학습 가중치 로드)
- Backbone LR: 1e-5, Transformer LR: 1e-4, weight_decay: 1e-4
- Gradient Clipping: 0.1
- AMP (혼합 정밀도): 활성화
- 랜덤 시드: 0

---

## 실험 1 — 기준 실험 (현재 진행 중)

### 목적
AdamW + StepLR + RandomHFlip 조합의 베이스라인 성능 측정.

### 학습 설정
| 항목 | 값 |
|---|---|
| 데이터 증강 | `RandomHorizontalFlip(prob=0.5)` |
| 옵티마이저 | AdamW |
| LR 스케줄러 | `StepLR(step_size=40, gamma=0.1)` |
| NUM_EPOCHS | 50 |

### 하이퍼파라미터
| 파라미터 | 값 |
|---|---|
| SHORTEST_EDGE | 480 |
| LONGEST_EDGE | 800 |
| BATCH_SIZE | 4 |
| EOS_COEFFICIENT | 0.1 |
| CLASS_COST | 1 |
| BBOX_COST | 5 |
| GIOU_COST | 2 |
| BBOX_LOSS_COEFFICIENT | 5 |
| GIOU_LOSS_COEFFICIENT | 2 |
| CONF_THRESHOLD | 0.5 |

### 노트북 변경 사항 (Cell 33 기준 현재 상태)
변경 없음 — 현재 노트북 그대로 실행.

### 결과
- Best Train Loss: _기록 예정_
- mAP@0.5: _기록 예정_
- 학습 시간: _기록 예정_
- 체크포인트: `detr_voc2007_best.pt` → `detr_voc2007_best_exp1.pt`로 복사

### 관찰 및 메모
_학습 완료 후 기록_

---

## 실험 2 — Loss 가중치 조정 + CosineAnnealing 스케줄러 ✅ 진행 중

### 목적
새 함수 추가 없이 숫자만 변경해서 성능 개선.
- Loss 가중치: 박스 회귀(BBox/GIoU)에 더 집중, no-object 패널티 완화
- `CosineAnnealingLR`로 smooth한 LR 감소 → StepLR의 급격한 감소 대비 안정적 수렴

### 변경 내용 (Cell 33 기준, 이미 적용됨)

**하이퍼파라미터 변경** (Cell 33)
| 파라미터 | Exp 1 | Exp 2 | 이유 |
|---|---|---|---|
| EOS_COEFFICIENT | 0.1 | **0.05** | no-object 패널티 완화 → 실제 객체 검출 강화 |
| BBOX_COST | 5 | **7** | 매칭 시 박스 위치 더 중요하게 |
| GIOU_COST | 2 | **3** | 매칭 시 GIoU 더 중요하게 |
| BBOX_LOSS_COEFFICIENT | 5 | **7** | 학습 시 박스 loss 강화 |
| GIOU_LOSS_COEFFICIENT | 2 | **3** | 학습 시 GIoU loss 강화 |

**스케줄러 변경** (Cell 34)
```python
# Exp 1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
# Exp 2 (현재 적용)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
```

**체크포인트**: `detr_voc2007_best_exp2.pt` / `detr_voc2007_exp2.pt`  
**Exp 1 백업**: `detr_voc2007_exp1.pt` (보존됨)

### 결과
- Best Train Loss: _기록 예정_
- mAP@0.5: _기록 예정_
- 학습 시간: _기록 예정_
- 체크포인트: `detr_voc2007_best_exp2.pt`

### 관찰 및 메모
_학습 완료 후 기록_

---

## 실험 3 — TTA (Test-Time Augmentation) 적용

### 목적
**추가 학습 없이** 추론 단계에서 성능 향상. 규칙 d에서 TTA 명시적 허용.

- Exp 1, 2 중 mAP가 더 높은 체크포인트 로드
- 각 이미지를 **원본 + 좌우반전** 두 번 추론 후 예측 병합
- 재현성 영향 없음 (추론만 변경, 학습 없음)

### TTA 방식
```
원본 이미지 → 추론 → boxes_orig, scores_orig
좌우반전 이미지 → 추론 → boxes_flipped → boxes 좌우 복원
두 결과 합산 → NMS 적용 → 최종 예측
```

박스 복원 공식 (center format, normalized):
```
cx_restored = 1.0 - cx_flipped   # cx만 반전, cy/w/h는 그대로
```

### 노트북 변경 방법

**Cell 37 (mAP 평가 셀) 또는 Cell 43 (submission 셀)**: 추론 루프 교체

```python
def tta_predict(model, images, orig_sizes, conf_threshold):
    """원본 + 좌우반전 TTA 추론"""
    # --- 원본 추론 ---
    with torch.no_grad():
        out_orig = model(images)
    
    # --- 좌우반전 추론 ---
    images_flipped = torch.flip(images, dims=[-1])  # W 축 반전
    with torch.no_grad():
        out_flip = model(images_flipped)
    
    # 반전된 박스의 cx 복원
    for item in out_flip:
        boxes = item['boxes']          # [N, 4] center format (cx,cy,w,h), normalized
        boxes[:, 0] = 1.0 - boxes[:, 0]
        item['boxes'] = boxes
    
    # 두 결과 병합 (concat 후 NMS는 기존 post_process에서 처리)
    merged = []
    for o, f in zip(out_orig, out_flip):
        merged.append({
            'boxes': torch.cat([o['boxes'], f['boxes']], dim=0),
            'scores': torch.cat([o['scores'], f['scores']], dim=0),
            'labels': torch.cat([o['labels'], f['labels']], dim=0),
        })
    return merged
```

### 사용할 체크포인트
Exp 1 mAP vs Exp 2 mAP 비교 후 더 높은 것 선택:
```python
state_dict = torch.load('detr_voc2007_best_exp1.pt')  # 또는 exp2
```

### 결과
- mAP@0.5 (원본 추론): _기록 예정_
- mAP@0.5 (TTA 적용): _기록 예정_
- TTA 향상폭: _기록 예정_

### 관찰 및 메모
_완료 후 기록_

---

## 최종 비교 요약

| 실험 | 핵심 기법 | Best Loss | mAP@0.5 | 비고 |
|---|---|---|---|---|
| Exp 1 | HFlip + StepLR | - | - | baseline |
| Exp 2 | HFlip + ColorJitter + CosineAnnealingLR | - | - | 강화된 학습 |
| Exp 3 | Exp1/2 best + TTA | 재학습 없음 | - | 추론 최적화 |

**최종 제출**: Exp 3의 mAP가 Exp 1/2보다 높으면 Exp 3, 아니면 Exp 1/2 중 최고값 제출

