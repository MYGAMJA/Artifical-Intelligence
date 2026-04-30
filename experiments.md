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
Exp 1 (기준 실험)  →  Exp 2 (Loss 가중치 + CosineAnnealingLR)  →  Exp 3 (해상도 + LR 조정)
   베이스라인 설정        회귀 loss 강화 + 스케줄러 개선              더 큰 이미지 + 낮은 LR
```

- **Exp 1**: AdamW + StepLR + RandomHFlip 기준 실험
- **Exp 2**: Loss 가중치 조정 + CosineAnnealingLR로 수렴 개선
- **Exp 3**: 해상도 향상 + 학습률 절반 + EOS 완화 + CONF_THRESHOLD 낮춤

---

## 공통 고정 사항 (모든 실험 동일)
- 모델: DETR ResNet-50 (사전학습 가중치 로드)
- Gradient Clipping: 0.1
- AMP (혼합 정밀도): 활성화
- 랜덤 시드: 0
- NUM_EPOCHS: 50, BATCH_SIZE: 4

---

## 실험 1 — 기준 실험

### 목적
AdamW + StepLR + RandomHFlip 조합의 베이스라인 성능 측정.

### 학습 설정
| 항목 | 값 |
|---|---|
| 데이터 증강 | 없음 |
| 옵티마이저 | AdamW (backbone 1e-5, transformer 1e-4, weight_decay 1e-4) |
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

### 노트북
`assignment2_DETR_exp1.ipynb` — 체크포인트: `detr_voc2007_best_exp1.pt`

### 결과
- Best Train Loss: _기록 예정_
- mAP@0.5: _기록 예정_
- 학습 시간: _기록 예정_
- 체크포인트: `detr_voc2007_best.pt` → `detr_voc2007_best_exp1.pt`로 복사

### 관찰 및 메모
_학습 완료 후 기록_

---

## 실험 2 — Loss 가중치 조정 + CosineAnnealing 스케줄러

### 목적
새 함수 추가 없이 숫자만 변경해서 성능 개선.
- Loss 가중치: 박스 회귀(BBox/GIoU)에 더 집중, no-object 패널티 완화
- `CosineAnnealingLR`로 smooth한 LR 감소 → StepLR의 급격한 감소 대비 안정적 수렴

### 변경 내용 (Exp 1 대비)

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
# Exp 2
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
```

### 노트북
`assignment2_DETR_exp2.ipynb` — 체크포인트: `detr_voc2007_best_exp2.pt`

### 결과
- Best Train Loss: _기록 예정_
- mAP@0.5: _기록 예정_
- 학습 시간: _기록 예정_
- 체크포인트: `detr_voc2007_best_exp2.pt`

### 관찰 및 메모
_학습 완료 후 기록_

---

## 실험 3 — 해상도 향상 + 학습률 조정 + 추론 임계값 완화

### 목적
이미지 해상도를 높여 작은 객체 검출 강화, 학습률을 낮춰 pretrained 가중치 보존, confidence threshold를 낮춰 recall 향상.

### 변경 내용 (Exp 2 대비)
| 파라미터 | Exp 1 | Exp 2 | **Exp 3** | 이유 |
|---|---|---|---|---|
| SHORTEST_EDGE | 480 | 480 | **600** | 더 높은 해상도 → 작은 객체 검출 강화 |
| LONGEST_EDGE | 800 | 800 | **1000** | 비율 유지 |
| backbone lr | 1e-5 | 1e-5 | **5e-6** | pretrained features 더 조심스럽게 fine-tuning |
| transformer lr | 1e-4 | 1e-4 | **5e-5** | 전체적 LR 절반으로 안정적 수렴 |
| EOS_COEFFICIENT | 0.1 | 0.05 | **0.02** | no-object 패널티 더 완화 |
| CONF_THRESHOLD | 0.5 | 0.5 | **0.3** | recall 향상, 더 많은 검출 허용 |
| BBOX_COST | 5 | 7 | 7 (유지) | |
| GIOU_COST | 2 | 3 | 3 (유지) | |
| 스케줄러 | StepLR | CosineAnnealingLR | CosineAnnealingLR (유지) | |
| eta_min | - | 1e-6 | **1e-7** | 더 낮게 수렴 |

### 노트북
`assignment2_DETR_exp3.ipynb` — 체크포인트: `detr_voc2007_best_exp3.pt`

### 주의사항
- 해상도 증가로 GPU 메모리 사용량 증가 → OOM 발생 시 BATCH_SIZE=2로 줄이기
- CONF_THRESHOLD=0.3은 mAP 계산에 영향 줌 (더 많은 FP 가능)

### 결과
- Best Train Loss: _기록 예정_
- mAP@0.5: _기록 예정_
- 학습 시간: _기록 예정_

### 관찰 및 메모
_학습 완료 후 기록_

---

## 최종 비교 요약

| 실험 | 핵심 변경 | Best Loss | mAP@0.5 | 비고 |
|---|---|---|---|---|
| Exp 1 | AdamW + StepLR | - | - | baseline |
| Exp 2 | Loss weights 강화 + CosineAnnealingLR | - | - | 스케줄러 개선 |
| Exp 3 | 해상도↑ + LR↓ + EOS↓ + CONF↓ | - | - | 전반적 튜닝 |

**최종 제출**: 세 실험 중 mAP@0.5 가장 높은 것 제출

