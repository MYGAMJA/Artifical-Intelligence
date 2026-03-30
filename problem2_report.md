# Assignment #1-1 - Problem #2 Report

## 개요
이 문서는 `assignment1_RetinaNet_ver2.ipynb`의 **Problem #2 (Building Models and Anchors)** 구현 내용을 정리한 보고서입니다.  
Problem #2 요구사항(a~e)에 맞춰 구현한 모듈의 목적, 동작, 입력/출력 형태를 초보자 관점에서 설명합니다.

---

## Problem #2-a) FPN(Feature Pyramid Network) 구조 구현
### 요구사항
- ResNet101 백본의 C3, C4, C5 특징맵을 입력으로 받아 P3~P7 피라미드 특징맵 생성

### 구현 요약
- 클래스: `PyramidFeatures`
- 핵심 구성:
  - `P5_1 (1x1 conv)`, `P5_2 (3x3 conv)`, `P5 upsample(nearest)`
  - `P4_1 (1x1 conv)`, `P4_2 (3x3 conv)`, `P4 upsample(nearest)`
  - `P3_1 (1x1 conv)`, `P3_2 (3x3 conv)`
  - `P6 (C5에서 stride=2 3x3 conv)`
  - `P7 (ReLU 후 stride=2 3x3 conv)`

### 초보자 포인트
- `1x1 conv`는 채널 수를 맞추기 위한 변환
- 상위 레벨 특징(P5)을 업샘플해서 하위 레벨(C4/C3)과 더함
- 마지막 `3x3 conv`는 합쳐진 특징을 부드럽게 정리하는 역할

---

## Problem #2-b) FPN forward pass 구현
### 구현 흐름
1. 입력: `[C3, C4, C5]`
2. `P5 = Conv1x1(C5) -> upsample`
3. `P4 = Conv1x1(C4) + upsample(P5) -> Conv3x3 -> upsample`
4. `P3 = Conv1x1(C3) + upsample(P4) -> Conv3x3`
5. `P6 = Conv3x3_stride2(C5)`
6. `P7 = Conv3x3_stride2(ReLU(P6))`
7. 출력: `[P3, P4, P5, P6, P7]`

### 출력 의미
- 서로 다른 해상도의 5개 특징맵을 만들어 작은 물체/큰 물체를 동시에 탐지할 수 있도록 함

---

## Problem #2-c) Regression Model(box subnet) 구조 + forward 구현
### 요구사항
- 각 anchor에 대해 박스 오프셋 4개(dx, dy, dw, dh) 예측

### 구현 요약
- 클래스: `RegressionModel`
- 구조:
  - `3x3 conv + ReLU` 4회
  - 최종 `3x3 conv` 출력 채널: `num_anchors * 4`
- forward:
  - 출력 텐서 순서를 `(N, H, W, C)`로 바꾸고
  - `(N, -1, 4)` 형태로 reshape

### 출력 형태
- `batch_size x 전체_anchor_개수 x 4`

---

## Problem #2-d) Classification Model(class subnet) 구조 + forward 구현
### 요구사항
- 각 anchor가 어떤 클래스인지 확률 예측

### 구현 요약
- 클래스: `ClassificationModel`
- 구조:
  - `3x3 conv + ReLU` 4회
  - 최종 `3x3 conv` 출력 채널: `num_anchors * num_classes`
- forward:
  - `(N, H, W, C)`로 permute
  - `(N, -1, num_classes)`로 reshape
  - `sigmoid()`로 클래스 확률화

### 출력 형태
- `batch_size x 전체_anchor_개수 x num_classes`

---

## Problem #2-e) `BBoxTransform`, `Anchors`, `generate_anchors()`, `shift()` 구현

### 1) `BBoxTransform`
#### 목적
- 회귀 헤드가 예측한 오프셋(`deltas`)을 실제 박스 좌표 `(x1, y1, x2, y2)`로 변환

#### 핵심 계산
- anchor 중심좌표/너비/높이 계산
- `dx,dy,dw,dh`에 std/mean 반영
- 예측 중심과 크기 계산 후 모서리 좌표로 변환

---

### 2) `Anchors` 클래스
#### 목적
- FPN의 각 피라미드 레벨(P3~P7)에서 사용할 모든 anchor를 한 번에 생성

#### 구현 포인트
- 기본 피라미드 레벨: `[3, 4, 5, 6, 7]`
- stride: `[8, 16, 32, 64, 128]`
- base size: `[32, 64, 128, 256, 512]`
- ratio: `[0.5, 1, 2]`
- scale: `[1, 2^(1/3), 2^(2/3)]`

---

### 3) `generate_anchors(base_size, ratios, scales)`
#### 목적
- 한 feature-map cell 기준의 기본(anchor template) 박스들 생성

#### 동작
- ratio/scale 조합 수 만큼 anchor 생성
- 중심 기준 좌표(대칭) 형태로 변환

---

### 4) `shift(shape, stride, anchors)`
#### 목적
- 템플릿 anchor를 feature-map의 모든 위치로 이동(tiling)

#### 동작
- 각 cell 중심 좌표 grid 생성
- 모든 템플릿 anchor에 grid shift를 더해 전체 anchor 생성

---

## ResNet RetinaNet 구성요소 연결(Problem #2 연계)
`ResNet` 클래스 내부 TODO에서 아래를 연결했습니다.
- `self.fpn = PyramidFeatures(...)`
- `self.regressionModel = RegressionModel(256)`
- `self.classificationModel = ClassificationModel(256, num_classes=num_classes)`
- `self.anchors = Anchors()`
- `self.regressBoxes = BBoxTransform()`

즉, 백본 특징맵 -> FPN -> (회귀/분류 헤드) -> anchor 생성/박스 변환으로 이어지는 RetinaNet 기본 파이프라인이 완성됩니다.

---

## 구현 체크 요약 (Problem #2 기준)
- [x] FPN 구조 구현
- [x] FPN forward 구현
- [x] Regression subnet 구조 + forward 구현
- [x] Classification subnet 구조 + forward 구현
- [x] `BBoxTransform` 구현
- [x] `Anchors`, `generate_anchors()`, `shift()` 구현
- [x] RetinaNet 본체에서 위 모듈 연결 완료

---

## 참고
- 본 보고서는 Problem #2 구현 내용만 다룹니다.
- 학습 루프(BCE/Focal), 실험 비교는 Problem #3 범위입니다.