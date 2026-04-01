# Assignment #1-1 - Problem #3 Report

## 개요
이 문서는 `assignment1_RetinaNet_ver2.ipynb`의 **Problem #3 (Training: BCE Loss, Focal Loss)** 내용을 정리한 보고서입니다.

---

## Problem #3-a) BCE Loss

### BCE(Binary Cross Entropy) 의미
- 이진 정답(0/1)과 예측 확률의 차이를 측정하는 손실 함수
- 식:
  - $L = -\left(y\log(p) + (1-y)\log(1-p)\right)$

### RetinaNet 분류에서의 적용
- `classification`: 각 anchor의 클래스별 예측 확률
- `targets`: anchor별 정답 라벨(양성=1, 음성=0, ignore=-1)
- 계산:
  - `bce = -(targets * log(classification) + (1-targets) * log(1-classification))`
- `targets == -1`인 anchor는 손실에서 제외

### BCE 학습 스텝 구현 요약
- `optimizer.zero_grad()`
- `classification_loss, regression_loss = retinanet([...])`
- `loss = classification_loss.mean() + regression_loss.mean()`
- `loss.backward()`
- `clip_grad_norm_`
- `optimizer.step()`

---

## Problem #3-b) Focal Loss

### Focal Loss를 쓰는 이유
- 객체 탐지에서는 배경(anchor 음성)이 객체(anchor 양성)보다 훨씬 많음
- BCE만 사용하면 쉬운 음성 샘플이 손실을 지배할 수 있음

### 핵심 아이디어
- 잘 맞춘 샘플의 손실은 줄이고, 어려운 샘플의 손실은 상대적으로 크게 반영
- 분류 불균형(특히 배경 과다 문제) 완화

### 구현 관점 요약
- BCE 항에 가중치(`focal_weight`)를 곱해 hard example 중심으로 학습
- RetinaNet 논문 설정($\alpha$, $\gamma$) 기반으로 분류 손실 조정

---

## 결론
- BCE는 기본적인 확률 분류 손실로 직관적이고 구현이 단순함
- Focal Loss는 BCE를 확장해 객체 탐지의 클래스 불균형 문제를 더 효과적으로 다룸
- Problem #3에서는 두 손실을 통해 RetinaNet 학습 전략의 차이를 확인할 수 있음
