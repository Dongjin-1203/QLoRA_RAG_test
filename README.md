---
title: RFPilot Model Comparison
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# QLoRA_RAG_test
---
이 프로젝트는 RFP 문서요약 RAG 챗봇 프로젝트의 후속 연구 입니다.

## 문제
기존 서비스 구조에서 Fine-Tuning된 모델에 RAG 시스템을 적용하였던 것이 과적합을 야기하는지, 어떤 효과가 있는지 확인 하지 못하였다.

## 실험 절차

- QLoRA 된 모델을 준비한다.
- Fine-Tuning 하지 않은 원본 모델을 준비한다.
- 평가 데이터셋을 생성한다.
- Fine-Tuning을 한 경우, Fine-Tuning을 하지 않고 RAG만 적용한 경우, 둘 다 적용한 경우를 나눠 테스트를 해본다.
- 결과를 확인 한다.

---

## 결과