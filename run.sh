#!/usr/bin/env bash
# ============================================================
# VAEP 실시간 축구 분석 시스템 - 설치 및 실행 스크립트
# ============================================================

set -e

echo "======================================"
echo "  VAEP 축구 분석 시스템 설치 중..."
echo "======================================"

# Python 패키지 설치
pip install flask flask-cors ultralytics opencv-python-headless scipy --break-system-packages -q

echo "[✓] Python 패키지 설치 완료"

# 디렉토리 생성
mkdir -p uploads output static

echo "[✓] 디렉토리 생성 완료"

# YOLO 모델 사전 다운로드 (선택)
python3 -c "
from ultralytics import YOLO
print('YOLO 모델 다운로드 중...')
m = YOLO('yolov8n.pt')
print('[✓] YOLO v8n 모델 준비 완료')
" 2>/dev/null || echo "[!] YOLO 모델은 첫 실행 시 자동 다운로드됩니다"

echo ""
echo "======================================"
echo "  서버 시작 중..."
echo "  브라우저에서: http://localhost:5000"
echo "======================================"
echo ""

cd "$(dirname "$0")"
python3 server.py
