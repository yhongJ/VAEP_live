# VAEP 축구 영상 분석 시스템

축구 영상을 업로드하면 선수별 VAEP(Value Added by Each Player Action)를 자동 계산하고, 분석 오버레이가 삽입된 새 영상을 출력하는 풀파이프라인 시스템입니다.

---

## 목차

1. [개요](#개요)
2. [파일 구성](#파일-구성)
3. [아키텍처](#아키텍처)
4. [VAEP 모델 설명](#vaep-모델-설명)
5. [설치 방법](#설치-방법)
6. [실행 방법](#실행-방법)
7. [API 명세](#api-명세)
8. [출력 결과](#출력-결과)
9. [한계 및 주의사항](#한계-및-주의사항)

---

## 개요

| 항목 | 내용 |
|------|------|
| 선수 탐지 | YOLOv8n (fallback: 색상 기반 배경차분) |
| 팀 분류 | HSV Hue K-means 클러스터링 (캘리브레이션 8프레임) |
| 선수 추적 | 헝가리안 매칭 Re-ID (최대 24트랙) |
| 볼 탐지 | Hough Circle + 칼만 필터 |
| VAEP 모델 | 온볼(패스/드리블/슛/태클) + 오프더볼(공간창출/압박/런) |
| 오버레이 | OpenCV + Pillow (한글 NanumGothic 폰트) |
| 서버 | Flask REST API, 멀티스레딩 |

---

## 파일 구성

```
.
├── server.py            # Flask API 서버 (업로드 → 분석 → 다운로드)
├── video_processor.py   # 비디오 처리 파이프라인 (탐지/추적/분석)
├── vaep_engine.py       # VAEP 계산 엔진 (온볼 + 오프더볼)
├── tracker.py           # 선수 추적 및 팀 색상 분류
├── overlay_renderer.py  # VAEP 오버레이 영상 렌더러
└── static/
    └── index.html       # 웹 UI
```

---

## 아키텍처

```
[ 영상 업로드 ]
      |
      v
[ VideoProcessor ]
  |- BallDetector         Hough Circle + 칼만 필터
  |- TeamColorClassifier  HSV K-means (8프레임 캘리브레이션)
  |- PlayerTracker        헝가리안 매칭, 최대 24 트랙
  |- PitchTransformer     픽셀 좌표 → 피치 좌표 (0~105, 0~68)
  |- ActionDetector       패스 / 드리블 / 슛 / 태클 탐지
  └- VAEPAnalyzer         온볼 + 오프더볼 VAEP 계산
      |
      v
[ result.json ]  frame_data, events, final_vaep 포함
      |
      v
[ OverlayRenderer ]
  |- 선수 원 + 번호        OpenCV
  |- 점선 액션 화살표      OpenCV
  |- VAEP 레이블 박스      Pillow (한글)
  |- TOP5 패널            Pillow (한글)
  |- 이벤트 배너           Pillow (한글)
  └- 스코어보드            Pillow (한글)
      |
      v
[ vaep_analysis_<job_id>.mp4 ]
```

---

## VAEP 모델 설명

### 참고 문헌

- Decroos et al. (2019) "Actions Speak Louder than Goals: Valuing Player Actions in Soccer"
- Singh (2019) "Introducing Expected Threat (xT)"

### 온볼 VAEP

```
VAEP_온볼 = (V_score_after - V_score_before) * w_score
          - (V_concede_after - V_concede_before) * w_concede
```

액션 전후 위치의 득점 기대값과 실점 기대값 차이를 계산합니다.

| 액션 | 득점 가중치 | 실점 가중치 |
|------|------------|------------|
| pass    | 1.0 | 0.8 |
| dribble | 1.2 | 0.6 |
| shot    | 1.5 | 0.2 |
| cross   | 1.1 | 0.7 |
| tackle  | 0.5 | 1.3 |

득점 확률 모델은 구역별 로지스틱 파라미터를 사용합니다.

| 구역 | x 범위 |
|------|--------|
| 페널티박스 내 | x > 83m |
| 파이널 서드 위험구역 | 70 < x <= 83m |
| 공격 절반 | 52.5 < x <= 70m |
| 수비 절반 | x <= 52.5m |

실패한 액션의 경우 `V_score_after = V_score_before * 0.05`, 실점 확률에 1.8 패널티 적용.

### 오프더볼 VAEP

볼을 갖지 않은 선수의 기여도를 세 가지로 분해합니다.

**공간 창출 (Space Creation)**
선수 이동으로 수비수가 끌려오면, 해방된 동료 위치의 득점 확률 * 수비수 이동 정도로 산출. 상한 0.15.

**압박 가치 (Pressing Value)**
볼 소유자 8m 이내 압박 시 계산. 압박 강도 * 볼 위치 위험도 * (1 + 추가 압박자 수 * 0.2) * 0.35. 상한 0.15.

**런 가치 (Run Value)**
공격 방향 이동 시 위치 가치 상승분. 수비수 등 뒤 공간 진입 시 수비수당 +0.025 보너스. 상한 0.20.

**통합 VAEP:**
```
VAEP_total = VAEP_온볼 + VAEP_오프더볼 * 0.25
```

누적 VAEP는 선수별 최근 20개 이벤트 합산값을 사용합니다.

### 액션 탐지 기준

| 액션 | 조건 |
|------|------|
| 패스 | 볼 소유자가 같은 팀 다른 선수로 변경 |
| 드리블 | 볼 소유자 동일 + 이동거리 >1m + 근처 수비수 1명 이상 |
| 슛 | 볼 속도 >8.0 + x>70m + 전진 방향 |
| 태클 | 볼 소유자가 다른 팀으로 변경 |

---

## 설치 방법

```bash
pip install flask flask-cors opencv-python-headless pillow numpy scipy ultralytics

# 한글 폰트 (Linux)
sudo apt-get install fonts-nanum
```

YOLO 모델(`yolov8n.pt`)은 ultralytics 첫 실행 시 자동 다운로드됩니다.

지원 OS: Linux / Windows (폰트 경로 자동 탐색)

---

## 실행 방법

```bash
python server.py
# http://localhost:5000
```

브라우저에서 `http://localhost:5000` 접속 후 영상 업로드.

Python에서 직접 호출하는 경우:

```python
from video_processor import VideoProcessor
from overlay_renderer import render_vaep_overlay

processor = VideoProcessor(use_yolo=True)
result = processor.process_video("match.mp4", sample_rate=3)
render_vaep_overlay("match.mp4", result, "output_vaep.mp4")
```

`sample_rate=3`은 3프레임마다 1프레임 처리(약 10fps 분석)를 의미하며, 처리 속도와 정확도의 트레이드오프입니다.

---

## API 명세

### POST /api/upload

영상 업로드 및 분석 작업 시작.

Form Data:

| 필드 | 타입 | 설명 |
|------|------|------|
| video | File | mp4/avi/mov/mkv/webm, 최대 2GB |
| show_names | string | "1" = 선수 이름 표시 |
| player_names | JSON string | {"1": "홍길동", "2": "김선수"} |
| color_a | string | 팀A 색상 hex (기본 #dc2626) |
| color_b | string | 팀B 색상 hex (기본 #3b82f6) |

Response:
```json
{ "job_id": "a1b2c3d4", "status": "queued" }
```

### GET /api/status/\<job_id\>

작업 진행 상태 조회.

```json
{
  "status": "processing",
  "progress": 72,
  "step": "오버레이 영상 렌더링 중..."
}
```

`status` 값: `queued` / `processing` / `done` / `error`

### GET /api/result/\<job_id\>

분석 결과 JSON 반환. `status=done`일 때만 유효.

```json
{
  "total_frames": 9000,
  "processed_frames": 3000,
  "fps": 30.0,
  "duration": 300.0,
  "total_events": 412,
  "final_vaep": { "0": 1.23, "1": 0.87 },
  "frame_data": [...],
  "events": [...]
}
```

### GET /api/download/\<job_id\>

VAEP 오버레이가 삽입된 mp4 파일 다운로드.

---

## 출력 결과

오버레이 영상에 포함되는 요소:

| 요소 | 설명 |
|------|------|
| 선수 원 | 팀별 색상, 볼 소유자는 크기 및 테두리 강조 |
| VAEP 레이블 | 누적 VAEP 상위 5명 + 볼 소유자에게 수치 표시 |
| 좌상단 패널 | 누적 VAEP TOP 5 선수 순위 |
| 하단 배너 | 이벤트 발생 시 4초간 액션 설명 및 VAEP 수치 표시 |
| 스코어보드 | 홈/원정 팀명 및 현재 스코어 |
| 점선 화살표 | 패스/드리블 등 액션 방향 애니메이션 |

---

## 한계 및 주의사항

**카메라 각도**
단일 고정 카메라를 가정합니다. 팬/줌이 많은 중계 영상에서는 호모그래피 정확도가 떨어집니다.

**볼 탐지**
Hough Circle 기반으로 부정확한 경우가 있으며, 볼 미탐지 시 화면 중앙 좌표로 대체합니다.

**선수 분류**
선수들끼리 겹쳐있거나 동일 선상에서 뛰는 경우 오분류가 발생할 수 있습니다.

**팀 분류**
유니폼 색상이 유사하거나 조명이 불균일한 경우 오분류가 발생할 수 있습니다.

**선수 번호**
실제 등번호가 아닌 추적 ID 기반 번호가 표시됩니다.

**VAEP 수치**
실제 이벤트 데이터 기반 통계 모델이 아닌 규칙 기반 근사치입니다.