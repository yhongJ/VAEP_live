"""
비디오 프로세서: YOLO 기반 선수 탐지 + 액션 인식 + 피치 변환
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import math
import json
import os
from dataclasses import asdict

from tracker import PlayerTracker, TeamColorClassifier
from vaep_engine import (
    FrameState, PlayerState, ActionEvent, VAEPAnalyzer
)


# ──────────────────────────────────────────────
# 색상 기반 팀 분류 (jersey color clustering)
# ──────────────────────────────────────────────

class TeamClassifier:
    def __init__(self):
        self.team_colors = {}  # team_id -> (H, S, V) 중심값
        self.calibrated = False
        self.color_samples = {0: [], 1: []}

    def extract_jersey_color(self, frame: np.ndarray, bbox: Tuple) -> Tuple[float, float, float]:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h = y2 - y1
        # 상체 영역 (유니폼)
        torso_y1 = y1 + int(h * 0.15)
        torso_y2 = y1 + int(h * 0.55)
        torso_x1 = x1 + int((x2 - x1) * 0.2)
        torso_x2 = x2 - int((x2 - x1) * 0.2)

        if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
            return (0, 0, 0)

        torso = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        if torso.size == 0:
            return (0, 0, 0)

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        # 녹색(잔디) 마스크 제거
        grass_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        non_grass = cv2.bitwise_not(grass_mask)
        pixels = hsv[non_grass > 0]

        if len(pixels) < 10:
            return (0, 0, 0)

        mean_color = np.mean(pixels, axis=0)
        return tuple(mean_color)

    def classify(self, frame: np.ndarray,
                 detections: List[Tuple]) -> Dict[int, int]:
        """탐지된 선수들을 두 팀으로 분류"""
        colors = []
        for i, det in enumerate(detections):
            color = self.extract_jersey_color(frame, det[:4])
            colors.append(color)

        if len(colors) < 2:
            return {i: 0 for i in range(len(detections))}

        # K-means로 두 팀 클러스터링
        color_array = np.array(colors, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            color_array, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
        )

        # 팀 0 = 더 많은 선수 (주로 홈팀)
        team_map = {}
        for i, label in enumerate(labels.flatten()):
            team_map[i] = int(label)

        return team_map


# ──────────────────────────────────────────────
# 피치 좌표 변환 (이미지 픽셀 → 실제 피치 좌표)
# ──────────────────────────────────────────────

class PitchTransformer:
    """
    이미지 좌표 → 실제 피치 좌표 (0~105, 0~68)
    호모그래피 기반 변환
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.fw = frame_width
        self.fh = frame_height
        # 기본 선형 변환 (실제 구현에서는 코너 포인트 탐지로 캘리브레이션)
        self.homography = None
        self._setup_default_homography()

    def _setup_default_homography(self):
        """기본 투시 변환 설정"""
        # 이미지의 피치 영역 → 실제 105x68 좌표
        src_points = np.float32([
            [self.fw * 0.05, self.fh * 0.1],   # 좌상
            [self.fw * 0.95, self.fh * 0.1],   # 우상
            [self.fw * 0.95, self.fh * 0.9],   # 우하
            [self.fw * 0.05, self.fh * 0.9],   # 좌하
        ])
        dst_points = np.float32([
            [0, 0], [105, 0], [105, 68], [0, 68]
        ])
        self.homography, _ = cv2.findHomography(src_points, dst_points)

    def image_to_pitch(self, px: float, py: float) -> Tuple[float, float]:
        """픽셀 좌표 → 피치 좌표"""
        if self.homography is None:
            return px / self.fw * 105, py / self.fh * 68
        point = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography)
        x = float(np.clip(transformed[0][0][0], 0, 105))
        y = float(np.clip(transformed[0][0][1], 0, 68))
        return x, y

    def pitch_to_image(self, pitch_x: float, pitch_y: float) -> Tuple[int, int]:
        """피치 좌표 → 픽셀 좌표"""
        px = int(pitch_x / 105 * self.fw)
        py = int(pitch_y / 68 * self.fh)
        return px, py


# ──────────────────────────────────────────────
# 액션 탐지기 (볼 움직임 + 선수 상호작용 분석)
# ──────────────────────────────────────────────

class ActionDetector:
    def __init__(self):
        self.prev_ball_pos: Optional[Tuple[float, float]] = None
        self.ball_carrier_history: List[int] = []
        self.frame_buffer: List[FrameState] = []
        self.pass_candidate: Optional[Dict] = None

    def detect(self, curr_frame: FrameState,
               prev_frame: Optional[FrameState]) -> List[Dict]:
        """프레임 간 변화로 액션 탐지"""
        if prev_frame is None:
            return []

        actions = []
        ball_carrier = next((p for p in curr_frame.players if p.has_ball), None)
        prev_carrier = next((p for p in prev_frame.players if p.has_ball), None)

        if ball_carrier is None:
            return actions

        # 볼 이동 벡터
        ball_dx = curr_frame.ball_x - prev_frame.ball_x
        ball_dy = curr_frame.ball_y - prev_frame.ball_y
        ball_speed = math.sqrt(ball_dx**2 + ball_dy**2)

        # ── 패스 탐지 ──
        if (prev_carrier and ball_carrier and
                prev_carrier.player_id != ball_carrier.player_id and
                prev_carrier.team == ball_carrier.team):
            # 볼 소유권이 같은 팀 내 다른 선수로 이동 → 패스
            actions.append({
                'type': 'pass',
                'player_id': prev_carrier.player_id,
                'target_id': ball_carrier.player_id,
                'start_x': prev_carrier.x,
                'start_y': prev_carrier.y,
                'end_x': ball_carrier.x,
                'end_y': ball_carrier.y,
                'success': True,
            })

        # ── 드리블 탐지 ──
        elif (ball_carrier and prev_carrier and
              ball_carrier.player_id == prev_carrier.player_id):
            dist_moved = math.sqrt(
                (ball_carrier.x - prev_carrier.x)**2 +
                (ball_carrier.y - prev_carrier.y)**2
            )
            # 수비수 근처에서 빠르게 이동 → 드리블
            nearby_opponents = sum(
                1 for p in curr_frame.players
                if p.team != ball_carrier.team and
                math.sqrt((p.x - ball_carrier.x)**2 + (p.y - ball_carrier.y)**2) < 4
            )
            if dist_moved > 1.0 and nearby_opponents > 0:
                actions.append({
                    'type': 'dribble',
                    'player_id': ball_carrier.player_id,
                    'start_x': prev_carrier.x,
                    'start_y': prev_carrier.y,
                    'end_x': ball_carrier.x,
                    'end_y': ball_carrier.y,
                    'success': True,
                })

        # ── 슛 탐지 ──
        if ball_carrier and ball_speed > 8.0:
            # 볼이 빠르게 골문 방향으로 → 슛
            if ball_carrier.x > 70 and ball_dx > 0:
                actions.append({
                    'type': 'shot',
                    'player_id': ball_carrier.player_id,
                    'start_x': ball_carrier.x,
                    'start_y': ball_carrier.y,
                    'end_x': curr_frame.ball_x,
                    'end_y': curr_frame.ball_y,
                    'success': True,
                })

        # ── 태클/점유 탈환 탐지 ──
        if (prev_carrier and ball_carrier and
                prev_carrier.team != ball_carrier.team):
            actions.append({
                'type': 'tackle',
                'player_id': ball_carrier.player_id,
                'start_x': prev_carrier.x,
                'start_y': prev_carrier.y,
                'end_x': ball_carrier.x,
                'end_y': ball_carrier.y,
                'success': True,
            })

        return actions


# ──────────────────────────────────────────────
# 볼 탐지기
# ──────────────────────────────────────────────

class BallDetector:
    def __init__(self):
        self.prev_ball = None
        self.kalman = self._init_kalman()

    def _init_kalman(self):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.03
        return kf

    def detect(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Hough Circle로 볼 탐지"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1,
            minDist=30, param1=50, param2=25,
            minRadius=3, maxRadius=20
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # 가장 밝은 원 선택 (볼은 하얀색)
            best = None
            best_brightness = 0
            for c in circles[0]:
                x, y, r = int(c[0]), int(c[1]), int(c[2])
                roi = gray[max(0,y-r):y+r, max(0,x-r):x+r]
                if roi.size > 0:
                    brightness = np.mean(roi)
                    if brightness > best_brightness:
                        best_brightness = brightness
                        best = (float(x), float(y))
            return best
        return self.prev_ball


# ──────────────────────────────────────────────
# 메인 비디오 처리 파이프라인
# ──────────────────────────────────────────────



# ──────────────────────────────────────────────
# 스코어 탐지기 (OCR 없이 픽셀 변화로 골 감지)
# ──────────────────────────────────────────────

class ScoreDetector:
    """
    골 탐지: 골망 근처 볼 진입 + 갑작스런 카메라 전환/밝기 변화 감지
    스코어: 영상 상단 스코어보드 영역 색상 변화 추적
    """
    def __init__(self):
        self.score = {'home': 0, 'away': 0}
        self.prev_frame = None
        self.goal_cooldown = 0   # 골 감지 후 쿨다운 (중복 방지)
        self.COOLDOWN_FRAMES = 60

    def detect_goal(self, frame: np.ndarray,
                    ball_x: float, ball_y: float,
                    fw: int, fh: int) -> str | None:
        """
        골 감지 방법:
        1. 볼이 골문 근처(x>95m 또는 x<10m)에 진입
        2. 이전 프레임과 밝기 변화가 큼 (리플레이/골 세리머니)
        """
        if self.goal_cooldown > 0:
            self.goal_cooldown -= 1
            return None

        goal = None

        # 볼 위치로 골 판단
        if ball_x > 98 and 28 < ball_y < 40:
            goal = 'home'
        elif ball_x < 7 and 28 < ball_y < 40:
            goal = 'away'

        # 프레임 밝기 급변 (골 세리머니/리플레이)
        if self.prev_frame is not None:
            gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            diff = np.mean(np.abs(gray_curr.astype(float) - gray_prev.astype(float)))
            if diff > 40:  # 급격한 화면 변화
                # 골 상황이었으면 골로 판정
                if goal:
                    self.score[goal] += 1
                    self.goal_cooldown = self.COOLDOWN_FRAMES
                    print(f"[골 감지] {goal} +1 → {self.score}")
                    self.prev_frame = frame.copy()
                    return goal

        if goal:
            self.score[goal] += 1
            self.goal_cooldown = self.COOLDOWN_FRAMES
            print(f"[골 감지] {goal} +1 → {self.score}")

        self.prev_frame = frame.copy()
        return goal


class VideoProcessor:
    def __init__(self, use_yolo: bool = True):
        self.team_classifier = TeamColorClassifier()
        self.action_detector = ActionDetector()
        self.ball_detector = BallDetector()
        self.vaep_analyzer = VAEPAnalyzer()
        self.use_yolo = use_yolo
        self.yolo_model = None

        if use_yolo:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO('yolov8n.pt')
                print("[INFO] YOLO 모델 로드 완료")
            except Exception as e:
                print(f"[WARN] YOLO 로드 실패: {e}, Fallback 모드 사용")
                self.use_yolo = False

        self.transformer = None
        self.player_tracker = PlayerTracker(sample_rate=2)
        self.score_detector = ScoreDetector()
        self.score = {'home': 0, 'away': 0}
        self.prev_frame_state: Optional[FrameState] = None
        self.all_events: List[ActionEvent] = []
        self.frame_results: List[Dict] = []

    def process_video(self, video_path: str,
                      progress_callback=None,
                      sample_rate: int = 2) -> Dict:
        """
        비디오 처리 메인 함수
        sample_rate: 몇 프레임마다 처리할지 (기본 3프레임 = ~10fps)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.transformer = PitchTransformer(fw, fh)

        print(f"[INFO] 비디오: {total_frames}프레임, {fps}fps, {fw}x{fh}")

        frame_idx = 0
        processed = 0
        player_registry: Dict[int, str] = {}  # id -> name (임시)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % sample_rate != 0:
                continue

            timestamp = frame_idx / fps

            # 선수 탐지
            detections = self._detect_players(frame)
            ball_pos = self.ball_detector.detect(frame)
            if ball_pos is None:
                ball_pos = (fw / 2, fh / 2)

            # 팀 분류 (캘리브레이션 중이면 calibrate 호출)
            if not self.team_classifier.calibrated:
                self.team_classifier.calibrate(frame, detections)
            team_map = self.team_classifier.classify(frame, detections)

            # FrameState 생성
            players = []
            ball_px, ball_py = ball_pos
            ball_pitch_x, ball_pitch_y = self.transformer.image_to_pitch(ball_px, ball_py)

            # 트래커용 탐지 리스트 생성
            tracker_dets = []
            det_meta = []
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf = det[:5]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                pitch_x, pitch_y = self.transformer.image_to_pitch(cx, cy)
                team = team_map.get(i, 0)
                tracker_dets.append({'x': cx, 'y': cy, 'team': team, 'conf': float(conf)})
                det_meta.append({'cx': cx, 'cy': cy, 'pitch_x': pitch_x, 'pitch_y': pitch_y, 'team': team})

            # 헝가리안 Re-ID 트래킹
            tracked = self.player_tracker.update(tracker_dets)

            for i, (td, meta) in enumerate(zip(tracked, det_meta)):
                tid = td['track_id']
                cx, cy = meta['cx'], meta['cy']
                pitch_x, pitch_y = meta['pitch_x'], meta['pitch_y']
                team = meta['team']

                dist_to_ball = math.sqrt((cx - ball_px)**2 + (cy - ball_py)**2)
                has_ball = dist_to_ball < 25 and pitch_x > 10

                prev_pos = self.vaep_analyzer.prev_positions.get(tid)
                vx, vy = 0.0, 0.0
                if prev_pos:
                    dt = sample_rate / fps
                    vx = (pitch_x - prev_pos[0]) / max(dt, 0.001)
                    vy = (pitch_y - prev_pos[1]) / max(dt, 0.001)

                # 트래커에서 고정 부여된 번호 사용 (없으면 건너뜀)
                display_num = self.player_tracker.display_number.get(tid, None)
                if display_num is None:
                    continue
                players.append(PlayerState(
                    player_id=tid,
                    team=team,
                    x=pitch_x,
                    y=pitch_y,
                    vx=vx,
                    vy=vy,
                    has_ball=has_ball,
                    number=display_num,
                    name=f"선수{display_num}"
                ))

            frame_state = FrameState(
                frame_idx=frame_idx,
                timestamp=timestamp,
                players=players,
                ball_x=ball_pitch_x,
                ball_y=ball_pitch_y,
            )

            # 액션 탐지
            detected_actions = self.action_detector.detect(
                frame_state, self.prev_frame_state
            )

            # 골 감지
            goal = self.score_detector.detect_goal(
                frame, ball_pitch_x, ball_pitch_y, fw, fh)
            if goal:
                self.score[goal] = self.score.get(goal, 0) + 1

            # VAEP 계산
            events = self.vaep_analyzer.process_frame(frame_state, detected_actions)
            self.all_events.extend(events)

            # 결과 저장 (시각화용)
            top_players = self.vaep_analyzer.get_top_players(5)
            self.frame_results.append({
                'frame_idx': frame_idx,
                'timestamp': round(timestamp, 2),
                'players': [
                    {
                        'id': p.player_id,
                        'team': p.team,
                        'x': round(p.x, 1),
                        'y': round(p.y, 1),
                        'has_ball': p.has_ball,
                        'name': p.name,
                        'number': p.number,
                        'cumulative_vaep': round(
                            self.vaep_analyzer.cumulative_vaep.get(p.player_id, 0.0), 3
                        )
                    }
                    for p in players
                ],
                'ball': {'x': round(ball_pitch_x, 1), 'y': round(ball_pitch_y, 1)},
                'events': [
                    {
                        'player_name': e.player_name,
                        'player_id': e.player_id,
                        'action': e.action_type,
                        'vaep_total': e.vaep_total,
                        'vaep_on': e.vaep_on_ball,
                        'vaep_off': e.vaep_off_ball,
                        'description': e.description,
                        'from': [round(e.start_x,1), round(e.start_y,1)],
                        'to': [round(e.end_x,1), round(e.end_y,1)],
                    }
                    for e in events
                ],
                'top5': [
                    {
                        'player_id': pid,
                        'name': f"선수{pid+1}",
                        'vaep': v
                    }
                    for pid, v in top_players
                ],
            })

            self.prev_frame_state = frame_state
            processed += 1

            if progress_callback:
                progress_callback(frame_idx, total_frames)

            # 진행률 출력 (100프레임마다)
            if processed % 100 == 0:
                print(f"[INFO] 처리 중: {processed}프레임 / {total_frames}프레임")

        cap.release()
        print(f"[INFO] 처리 완료: {processed}프레임, {len(self.all_events)}개 이벤트")

        return {
            'total_frames': total_frames,
            'processed_frames': processed,
            'fps': fps,
            'duration': total_frames / fps,
            'total_events': len(self.all_events),
            'frame_data': self.frame_results,
            'final_vaep': dict(self.vaep_analyzer.cumulative_vaep),
            'events': [
                {
                    'timestamp': e.timestamp,
                    'player': e.player_name,
                    'action': e.action_type,
                    'vaep': e.vaep_total,
                    'vaep_on': e.vaep_on_ball,
                    'vaep_off': e.vaep_off_ball,
                    'description': e.description,
                }
                for e in self.all_events
            ]
        }

    def _detect_players(self, frame: np.ndarray) -> List[Tuple]:
        """선수 탐지 (YOLO 또는 배경 차분)"""
        if self.use_yolo and self.yolo_model is not None:
            return self._yolo_detect(frame)
        else:
            return self._color_detect(frame)

    def _yolo_detect(self, frame: np.ndarray) -> List[Tuple]:
        results = self.yolo_model(frame, verbose=False, classes=[0],
                                  conf=0.45, iou=0.35)  # NMS iou 강화
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                w = x2 - x1
                h = y2 - y1
                # 비율 필터: 사람은 키가 너비보다 커야 함 + 너무 작은 박스 제외
                if h < 20 or w < 8:
                    continue
                aspect = h / max(w, 1)
                if aspect < 1.0 or aspect > 5.0:
                    continue
                detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))

        # 추가 NMS (중복 탐지 제거)
        if len(detections) > 1:
            import cv2
            boxes  = [[d[0], d[1], d[2]-d[0], d[3]-d[1]] for d in detections]
            scores = [d[4] for d in detections]
            idx = cv2.dnn.NMSBoxes(boxes, scores, 0.45, 0.35)
            if len(idx) > 0:
                idx = idx.flatten() if hasattr(idx, 'flatten') else list(idx)
                detections = [detections[i] for i in idx]

        return detections[:25]  # 최대 25명

    def _color_detect(self, frame: np.ndarray) -> List[Tuple]:
        """색상 기반 선수 탐지 (YOLO 없을 때 fallback)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 잔디 색 마스크
        grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        # 사람 영역 = 잔디 아닌 부분
        not_grass = cv2.bitwise_not(grass)

        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(not_grass, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 8000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = h / max(w, 1)
                if 1.2 < aspect < 4.0:  # 사람 비율
                    detections.append((x, y, x+w, y+h, 0.8))

        return detections[:22]  # 최대 22명