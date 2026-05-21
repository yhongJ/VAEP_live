"""
선수 추적기 v4
- 팀 분류: HSV Hue 기반 클러스터링 (K-means 보다 안정적)
- ID 추적: 헝가리안 매칭, 새 ID 생성 최소화
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple
import math
import cv2


class TeamColorClassifier:
    """
    HSV Hue 기반 팀 분류
    - 첫 CALIB_FRAMES 프레임에서 두 팀 대표 색상 학습
    - 이후 고정된 색상으로 분류 (매 프레임 재계산 X)
    """
    CALIB_FRAMES = 8

    def __init__(self):
        self.calibrated  = False
        self.center0     = None  # 팀0 HSV 중심
        self.center1     = None  # 팀1 HSV 중심
        self.frame_count = 0
        self.all_colors  = []    # 캘리브레이션용 색상 샘플

    # ── 유니폼 색상 추출 ──
    def get_jersey_hsv(self, frame: np.ndarray, bbox: Tuple) -> np.ndarray | None:
        x1,y1,x2,y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        bh = y2 - y1
        if bh < 20:
            return None
        # 상체 중앙만 크롭
        ty1 = y1 + int(bh * 0.15)
        ty2 = y1 + int(bh * 0.50)
        tx1 = x1 + int((x2-x1) * 0.25)
        tx2 = x2 - int((x2-x1) * 0.25)
        if ty2 <= ty1 or tx2 <= tx1:
            return None
        roi = frame[ty1:ty2, tx1:tx2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 잔디색(H:35~85) 제거, 너무 어두운 픽셀 제거
        mask  = cv2.inRange(hsv, (35,40,40), (85,255,255))
        mask2 = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,50]))
        combined = cv2.bitwise_or(mask, mask2)
        valid = hsv[combined == 0]
        if len(valid) < 8:
            return None
        # 중앙값 사용 (이상치에 강함)
        return np.median(valid, axis=0)

    # ── 캘리브레이션 ──
    def calibrate(self, frame: np.ndarray, detections: List[Tuple]) -> bool:
        if self.calibrated:
            return True
        self.frame_count += 1
        for det in detections:
            c = self.get_jersey_hsv(frame, det[:4])
            if c is not None:
                self.all_colors.append(c)

        if self.frame_count < self.CALIB_FRAMES:
            return False
        if len(self.all_colors) < 6:
            return False

        arr = np.array(self.all_colors, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        try:
            _, labels, centers = cv2.kmeans(
                arr, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )
        except:
            return False

        self.center0 = centers[0]
        self.center1 = centers[1]
        self.calibrated = True

        h0, h1 = centers[0][0], centers[1][0]
        s0, s1 = centers[0][1], centers[1][1]
        print(f"[TeamClassifier] 팀 색상 확정: "
              f"팀0 H={h0:.0f} S={s0:.0f} | 팀1 H={h1:.0f} S={s1:.0f}")
        return True

    # ── 분류 ──
    def classify(self, frame: np.ndarray, detections: List[Tuple]) -> Dict[int, int]:
        result = {}

        if not self.calibrated:
            # 캘리브레이션 전: 임시 분류
            colors = []
            for det in detections:
                c = self.get_jersey_hsv(frame, det[:4])
                colors.append(c if c is not None else np.array([0, 0, 128]))
            if len(colors) < 2:
                return {i: 0 for i in range(len(detections))}
            arr = np.array(colors, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
            try:
                _, labels, _ = cv2.kmeans(arr, 2, None, criteria, 3,
                                           cv2.KMEANS_RANDOM_CENTERS)
                return {i: int(l) for i, l in enumerate(labels.flatten())}
            except:
                return {i: 0 for i in range(len(detections))}

        # 캘리브레이션 완료: 고정 색상으로 분류
        for i, det in enumerate(detections):
            c = self.get_jersey_hsv(frame, det[:4])
            if c is None:
                result[i] = 0
                continue
            # HSV 전체 거리
            d0 = np.linalg.norm(c - self.center0)
            d1 = np.linalg.norm(c - self.center1)
            result[i] = 0 if d0 <= d1 else 1

        return result


class PlayerTracker:
    """
    헝가리안 매칭 기반 ID 추적
    - MAX_TRACKS 고정 (새 ID 최소화)
    - 매칭 실패 시 가장 가까운 같은 팀 트랙에 강제 배정
    """

    def __init__(self, sample_rate: int = 5):
        self.tracks: Dict[int, dict] = {}
        self.next_id    = 0
        self.MAX_DIST   = 120 + (sample_rate - 3) * 30
        self.MAX_TRACKS = 24
        # 전체 번호 카운터 - track_id 생성 시 1번부터 고정 부여
        self._num_counter = 1
        # track_id → 고정 번호 매핑 (절대 변경 안 됨)
        self.display_number: Dict[int, int] = {}

    def update(self, detections: List[dict]) -> List[dict]:
        if not detections:
            for t in self.tracks.values():
                t['lost'] += 1
            return []

        # 첫 프레임 초기화
        if not self.tracks:
            for i, d in enumerate(detections[:self.MAX_TRACKS]):
                self.tracks[i] = {
                    'x': d['x'], 'y': d['y'],
                    'team': d['team'], 'lost': 0
                }
                self.display_number[i] = self._num_counter
                self._num_counter += 1
            self.next_id = len(self.tracks)
            return [dict(d, track_id=i)
                    for i, d in enumerate(detections[:self.MAX_TRACKS])]

        trk_ids = list(self.tracks.keys())
        n_d, n_t = len(detections), len(trk_ids)

        # 비용 행렬
        cost = np.full((n_d, n_t), 9999.0)
        for di, d in enumerate(detections):
            for ti, tid in enumerate(trk_ids):
                t = self.tracks[tid]
                dist = math.sqrt((d['x']-t['x'])**2 + (d['y']-t['y'])**2)
                if d['team'] != t['team']:
                    dist += 400   # 다른 팀 패널티
                cost[di, ti] = dist

        row_ind, col_ind = linear_sum_assignment(cost)

        result_map   = {}
        matched_trks = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < self.MAX_DIST:
                tid = trk_ids[c]
                d   = detections[r]
                self.tracks[tid].update(
                    {'x': d['x'], 'y': d['y'], 'team': d['team'], 'lost': 0}
                )
                result_map[r]   = tid
                matched_trks.add(c)

        # 매칭 안 된 탐지 처리
        for di in range(n_d):
            if di in result_map:
                continue
            d = detections[di]

            # 같은 팀 중 가장 가까운 트랙에 강제 배정
            same_team = [(tid, t) for tid, t in self.tracks.items()
                         if t['team'] == d['team']]
            if same_team:
                best = min(same_team,
                    key=lambda x: math.sqrt((d['x']-x[1]['x'])**2 +
                                            (d['y']-x[1]['y'])**2))
                best_tid, best_dist = best[0], math.sqrt(
                    (d['x']-best[1]['x'])**2 + (d['y']-best[1]['y'])**2)

                if best_dist < self.MAX_DIST * 2.5:
                    self.tracks[best_tid].update(
                        {'x': d['x'], 'y': d['y'], 'team': d['team'], 'lost': 0}
                    )
                    result_map[di] = best_tid
                    continue

            # 그래도 없으면 새 트랙 (MAX_TRACKS 미만)
            if len(self.tracks) < self.MAX_TRACKS:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'x': d['x'], 'y': d['y'],
                    'team': d['team'], 'lost': 0
                }
                # 번호 고정 부여
                self.display_number[tid] = self._num_counter
                self._num_counter += 1
                result_map[di] = tid

        # lost 증가
        for ci, tid in enumerate(trk_ids):
            if ci not in matched_trks:
                self.tracks[tid]['lost'] += 1

        # 결과 - 같은 프레임 내 중복 track_id 제거
        used_ids = set()
        results = []
        for di, d in enumerate(detections):
            tid = result_map.get(di, -1)
            if tid in used_ids:
                # 중복이면 -1로 표시 (레이블 안 표시)
                tid = -1
            elif tid >= 0:
                used_ids.add(tid)
            out = dict(d)
            out['track_id'] = tid
            results.append(out)
        return results