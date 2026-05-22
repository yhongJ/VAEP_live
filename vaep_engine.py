"""
VAEP Engine: Value Added by Each Player Action (오프더볼 확장 포함)
=================================================================
기존 VAEP(패스/드리블/슛 등 볼 터치 Action) + 오프더볼 움직임 기여도 통합 모델
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math


# ──────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────

@dataclass
class PlayerState:
    player_id: int
    team: int                    # 0: 공격팀, 1: 수비팀
    x: float                    # 0~105m (피치 가로)
    y: float                    # 0~68m (피치 세로)
    vx: float = 0.0             # 속도 x (m/frame)
    vy: float = 0.0             # 속도 y
    has_ball: bool = False
    number: int = 0
    name: str = ""

@dataclass
class FrameState:
    frame_idx: int
    timestamp: float            # 초 단위
    players: List[PlayerState] = field(default_factory=list)
    ball_x: float = 52.5
    ball_y: float = 34.0
    ball_vx: float = 0.0
    ball_vy: float = 0.0

@dataclass
class ActionEvent:
    frame_idx: int
    timestamp: float
    player_id: int
    player_name: str
    team: int
    action_type: str            # pass, dribble, shot, cross, tackle, off_ball_run, pressing
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    success: bool
    vaep_on_ball: float = 0.0
    vaep_off_ball: float = 0.0
    vaep_total: float = 0.0
    description: str = ""


# ──────────────────────────────────────────────
# 위치 가치 모델 (xT/xG 혼합)
# ──────────────────────────────────────────────

class ScoringProbabilityModel:
    """
    위치 기반 득점 기대값 (xT + xG 혼합)
    - 페널티박스 안: 실제 xG 모델
    - 박스 밖: Expected Threat (해당 위치에서 경기 전개 시 득점 기여 확률)
    - 골문 기준: x=105, y=34 (오른쪽 공격)
    """

    def scoring_prob(self, x: float, y: float,
                     n_defenders_between: int = 0,
                     angle_blocked: float = 0.0) -> float:
        dist = math.sqrt((105 - x) ** 2 + (34 - y) ** 2)
        goal_half_width = 3.66
        angle = self._shooting_angle(x, y, goal_half_width)

        # 구역별 로지스틱 파라미터 (xT 근사)
        if x > 83:          # 페널티박스 내
            b0, b1, b2 = 0.38,  -0.055, 2.10
        elif x > 70:        # 파이널 서드 위험 구역
            b0, b1, b2 = -0.80, -0.040, 1.80
        elif x > 52.5:      # 공격 절반
            b0, b1, b2 = -2.50, -0.025, 1.20
        else:               # 수비 절반
            b0, b1, b2 = -4.50, -0.010, 0.60

        logit    = b0 + b1 * dist + b2 * angle
        base_xg  = 1.0 / (1.0 + math.exp(-logit))
        prob     = max(0.0, base_xg - 0.12 * n_defenders_between - 0.25 * angle_blocked)
        return min(prob, 0.92)

    def conceding_prob(self, x: float, y: float) -> float:
        """실점 확률 (자기 골문 = x=0 기준)"""
        dist  = math.sqrt(x ** 2 + (34 - y) ** 2)
        angle = self._shooting_angle_left(x, y, 3.66)

        if x < 22:
            b0, b1, b2 = 0.38,  -0.055, 2.10
        elif x < 35:
            b0, b1, b2 = -0.80, -0.040, 1.80
        elif x < 52.5:
            b0, b1, b2 = -2.50, -0.025, 1.20
        else:
            b0, b1, b2 = -4.50, -0.010, 0.60

        logit = b0 + b1 * dist + b2 * angle
        return min(1.0 / (1.0 + math.exp(-logit)), 0.92)

    def _shooting_angle(self, x, y, gw):
        goal_y_top = 34 + gw
        goal_y_bot = 34 - gw
        dx = max(105 - x, 0.1)
        a1 = math.atan2(goal_y_top - y, dx)
        a2 = math.atan2(goal_y_bot - y, dx)
        return max(0.0, a1 - a2)

    def _shooting_angle_left(self, x, y, gw):
        goal_y_top = 34 + gw
        goal_y_bot = 34 - gw
        dx = max(x, 0.1)
        a1 = math.atan2(goal_y_top - y, dx)
        a2 = math.atan2(goal_y_bot - y, dx)
        return max(0.0, a1 - a2)


# ──────────────────────────────────────────────
# 온볼 VAEP
# ──────────────────────────────────────────────

class OnBallVAEP:
    """
    온볼 VAEP = ΔV_score - ΔV_concede
    = [V_score(after) - V_score(before)] - [V_concede(after) - V_concede(before)]

    액션 유형별로 득점/실점 위협 가중치를 다르게 적용
    """

    def __init__(self):
        self.model = ScoringProbabilityModel()
        self.weights = {
            'pass':    (1.0, 0.8),
            'dribble': (1.2, 0.6),
            'shot':    (1.5, 0.2),
            'cross':   (1.1, 0.7),
            'tackle':  (0.5, 1.3),
        }

    def compute(self, action: str,
                before: Tuple[float, float],
                after: Tuple[float, float],
                success: bool,
                frame=None) -> float:

        bx, by = before
        ax, ay = after
        ws, wc = self.weights.get(action, (1.0, 1.0))

        n_def_b = self._count_defenders(bx, by, frame)
        n_def_a = self._count_defenders(ax, ay, frame)

        vs_b = self.model.scoring_prob(bx, by, n_def_b)
        vc_b = self.model.conceding_prob(bx, by)

        if success:
            vs_a = self.model.scoring_prob(ax, ay, n_def_a)
            vc_a = self.model.conceding_prob(ax, ay)
        else:
            vs_a = vs_b * 0.05
            vc_a = self.model.conceding_prob(ax, ay) * 1.8

        vaep = (vs_a - vs_b) * ws - (vc_a - vc_b) * wc

        # 슛 보너스 (유효 슈팅 범위)
        if action == "shot" and success and bx > 75:
            vaep += 0.08

        return round(float(np.clip(vaep, -0.80, 0.80)), 4)

    def _count_defenders(self, x, y, frame):
        if frame is None:
            return 0
        return min(sum(
            1 for p in frame.players
            if p.team == 1 and x < p.x < 105 and abs(p.y - y) < 8
        ), 4)


# ──────────────────────────────────────────────
# 오프더볼 VAEP (핵심 확장 모델)
# ──────────────────────────────────────────────

class OffBallVAEP:
    """
    오프더볼 기여도 = 공간 창출 + 수비 압박 + 런 가치

    1. Space Creation (공간 창출)
       선수 움직임 → 수비수 끌어당김 → 동료 공간 확보
       ΔV = 해방된 동료 위치 가치 × 수비수 이동 정도

    2. Pressing Value (압박 가치)
       볼 소유자 압박 → 패스 옵션 차단 → 실점 확률 감소
       ΔV = 압박 강도 × 볼 위험 지역 가중치

    3. Run Value (런 가치)
       볼 없이 위협 공간 진입 → 잠재 득점 확률 상승
       ΔV = 이동 후 위치 가치 - 이동 전 위치 가치
    """

    def __init__(self):
        self.model = ScoringProbabilityModel()

    def compute_space_creation(self,
                               player: PlayerState,
                               teammates: List[PlayerState],
                               opponents: List[PlayerState],
                               prev_positions: Dict[int, Tuple[float, float]]) -> float:
        if player.player_id not in prev_positions:
            return 0.0

        prev_x, prev_y = prev_positions[player.player_id]
        curr_x, curr_y = player.x, player.y
        dist_moved = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)

        if dist_moved < 0.5:
            return 0.0

        space_value = 0.0
        for opp in opponents:
            d_prev = math.sqrt((opp.x - prev_x)**2 + (opp.y - prev_y)**2)
            d_curr = math.sqrt((opp.x - curr_x)**2 + (opp.y - curr_y)**2)
            # 수비수가 나를 따라와서 가까워졌다면
            if d_prev > 5 and d_curr < d_prev * 0.7:
                for tm in teammates:
                    if tm.player_id == player.player_id:
                        continue
                    freed = max(0, d_prev - math.sqrt((opp.x - tm.x)**2 + (opp.y - tm.y)**2))
                    space_value += freed * self.model.scoring_prob(tm.x, tm.y) * 0.06

        return float(np.clip(space_value, 0.0, 0.15))

    def compute_pressing_value(self,
                               player: PlayerState,
                               ball_carrier: Optional[PlayerState],
                               frame: FrameState) -> float:
        if ball_carrier is None or player.team == ball_carrier.team:
            return 0.0

        dist = math.sqrt((player.x - ball_carrier.x)**2 + (player.y - ball_carrier.y)**2)
        if dist > 8:
            return 0.0

        press_intensity = (8 - dist) / 8
        carrier_danger  = self.model.scoring_prob(ball_carrier.x, ball_carrier.y)
        n_pressers = sum(
            1 for p in frame.players
            if p.team == player.team and p.player_id != player.player_id
            and math.sqrt((p.x - ball_carrier.x)**2 + (p.y - ball_carrier.y)**2) < 6
        )
        val = press_intensity * carrier_danger * (1 + 0.2 * n_pressers) * 0.35
        return float(np.clip(val, 0.0, 0.15))

    def compute_run_value(self,
                          player: PlayerState,
                          prev_positions: Dict[int, Tuple[float, float]],
                          frame: FrameState) -> float:
        if player.has_ball or player.player_id not in prev_positions:
            return 0.0

        prev_x, prev_y = prev_positions[player.player_id]
        curr_x, curr_y = player.x, player.y

        # 공격 방향 (→) 이동
        if curr_x - prev_x < 0.3:
            return 0.0

        prev_val = self.model.scoring_prob(prev_x, prev_y)
        curr_val = self.model.scoring_prob(curr_x, curr_y)
        value_gained = curr_val - prev_val

        # 수비수 등 뒤 공간 진입 보너스
        depth_bonus = 0.0
        for opp in frame.players:
            if opp.team != player.team and opp.x < curr_x and abs(opp.y - curr_y) < 5:
                depth_bonus += 0.025

        run_val = value_gained * 0.45 + depth_bonus
        return float(np.clip(run_val, 0.0, 0.20))

    def compute_total(self,
                      player: PlayerState,
                      frame: FrameState,
                      prev_positions: Dict[int, Tuple[float, float]]) -> float:
        teammates  = [p for p in frame.players if p.team == player.team and p.player_id != player.player_id]
        opponents  = [p for p in frame.players if p.team != player.team]
        ball_carrier = next((p for p in frame.players if p.has_ball), None)

        s = self.compute_space_creation(player, teammates, opponents, prev_positions)
        p = self.compute_pressing_value(player, ball_carrier, frame)
        r = self.compute_run_value(player, prev_positions, frame)

        return round(float(np.clip(s + p + r, -0.10, 0.25)), 4)


# ──────────────────────────────────────────────
# 통합 VAEP 분석기
# ──────────────────────────────────────────────

class VAEPAnalyzer:

    def __init__(self):
        self.on_ball  = OnBallVAEP()
        self.off_ball = OffBallVAEP()
        self.prev_positions: Dict[int, Tuple[float, float]] = {}
        self.cumulative_vaep: Dict[int, float] = {}
        self.vaep_history: Dict[int, List[float]] = {}   # 최근 N개 이벤트
        self.action_history: List[ActionEvent] = []
        self.HISTORY_LEN = 20   # 최근 20개 액션만 누적

    def process_frame(self, frame: FrameState,
                      detected_actions: List[Dict]) -> List[ActionEvent]:
        events = []

        # 1. 온볼 액션 처리
        for act in detected_actions:
            pid = act['player_id']
            player = next((p for p in frame.players if p.player_id == pid), None)
            if player is None:
                continue

            vaep_ob = self.on_ball.compute(
                action=act['type'],
                before=(act['start_x'], act['start_y']),
                after=(act['end_x'], act['end_y']),
                success=act.get('success', True),
                frame=frame
            )
            vaep_off = self.off_ball.compute_total(player, frame, self.prev_positions)
            vaep_total = round(vaep_ob + vaep_off * 0.25, 4)  # 온볼 주, 오프볼 보조

            ev = ActionEvent(
                frame_idx=frame.frame_idx,
                timestamp=frame.timestamp,
                player_id=pid,
                player_name=player.name or f"#{player.number}",
                team=player.team,
                action_type=act['type'],
                start_x=act['start_x'], start_y=act['start_y'],
                end_x=act['end_x'],   end_y=act['end_y'],
                success=act.get('success', True),
                vaep_on_ball=vaep_ob,
                vaep_off_ball=vaep_off,
                vaep_total=vaep_total,
                description=self._describe(act['type'], player.name, vaep_total)
            )
            events.append(ev)
            self.action_history.append(ev)
            if pid not in self.vaep_history:
                self.vaep_history[pid] = []
            self.vaep_history[pid].append(vaep_total)
            if len(self.vaep_history[pid]) > self.HISTORY_LEN:
                self.vaep_history[pid].pop(0)
            self.cumulative_vaep[pid] = round(sum(self.vaep_history[pid]), 4)

        # 2. 오프더볼 선수들 (볼 관련 액션 없는 선수)
        acted_ids = {a['player_id'] for a in detected_actions}
        for player in frame.players:
            if player.has_ball or player.player_id in acted_ids:
                continue

            vaep_off = self.off_ball.compute_total(player, frame, self.prev_positions)
            if abs(vaep_off) > 0.005:
                pid2 = player.player_id
                if pid2 not in self.vaep_history:
                    self.vaep_history[pid2] = []
                self.vaep_history[pid2].append(vaep_off)
                if len(self.vaep_history[pid2]) > self.HISTORY_LEN:
                    self.vaep_history[pid2].pop(0)
                self.cumulative_vaep[pid2] = round(sum(self.vaep_history[pid2]), 4)
            if vaep_off > 0.03:
                act_type = "off_ball_run" if getattr(player, 'vx', 0) > 0.5 else "pressing"
                ev = ActionEvent(
                    frame_idx=frame.frame_idx,
                    timestamp=frame.timestamp,
                    player_id=player.player_id,
                    player_name=player.name or f"#{player.number}",
                    team=player.team,
                    action_type=act_type,
                    start_x=self.prev_positions.get(player.player_id, (player.x, player.y))[0],
                    start_y=self.prev_positions.get(player.player_id, (player.x, player.y))[1],
                    end_x=player.x, end_y=player.y,
                    success=True,
                    vaep_on_ball=0.0,
                    vaep_off_ball=vaep_off,
                    vaep_total=vaep_off,
                    description=f"{player.name} 오프더볼 → +{vaep_off:.3f}"
                )
                events.append(ev)
                self.action_history.append(ev)

        # 현재 위치 저장
        for p in frame.players:
            self.prev_positions[p.player_id] = (p.x, p.y)

        return events

    def get_top_players(self, n: int = 5) -> List[Tuple[int, float]]:
        return sorted(self.cumulative_vaep.items(), key=lambda x: x[1], reverse=True)[:n]

    def _describe(self, action: str, name: str, vaep: float) -> str:
        sign = "+" if vaep >= 0 else ""
        labels = {
            'pass': '패스', 'dribble': '드리블', 'shot': '슛 시도',
            'cross': '크로스', 'tackle': '태클',
        }
        return f"{name} {labels.get(action, action)} → VAEP {sign}{vaep:.3f}"