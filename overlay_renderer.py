"""
VAEP 오버레이 렌더러
원본 영상 위에 직접 선수 레이블, VAEP 수치, TOP5 패널, 이벤트 배너를 그려서 새 mp4로 저장
한글 폰트: Pillow + NanumGothic
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
import math
import os

# ── 폰트 경로 (Windows / Linux 공통) ──
FONT_PATHS = [
    # Linux
    '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
    '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
    # Windows
    'C:/Windows/Fonts/malgun.ttf',
    'C:/Windows/Fonts/malgunbd.ttf',
    'C:/Windows/Fonts/NanumGothic.ttf',
    'C:/Windows/Fonts/NanumGothicBold.ttf',
]

def _load_font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except:
                continue
    return ImageFont.load_default()

# 폰트 캐시
_font_cache = {}
def get_font(size):
    if size not in _font_cache:
        _font_cache[size] = _load_font(size)
    return _font_cache[size]

# ── 색상 (RGB - Pillow 기준) ──
C_HOME     = (220,  60,  60)   # 빨강
C_AWAY     = ( 60, 120, 210)   # 파랑
C_POS      = ( 80, 200,  80)   # 초록
C_NEG      = (220,  60,  60)   # 빨강
C_WHITE    = (255, 255, 255)
C_BLACK    = (  0,   0,   0)
C_GOLD     = (255, 200,   0)
C_PANEL    = ( 15,  15,  15)
C_ARROW_CV = (  0,  80, 220)   # OpenCV BGR 화살표용


def cv2pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def pil2cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_rrect(draw, x1, y1, x2, y2, fill, radius=10):
    """필로우로 둥근 사각형"""
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill)


def draw_label_box(img_pil, x, y, number, name, vaep, show_names=False):
    """선수 VAEP 레이블 박스"""
    draw = ImageDraw.Draw(img_pil, 'RGBA')
    fn  = get_font(22)
    fv  = get_font(21)

    vaep_str  = f"VAEP {'+' if vaep >= 0 else ''}{vaep:.2f} ▲"
    name_str  = f"{number}  {name}" if show_names and name else str(number)

    nw = draw.textlength(name_str, font=fn)
    vw = draw.textlength(vaep_str, font=fv)
    bw = int(max(nw, vw)) + 20
    bh = 52

    bx = x - bw // 2
    by = y - bh - 16

    # 배경
    draw_rrect(draw, bx, by, bx + bw, by + bh, (15, 15, 15, 210))
    # 테두리
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=8,
                           outline=(80, 80, 80, 180), width=1)
    # 이름/번호
    draw.text((bx + 10, by + 6), name_str, font=fn, fill=C_WHITE)
    # VAEP
    vc = C_POS if vaep >= 0 else C_NEG
    draw.text((bx + 10, by + 28), vaep_str, font=fv, fill=vc)
    # 연결선
    draw.line([(x, by + bh), (x, y - 8)], fill=(120, 120, 120, 150), width=1)


def draw_player_circle_cv(frame, x, y, team, has_ball, number):
    """OpenCV로 선수 원 그리기"""
    color = (60, 60, 220) if team == 0 else (200, 100, 30)
    r = 13 if has_ball else 9
    cv2.circle(frame, (x, y), r, color, -1)
    cv2.circle(frame, (x, y), r, (255, 255, 255), 1)
    if has_ball:
        cv2.circle(frame, (x, y), r + 4, (255, 255, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    num_str = str(number)
    (tw, th), _ = cv2.getTextSize(num_str, font, 0.32, 1)
    cv2.putText(frame, num_str, (x - tw//2, y + th//2),
                font, 0.32, (255,255,255), 1, cv2.LINE_AA)


def draw_vaep_top5(img_pil, top5, fw, fh):
    """좌상단 VAEP TOP5 패널"""
    draw = ImageDraw.Draw(img_pil, 'RGBA')
    ft = get_font(24)
    fn = get_font(21)
    fv = get_font(21)

    px, py = 18, 70
    pw, ph = 290, 230

    # 배경
    draw_rrect(draw, px, py, px+pw, py+ph, (15, 15, 15, 210))
    draw.rounded_rectangle([px, py, px+pw, py+ph], radius=10,
                           outline=(70, 70, 70, 180), width=1)

    # 타이틀
    draw.text((px+12, py+8), "VAEP  TOP 5", font=ft, fill=C_GOLD)
    draw.line([(px+10, py+38), (px+pw-10, py+38)], fill=(60,60,60,200), width=1)

    for i, p in enumerate(top5[:5]):
        ry = py + 48 + i * 34
        vaep = p.get('vaep', 0)

        # 순위
        draw.text((px+10, ry), str(i+1), font=fn, fill=(150,150,150))
        # 점
        draw.ellipse([px+32, ry+4, px+44, ry+16], fill=C_HOME)
        # 이름
        draw.text((px+52, ry), p.get('name',''), font=fn, fill=C_WHITE)
        # VAEP
        vs = f"+{vaep:.2f}"
        vw = draw.textlength(vs, font=fv)
        draw.text((px+pw-vw-12, ry), vs, font=fv, fill=C_POS)


def draw_event_banner(img_pil, event, fw, fh):
    """하단 이벤트 배너"""
    if not event:
        return
    draw = ImageDraw.Draw(img_pil, 'RGBA')
    fb = get_font(26)
    ft = get_font(22)
    fv = get_font(52)
    fp = get_font(20)

    bh = 100
    by = fh - bh - 45

    # 전체 배경
    draw_rrect(draw, 10, by, fw-10, by+bh, (10, 10, 10, 225))

    # 좌측 빨간 박스
    draw_rrect(draw, 14, by+4, 165, by+bh-4, (170, 30, 30, 240))
    draw.text((22, by+10), "VAEP", font=fb, fill=C_WHITE)
    draw.text((22, by+38), "EVENT", font=fb, fill=C_WHITE)
    mins = int(event.get('timestamp', 0) // 60)
    secs = int(event.get('timestamp', 0) % 60)
    draw.text((22, by+68), f"{mins:02d}:{secs:02d}", font=ft, fill=(200,200,200))

    # 액션 설명
    desc = event.get('description', '')
    draw.text((178, by+28), desc, font=fb, fill=C_WHITE)

    # VAEP 수치
    vaep = event.get('vaep_total', 0)
    vaep_str = f"{'+' if vaep>=0 else ''}{vaep:.2f}"
    vc = C_POS if vaep >= 0 else C_NEG
    vw = draw.textlength(vaep_str, font=fv)
    draw.text((fw - vw - 180, by+14), vaep_str, font=fv, fill=vc)

    # 득점 확률
    prob_str = f"득점 확률  +{abs(vaep)*100:.2f}%"
    draw.text((fw - 280, by+bh-28), prob_str, font=fp, fill=C_POS)



def draw_score_overlay(img_pil, score, fw):
    """상단 스코어보드"""
    draw = ImageDraw.Draw(img_pil, 'RGBA')
    ft = get_font(22)
    fs = get_font(28)

    home  = score.get('home_team', '홈')
    away  = score.get('away_team', '원정')
    sh    = score.get('home', 0)
    sa    = score.get('away', 0)
    cx    = fw // 2

    # 배경
    draw_rrect(draw, cx-130, 6, cx+130, 50, (15,15,15,210))

    # 홈팀
    draw_rrect(draw, cx-128, 8, cx-52, 48, (170,30,30,230))
    draw.text((cx-122, 16), home, font=ft, fill=C_WHITE)

    # 스코어
    score_str = f"{sh}  -  {sa}"
    sw = draw.textlength(score_str, font=fs)
    draw.text((cx - sw//2, 12), score_str, font=fs, fill=C_WHITE)

    # 원정팀
    draw_rrect(draw, cx+52, 8, cx+128, 48, (30,80,170,230))
    aw = draw.textlength(away, font=ft)
    draw.text((cx+128-aw-6, 16), away, font=ft, fill=C_WHITE)

def draw_dashed_arrow(frame, pt1, pt2, color=(0,80,220), thickness=2, dash=12):
    """OpenCV 점선 화살표"""
    x1,y1 = pt1; x2,y2 = pt2
    dist = math.sqrt((x2-x1)**2+(y2-y1)**2)
    if dist < 1: return
    dx=(x2-x1)/dist; dy=(y2-y1)/dist
    d=0; on=True
    while d < dist-dash:
        if on:
            sx=int(x1+dx*d); sy=int(y1+dy*d)
            ex=int(x1+dx*min(d+dash,dist)); ey=int(y1+dy*min(d+dash,dist))
            cv2.line(frame,(sx,sy),(ex,ey),color,thickness)
        d+=dash; on=not on
    ang=math.atan2(y2-y1,x2-x1)
    for a in [ang+2.5,ang-2.5]:
        ex_=int(x2-14*math.cos(a)); ey_=int(y2-14*math.sin(a))
        cv2.line(frame,(x2,y2),(ex_,ey_),color,thickness)


# ──────────────────────────────────────────────
# 메인 오버레이 파이프라인
# ──────────────────────────────────────────────

def render_vaep_overlay(video_path: str,
                         result_data: dict,
                         output_path: str,
                         progress_callback=None,
                         show_names: bool = False,
                         player_names: dict = None) -> str:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))

    frame_data  = result_data.get('frame_data', [])
    frame_map   = {f['frame_idx']: f for f in frame_data}
    sorted_keys = sorted(frame_map.keys())

    # 피치 → 이미지 변환 (호모그래피)
    src = np.float32([[fw*.05,fh*.10],[fw*.95,fh*.10],[fw*.95,fh*.90],[fw*.05,fh*.90]])
    dst = np.float32([[0,0],[105,0],[105,68],[0,68]])
    H_inv, _ = cv2.findHomography(dst, src)

    def p2img(px, py):
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        r  = cv2.perspectiveTransform(pt, H_inv)
        return int(r[0][0][0]), int(r[0][0][1])

    frame_idx     = 0
    current_event = None
    event_timer   = 0
    EVENT_FRAMES  = int(fps * 4)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 가장 가까운 분석 프레임
        fd = None
        if sorted_keys:
            closest = min(sorted_keys, key=lambda x: abs(x - frame_idx))
            if abs(closest - frame_idx) < 6:
                fd = frame_map[closest]

        if fd is not None:
            players = fd.get('players', [])
            top5    = fd.get('top5', [])
            events  = fd.get('events', [])

            if events:
                current_event = events[-1]
                event_timer   = EVENT_FRAMES

            # ── 1. 선수 원 (OpenCV) ──
            for p in players:
                px, py = p2img(p['x'], p['y'])
                if not (0 < px < fw and 0 < py < fh):
                    continue
                draw_player_circle_cv(frame, px, py,
                    p.get('team', 0), p.get('has_ball', False),
                    p.get('number', p.get('id',0)+1))

            # ── 2. 화살표 (OpenCV) ──
            if current_event and event_timer > 0:
                frm = current_event.get('from')
                to  = current_event.get('to')
                if frm and to:
                    ax1,ay1 = p2img(frm[0], frm[1])
                    ax2,ay2 = p2img(to[0],  to[1])
                    prog = 1.0 - event_timer / EVENT_FRAMES
                    ex = int(ax1+(ax2-ax1)*min(prog*2,1.0))
                    ey = int(ay1+(ay2-ay1)*min(prog*2,1.0))
                    draw_dashed_arrow(frame,(ax1,ay1),(ex,ey))
                    if prog > 0.5:
                        cv2.circle(frame,(ax2,ay2),20,C_ARROW_CV,2)

            # ── 3. 한글 오버레이 (Pillow) ──
            img_pil = cv2pil(frame)

            # 스코어보드
            score = fd.get('score', {})
            if score:
                draw_score_overlay(img_pil, score, fw)

            # 레이블: VAEP 상위 5명 + 볼 소유자만
            sorted_p = sorted(players, key=lambda p: abs(p.get('cumulative_vaep',0)), reverse=True)
            label_count = 0
            for p in sorted_p:
                px, py = p2img(p['x'], p['y'])
                if not (0 < px < fw and 0 < py < fh):
                    continue
                vaep = p.get('cumulative_vaep', 0.0)
                is_ball = p.get('has_ball', False)
                # 볼 소유자는 무조건, 나머지는 상위 5명만
                if not is_ball and label_count >= 5:
                    continue
                if abs(vaep) > 0.03 or is_ball:
                    name = p.get('name','') if show_names else ''
                    label_y = max(py, 80)
                    draw_label_box(img_pil, px, label_y,
                                   p.get('number', p.get('id',0)+1),
                                   name, vaep, show_names)
                    if not is_ball:
                        label_count += 1

            # TOP5
            if top5:
                draw_vaep_top5(img_pil, top5, fw, fh)

            # 이벤트 배너
            if event_timer > 0:
                draw_event_banner(img_pil, current_event, fw, fh)
                event_timer -= 1

            frame = pil2cv(img_pil)

        out.write(frame)
        if progress_callback:
            progress_callback(frame_idx, total)

    cap.release()
    out.release()
    print(f"[INFO] 오버레이 영상 저장 완료: {output_path}")
    return output_path