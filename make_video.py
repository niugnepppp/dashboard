#!/usr/bin/env python3
"""
피자 매장 실적 시사점 — 30초 키네틱 타이포그래피 광고 (코드만으로 생성)

  영상: Pillow + NumPy로 프레임을 직접 그림 (1920x1080, 30fps)
  음성: macOS `say` 한국어 음성(Yuna)
  음악·효과음: NumPy로 합성
  인코딩: imageio-ffmpeg에 들어 있는 ffmpeg

구성 (후킹 → 시사점 3개 → 클리프행어)
  HOOK   5년 만에, 매출 +56%. 그런데. 숫자가 말하지 않은 세 가지.
  01     주문은 멈췄다. 성장은 가격이 만들었다.
  02     1인당 인건비 +89%. 이익을 조용히 깎고 있다.
  03     자사앱 31%. 하지만 배달앱 비용은 그대로.
  CLIFF  그리고 지금, 두 매장이 흔들리고 있다. 다음은 어디일까?

실행:  python make_video.py --out 피자_인사이트_30초.mp4
"""
import argparse, functools, math, os, subprocess, sys, tempfile, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS, TOTAL, SR = 1920, 1080, 30, 30.0, 48000
VOICE = "Yuna"

# ───────────────────────── 색 ─────────────────────────
BLACK = (0, 0, 0)
WHITE_BG = (245, 245, 247)
WHITE = (245, 245, 247)
DARK = (29, 29, 31)
GRAY = (134, 134, 139)
LGRAY = (200, 200, 205)
DGRAY = (80, 80, 86)
RED = (255, 69, 58)
TRACK = (44, 44, 46)
GR = {
    "warm": [(255, 159, 10), (255, 55, 95), (191, 90, 242)],
    "cool": [(100, 210, 255), (10, 132, 255), (94, 92, 230)],
    "hot":  [(255, 159, 10), (255, 69, 58), (255, 45, 85)],
}

# ───────────────────────── 폰트 ─────────────────────────
PRET = {"xb": "Pretendard-ExtraBold.otf", "b": "Pretendard-Bold.otf",
        "sb": "Pretendard-SemiBold.otf", "m": "Pretendard-Medium.otf"}
SDG = {"xb": 6, "b": 6, "sb": 4, "m": 2}


@functools.lru_cache(None)
def font(w, size):
    p = os.path.expanduser(f"~/Library/Fonts/{PRET[w]}")
    if os.path.exists(p):
        return ImageFont.truetype(p, size)
    return ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", size, index=SDG[w])


@functools.lru_cache(None)
def vcenter(w, size):
    f = font(w, size)
    asc, _ = f.getmetrics()
    top, bot = f.getbbox("가나다", anchor="ls")[1::2]
    return asc + (top + bot) / 2


@functools.lru_cache(4096)
def glyph(s, w, size):
    f = font(w, size)
    asc, desc = f.getmetrics()
    adv = f.getlength(s)
    P = int(size * 0.15)
    img = Image.new("L", (int(math.ceil(adv)) + 2 * P, asc + desc + P), 0)
    ImageDraw.Draw(img).text((P, asc), s, font=f, fill=255, anchor="ls")
    return np.asarray(img, dtype=np.float32) / 255.0, adv, P


@functools.lru_cache(512)
def grad(name, h, w):
    st = np.array(GR[name], np.float32)
    n = len(st)
    x = (np.linspace(0, 1, w)[None, :] + np.linspace(0, 0.3, h)[:, None]) / 1.3
    idx = x * (n - 1)
    i0 = np.clip(np.floor(idx).astype(int), 0, n - 2)
    fr = (idx - i0)[..., None]
    return st[i0] * (1 - fr) + st[i0 + 1] * fr


# ───────────────────────── 이징 ─────────────────────────
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, s, d):
    return clamp((t - s) / d)


def eo(p):  # expo out
    p = clamp(p)
    return 1.0 if p >= 1 else 1 - 2 ** (-10 * p)


def ec(p):  # cubic out
    p = clamp(p)
    return 1 - (1 - p) ** 3


# ───────────────────────── 그리기 ─────────────────────────
def blit(F, m, x0, y0, col, a=1.0, clip=None):
    if a <= 0.004:
        return
    h, w = m.shape
    x0, y0 = int(round(x0)), int(round(y0))
    Y0, Y1 = 0, H
    if clip:
        Y0, Y1 = max(0, int(clip[0])), min(H, int(clip[1]))
    ax0, ay0, ax1, ay1 = max(x0, 0), max(y0, Y0), min(x0 + w, W), min(y0 + h, Y1)
    if ax1 <= ax0 or ay1 <= ay0:
        return
    mm = m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0][..., None] * a
    if isinstance(col, str):
        c = grad(col, h, w)[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
    else:
        c = np.array(col, np.float32)
    reg = F[ay0:ay1, ax0:ax1]
    reg *= 1 - mm
    reg += c * mm


def scaled(m, s):
    if abs(s - 1) < 1e-3:
        return m
    h, w = m.shape
    im = Image.fromarray((m * 255).astype(np.uint8)).resize(
        (max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def text(F, s, w, size, cx, cy, col, a=1.0, scale=1.0, dx=0, dy=0, clip=None, align="c"):
    """cx,cy = 글자의 시각적 중심(align='l'이면 왼쪽 끝). 진행폭 반환."""
    m, adv, P = glyph(s, w, size)
    m = scaled(m, scale)
    ox = (P + adv / 2) * scale if align == "c" else P * scale
    blit(F, m, cx - ox + dx, cy - vcenter(w, size) * scale + dy, col, a, clip)
    return adv * scale


def rise(F, s, w, size, cx, cy, col, t, t0, dur=0.6, align="c", a=1.0, shave=0.0):
    """아래에서 위로 마스크 안에서 떠오르는 애플식 등장."""
    p = prog(t, t0, dur)
    if p <= 0:
        return
    hgt = size * 1.2
    clip = (cy - hgt * 0.6 + shave * hgt, cy + hgt * 0.55)
    text(F, s, w, size, cx, cy, col, a=a * clamp(p * 3), dy=(1 - eo(p)) * hgt, clip=clip, align=align)


def adv(s, w, size):
    return glyph(s, w, size)[1]


def line(F, parts, size, cx, cy, t, starts, dur=0.6, shaves=None):
    """여러 색·굵기를 한 줄에. parts=[(문자열, 굵기, 색)], 단어별 시차 등장."""
    tot = sum(adv(s, w, size) for s, w, _ in parts)
    x = cx - tot / 2
    for i, ((s, w, col), st) in enumerate(zip(parts, starts)):
        sh = shaves[i] if shaves else 0.0
        rise(F, s, w, size, x, cy, col, t, st, dur, align="l", shave=sh)
        x += adv(s, w, size)
    return cx - tot / 2


def chars(F, s, w, size, cx, cy, col, track=0.0, offs=None, a=1.0):
    """글자 단위 배치(자간 애니메이션, 흔들림)."""
    f = font(w, size)
    advs = [f.getlength(ch) for ch in s]
    x = cx - (sum(advs) + track * (len(s) - 1)) / 2
    for i, (ch, ad) in enumerate(zip(s, advs)):
        dx, dy, aa = offs(i) if offs else (0, 0, 1)
        if ch != " ":
            text(F, ch, w, size, x + ad / 2 + dx, cy + dy, col, a=a * aa)
        x += ad + track


def rect(F, x0, y0, x1, y1, col, a=1.0):
    x0, x1 = int(max(0, x0)), int(min(W, x1))
    y0, y1 = int(max(0, y0)), int(min(H, y1))
    if x1 <= x0 or y1 <= y0 or a <= 0:
        return
    if isinstance(col, str):
        c = grad(col, y1 - y0, x1 - x0)
    else:
        c = np.array(col, np.float32)
    reg = F[y0:y1, x0:x1]
    reg *= 1 - a
    reg += c * a


@functools.lru_cache(16)
def disc(r):
    y, x = np.mgrid[-r - 2:r + 2, -r - 2:r + 2].astype(np.float32)
    return np.clip(r + 0.5 - np.sqrt(x * x + y * y), 0, 1)


def dot(F, cx, cy, r, col, a=1.0):
    m = disc(int(r))
    blit(F, m, cx - m.shape[1] / 2, cy - m.shape[0] / 2, col, a)


@functools.lru_cache(8)
def blob(r):
    y, x = np.mgrid[-2 * r:2 * r, -2 * r:2 * r].astype(np.float32)
    return np.exp(-(x * x + y * y) / (r * r))[..., None]


def glow(F, cx, cy, r, col, a):
    if a <= 0.003:
        return
    b = blob(r)
    h, w = b.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    ax0, ay0, ax1, ay1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    F[ay0:ay1, ax0:ax1] += np.array(col, np.float32) * b[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] * a


def chapter(F, num, title, t, dark=True):
    p = eo(prog(t, 0.05, 0.7))
    c1 = WHITE if dark else DARK
    x, y = 140, 118
    text(F, num, "b", 34, x, y, c1, a=p, align="l", dx=-(1 - p) * 40)
    text(F, title, "m", 34, x + 66, y, GRAY, a=p, align="l", dx=-(1 - p) * 40)
    rect(F, x, y + 36, x + 66 * p, y + 39, c1, a=p)
    # 진행 표시 01 · 02 · 03
    n = int(num)
    for i in range(3):
        dot(F, W - 220 + i * 40, y, 6, c1 if i < n else (GRAY if dark else LGRAY), a=p)


def count(F, t, t0, v0, v1, fmt, size, cy, col, dur=0.9, glow_col=None, pop=0.1):
    p = prog(t, t0, dur)
    if p <= 0:
        return
    v = v0 + (v1 - v0) * ec(p)
    if glow_col:
        glow(F, W / 2, cy, 430, glow_col, 0.22 * eo(prog(t, t0, 0.5)))
    s = (1 - pop) + pop * eo(prog(t, t0, 0.7))
    text(F, fmt(v), "xb", size, W / 2, cy, col, a=clamp(p * 5), scale=s)


# ───────────────────────── 장면 ─────────────────────────
# 각 장면: fn(F, t, d, c)  t=장면 안 시간, d=장면 길이, c=음성 구간 시작 시각 리스트
def s_hook1(F, t, d, c):
    line(F, [("5년", "xb", WHITE), (" 만에,", "b", WHITE)], 170, W / 2, H / 2, t, [c[0], c[0] + 0.2])


def s_hook2(F, t, d, c):
    rise(F, "매출", "m", 64, W / 2, 300, GRAY, t, c[0])
    count(F, t, c[1] - 0.05, 0, 56, lambda v: f"+{round(v)}%", 340, 570, "warm", glow_col=(255, 55, 95))
    rise(F, "2020 → 2025  ·  9개 매장", "m", 34, W / 2, 860, GRAY, t, c[1] + 0.6)


def s_hook3(F, t, d, c):
    p = eo(prog(t, 0.0, 0.55))
    chars(F, "그런데.", "xb", 220, W / 2, H / 2, DARK, track=70 * (1 - p), a=clamp(p * 3))


def s_hook4(F, t, d, c):
    line(F, [("숫자가 ", "sb", LGRAY), ("말하지 ", "sb", LGRAY), ("않은", "sb", LGRAY)], 92,
         W / 2, 400, t, [c[0], c[0] + 0.15, c[0] + 0.3])
    p = eo(prog(t, c[1], 0.55))
    if p > 0:
        glow(F, W / 2, 600, 380, (10, 132, 255), 0.2 * p)
        text(F, "세 가지.", "xb", 240, W / 2, 600, "cool", a=clamp(p * 3), scale=1.25 - 0.25 * p)
    for i, n in enumerate(["01", "02", "03"]):
        rise(F, n, "sb", 32, W / 2 + (i - 1) * 110, 850, GRAY, t, c[1] + 0.35 + i * 0.1, dur=0.4)


CH = [("01", "성장의 엔진이 바뀌었다"), ("02", "인건비가 이익을 깎는다"), ("03", "자사앱의 착시")]


def s_m1a(F, t, d, c):
    chapter(F, *CH[0], t)
    rise(F, "주문은,", "b", 120, W / 2, 420, WHITE, t, c[0])
    t0 = c[1] - 0.05
    p = prog(t, t0, 0.26)
    if p > 0:
        if p < 1:  # 빠르게 들어와서
            off = (1 - p ** 1.6) * 1300
            for k, ga in ((3, 0.12), (2, 0.2), (1, 0.35)):  # 모션 블러 잔상
                text(F, "멈췄습니다.", "xb", 210, W / 2 + off + k * 70, 640, WHITE, a=ga)
            text(F, "멈췄습니다.", "xb", 210, W / 2 + off, 640, WHITE)
        else:      # 그대로 정지 (반동 한 번)
            k = t - t0 - 0.26
            rec = -18 * math.sin(k * 30) * math.exp(-k * 14)
            text(F, "멈췄습니다.", "xb", 210, W / 2 + rec, 640, WHITE)
    line(F, [("총 주문 수  37.7만 → 36.6만건", "m", GRAY), ("   2023 → 2025", "m", DGRAY)], 36,
         W / 2, 880, t, [c[1] + 0.6, c[1] + 0.7])


def s_m1b(F, t, d, c):
    chapter(F, *CH[0], t + 5)
    rise(F, "성장은,", "sb", 90, W / 2, 380, GRAY, t, c[0])
    p = prog(t, c[1], 0.6)
    if p > 0:
        glow(F, W / 2 - 330, 600, 300, (255, 55, 95), 0.16 * eo(p))
    line(F, [("가격", "xb", "warm"), ("이 만들었습니다.", "xb", WHITE)], 170, W / 2, 600, t, [c[1], c[1] + 0.18])
    line(F, [("객단가  2.65만 → 3.00만원   ", "m", GRAY), ("+13%", "b", "warm")], 44,
         W / 2, 870, t, [c[1] + 0.6, c[1] + 0.75])


def s_m2a(F, t, d, c):
    chapter(F, *CH[1], t, dark=False)
    rise(F, "1인당 인건비", "sb", 70, W / 2, 310, DGRAY, t, c[0])
    count(F, t, c[1] - 0.05, 0, 89, lambda v: f"+{round(v)}%", 340, 580, "hot")
    rise(F, "월 172만 → 325만원  ·  인원은 그대로", "m", 36, W / 2, 860, GRAY, t, c[1] + 0.7)


def s_m2b(F, t, d, c):
    chapter(F, *CH[1], t + 5)
    rise(F, "이익을,", "sb", 90, W / 2, 380, GRAY, t, c[0])
    t_sh = c[1] + 0.7
    sh = 0.32 * ec(prog(t, t_sh, 0.7))  # '깎고'가 위에서부터 깎여 나감
    x0 = line(F, [("조용히 ", "xb", WHITE), ("깎고", "xb", RED), (" 있습니다.", "xb", WHITE)], 160, W / 2, 600, t,
              [c[1], c[1] + 0.15, c[1] + 0.3], shaves=[0, sh, 0])
    if t > t_sh:  # 깎여 떨어지는 조각
        gx = x0 + adv("조용히 ", "xb", 160)
        gw = adv("깎고", "xb", 160)
        rng = np.random.default_rng(7)
        for i in range(14):
            k = t - t_sh - i * 0.04
            if k <= 0:
                continue
            px = gx + rng.uniform(0.05, 0.95) * gw + k * rng.uniform(-60, 60)
            py = 600 - 160 * 0.72 + sh * 160 * 1.2 + 260 * k * k + 40 * k
            rect(F, px, py, px + rng.uniform(10, 26), py + 5, RED, a=clamp(1 - k / 1.2))
    line(F, [("인건비율  24.4% → 29.2%   ", "m", GRAY), ("+4.8%p", "b", RED)], 44,
         W / 2, 870, t, [c[1] + 0.8, c[1] + 0.95])


def s_m3a(F, t, d, c):
    chapter(F, *CH[2], t)
    rise(F, "자사앱 주문 비중", "sb", 70, W / 2, 290, GRAY, t, c[0])
    t0 = c[1] - 0.05
    count(F, t, t0, 6, 31, lambda v: f"{round(v)}%", 300, 520, "cool", glow_col=(10, 132, 255))
    p = prog(t, t0, 0.9)
    if p > 0:
        x0, x1, y = 360, 1560, 760
        rect(F, x0, y, x1, y + 14, TRACK, a=clamp(p * 4))
        v = 6 + 25 * ec(p)
        rect(F, x0, y, x0 + (x1 - x0) * v / 100, y + 14, "cool", a=clamp(p * 4))
    rise(F, "2020년 6%  →  2026년 31%", "m", 34, W / 2, 850, GRAY, t, c[1] + 0.6)


def s_m3b(F, t, d, c):
    chapter(F, *CH[2], t + 5, dark=False)
    rise(F, "하지만 배달앱 비용은,", "sb", 76, W / 2, 330, DGRAY, t, c[0])
    if t >= c[1]:  # 움직임 없이 '그대로'
        text(F, "그대로.", "xb", 250, W / 2, 560, DARK)
    vals = [17.2, 16.6, 15.7, 16.2, 16.0, 15.8]  # 수수료+쿠폰, 매출 대비 %
    x0, x1, yb = 460, 1460, 790
    p = ec(prog(t, c[1] + 0.2, 1.0))
    if p > 0:
        pts = [(x0 + (x1 - x0) * i / 5, yb - (v - 16) * 14) for i, v in enumerate(vals)]
        xe = x0 + (x1 - x0) * p
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            n = 40
            for j in range(n):
                xx = ax + (bx - ax) * j / n
                if xx > xe:
                    break
                yy = ay + (by - ay) * j / n
                rect(F, xx, yy - 2, xx + (bx - ax) / n + 1, yy + 2, DARK)
        for i, (px, py) in enumerate(pts):
            if px <= xe + 1:
                dot(F, px, py, 9, DARK)
                text(F, str(2020 + i), "m", 24, px, yb + 52, GRAY)
    rise(F, "배달앱 수수료 + 쿠폰  =  매출의 16%  ·  5년째", "m", 34, W / 2, 920, GRAY, t, c[1] + 1.0)


def s_c1(F, t, d, c):
    s = 1 + 0.04 * (t / d)
    p = prog(t, c[0], 0.9)
    if p > 0:
        text(F, "그리고 지금,", "m", 110, W / 2, H / 2, WHITE, a=clamp(p * 2), scale=s, dy=(1 - eo(p)) * 40)


def s_c2(F, t, d, c):
    rise(F, "두 매장이,", "b", 130, W / 2, 410, WHITE, t, c[0])
    p = prog(t, c[1] - 0.05, 0.4)
    if p > 0:
        amp = 4 + 10 * (0.5 + 0.5 * math.sin(t * 9))

        def off(i):
            ph = i * 1.7
            return (amp * math.sin(t * 47 + ph) * math.sin(t * 13 + ph * 2),
                    amp * 0.7 * math.sin(t * 59 + ph * 3), clamp(p * 3))
        chars(F, "흔들리고 있습니다.", "xb", 170, W / 2, 620, RED, offs=off)
    rise(F, "광주 충장로점  ·  제주 연동점   이익률 하락", "m", 34, W / 2, 860, GRAY, t, c[1] + 0.6)


def s_c3(F, t, d, c):
    rise(F, "다음은,", "m", 100, W / 2, 400, GRAY, t, c[0])
    rise(F, "어디일까요?", "xb", 200, W / 2, 590, WHITE, t, c[1], dur=0.7)
    # 9개 매장: 이미 흔들리는 2곳(빨강) + 다음 후보가 깜빡임
    for i in range(9):
        p = prog(t, c[0] + i * 0.06, 0.3)
        x = W / 2 + (i - 4) * 70
        if i in (7, 8):
            dot(F, x, 860, 11, RED, a=p)
        elif i in (2, 5):
            blink = 0.5 + 0.5 * math.sin((t - c[1]) * 14) if t > c[1] else 0
            dot(F, x, 860, 11, GRAY, a=p)
            dot(F, x, 860, 11, WHITE, a=p * blink)
        else:
            dot(F, x, 860, 11, (60, 60, 64), a=p)


def s_end(F, t, d, c):
    if t < 0.9:
        p = eo(prog(t, 0, 0.6))
        glow(F, W / 2, H / 2, 420, (255, 55, 95), 0.25 * p)
        text(F, "?", "xb", 560, W / 2, H / 2, "warm", a=clamp(p * 4), scale=1.4 - 0.4 * p)
        return
    fade = clamp((d - t) / 0.45)
    p = eo(prog(t, 1.05, 0.8))
    text(F, "다음 편에서 계속됩니다.", "sb", 72, W / 2, 505, WHITE, a=p * fade, dy=(1 - p) * 20)
    chars(F, "PIZZA INSIGHTS · PART 2", "sb", 28, W / 2, 610, GRAY, track=9, a=eo(prog(t, 1.35, 0.8)) * fade)


# 음성은 '|'로 구간을 나눔(그 시점에 화면이 바뀜). sfx = [(구간 번호 또는 None, 지연, 종류)]
BEATS = [
    dict(v="오 년 만에,",                       bg=BLACK,    fn=s_hook1, sfx=[(0, 0, "tick"), (0, .2, "tick")]),
    dict(v="매출,|오십육 퍼센트.",               bg=BLACK,    fn=s_hook2, sfx=[(0, 0, "tick"), (1, -.05, "boom")]),
    dict(v="그런데.",                           bg=WHITE_BG, fn=s_hook3, sfx=[(None, 0, "hit")]),
    dict(v="숫자가 말하지 않은,|세 가지.",       bg=BLACK,    fn=s_hook4, sfx=[(0, 0, "tick"), (0, .15, "tick"), (0, .3, "tick"), (1, 0, "hit")]),
    dict(v="주문은,|멈췄습니다.",               bg=BLACK,    fn=s_m1a,   sfx=[(None, 0, "boom"), (0, 0, "tick"), (1, -.1, "whoosh"), (1, .21, "thunk")]),
    dict(v="성장은,|가격이 만들었습니다.",       bg=BLACK,    fn=s_m1b,   sfx=[(0, 0, "tick"), (1, 0, "hit")]),
    dict(v="일 인당 인건비,|팔십구 퍼센트 증가.", bg=WHITE_BG, fn=s_m2a,   sfx=[(None, 0, "hit"), (0, 0, "tick"), (1, -.05, "boom")]),
    dict(v="이익을,|조용히 깎고 있습니다.",      bg=BLACK,    fn=s_m2b,   sfx=[(0, 0, "tick"), (1, 0, "tick"), (1, .7, "shave")]),
    dict(v="자사앱 주문,|삼십일 퍼센트.",        bg=BLACK,    fn=s_m3a,   sfx=[(None, 0, "boom"), (0, 0, "tick"), (1, -.05, "rise_s")]),
    dict(v="하지만 배달앱 비용은,|그대로입니다.", bg=WHITE_BG, fn=s_m3b,   sfx=[(0, 0, "tick"), (1, 0, "thunk")]),
    dict(v="그리고 지금,",                      bg=BLACK,    fn=s_c1,    sfx=[(0, 0, "whoosh")]),
    dict(v="두 매장이,|흔들리고 있습니다.",      bg=BLACK,    fn=s_c2,    sfx=[(0, 0, "tick"), (1, -.05, "hit"), (1, -.05, "rumble")]),
    dict(v="다음은,|어디일까요?",               bg=BLACK,    fn=s_c3,    sfx=[(0, 0, "tick"), (1, 0, "tick"), (None, 0, "riser")]),
]
END_BG = BLACK


# ───────────────────────── 음성 ─────────────────────────
def read_wav(path):
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return x


def write_wav(path, st):
    st = np.clip(st, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


def tts(text_, rate, base):
    subprocess.run(["say", "-v", VOICE, "-r", str(rate), "-o", base + ".aiff", text_], check=True)
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", base + ".aiff", "-ar", str(SR), "-ac", "1",
                    base + ".wav"], check=True)
    x = read_wav(base + ".wav")
    nz = np.where(np.abs(x) > 0.012)[0]
    x = x[max(0, nz[0] - 200): nz[-1] + 1200]
    return x / (np.abs(x).max() + 1e-9) * 0.9


def cue_times(v, L, lead):
    segs = v.split("|")
    wts = [len(s.replace(" ", "")) + (1.5 if s.strip()[-1:] in ",." else 0) for s in segs]
    tot, cum, out = sum(wts), 0, []
    for wt in wts:
        out.append(lead + L * cum / tot)
        cum += wt
    return out


def build_timeline(tmp):
    LEAD, END_T = 0.15, 2.4
    for rate in (195, 205, 215, 225, 235):
        clips = [tts(b["v"].replace("|", " "), rate, os.path.join(tmp, f"v{i}")) for i, b in enumerate(BEATS)]
        lens = [len(c) / SR for c in clips]
        speech = sum(LEAD + L for L in lens)
        if speech + 0.3 * len(BEATS) + 2.0 <= TOTAL:
            break
    gap = clamp((TOTAL - END_T - speech) / len(BEATS), 0.3, 0.8)
    t, starts, durs, cues = 0.0, [], [], []
    for b, L in zip(BEATS, lens):
        starts.append(t); durs.append(LEAD + L + gap)
        cues.append(cue_times(b["v"], L, LEAD))
        t += LEAD + L + gap
    end_start = t
    print(f"  음성 속도 {rate}, 문장 사이 {gap:.2f}s, 엔딩 {TOTAL - end_start:.2f}s")
    return clips, starts, durs, cues, end_start, LEAD


# ───────────────────────── 효과음·음악 ─────────────────────────
rng = np.random.default_rng(1)


def env(n, a=0.002, decay=0.2):
    t = np.arange(n) / SR
    return np.minimum(t / a, 1) * np.exp(-t / decay)


def lp(x, k):
    return np.convolve(x, np.ones(k) / k, mode="same")


def sfx(kind, dur=1.0):
    if kind == "tick":
        n = int(0.05 * SR); t = np.arange(n) / SR
        return (np.sin(2 * np.pi * 2600 * t) * 0.5 + rng.normal(0, 0.3, n)) * env(n, 0.0005, 0.008) * 0.22
    if kind in ("boom", "hit"):
        n = int((1.4 if kind == "boom" else 0.6) * SR); t = np.arange(n) / SR
        f = 38 + 85 * np.exp(-t * 9)
        body = np.sin(2 * np.pi * np.cumsum(f) / SR)
        click = lp(rng.normal(0, 1, n), 6) * env(n, 0.0005, 0.012)
        g = 0.75 if kind == "boom" else 0.45
        return (body * env(n, 0.002, 0.45 if kind == "boom" else 0.16) + click * 0.5) * g
    if kind == "thunk":
        n = int(0.3 * SR); t = np.arange(n) / SR
        return np.sin(2 * np.pi * (70 + 120 * np.exp(-t * 30)) * t) * env(n, 0.001, 0.07) * 0.6
    if kind == "whoosh":
        n = int(0.45 * SR); t = np.arange(n) / SR
        e = np.sin(np.pi * t / t[-1]) ** 2
        return lp(rng.normal(0, 1, n), 12) * e * 0.18
    if kind == "shave":
        n = int(0.5 * SR)
        x = rng.normal(0, 1, n); x = x - lp(x, 4)
        return x * env(n, 0.01, 0.15) * 0.12
    if kind in ("riser", "rise_s"):
        n = int((dur if kind == "riser" else 0.9) * SR); t = np.arange(n) / SR
        pr = t / t[-1]
        f = 180 + 900 * pr ** 2
        y = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.25 + lp(rng.normal(0, 1, n), 3) * 0.35
        return y * pr ** 2 * (0.32 if kind == "riser" else 0.15)
    if kind == "rumble":
        n = int(dur * SR); t = np.arange(n) / SR
        y = lp(rng.normal(0, 1, n), 300) * 8 * (1 + 0.5 * np.sin(2 * np.pi * 6 * t))
        e = np.minimum(t / 0.15, 1) * np.minimum((t[-1] - t) / 0.2, 1)
        return y * e * 0.35
    raise ValueError(kind)


def add(buf, x, at):
    i = int(at * SR)
    if i >= len(buf) or i + len(x) <= 0:
        return
    if i < 0:
        x, i = x[-i:], 0
    x = x[: len(buf) - i]
    buf[i:i + len(x)] += x


def music(n, stop_at):
    """미니멀 펄스 + 패드. 클리프행어 직전에 고조, 엔딩 카드에서 완전히 끊음."""
    t = np.arange(n) / SR
    pad = sum(np.sin(2 * np.pi * f * t + i) * a for i, (f, a) in
              enumerate([(110, .5), (164.81, .35), (220, .3), (261.63, .22), (329.63, .12)]))
    pad *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t)
    pad = lp(pad, 8) * 0.05
    out = pad.copy()
    bpm = 100
    beat = 60 / bpm
    kick_n = int(0.3 * SR); kt = np.arange(kick_n) / SR
    kick = np.sin(2 * np.pi * (48 + 60 * np.exp(-kt * 25)) * kt) * env(kick_n, 0.001, 0.12)
    hat_n = int(0.03 * SR)
    for k in range(int(stop_at / beat * 2) + 1):
        tt = k * beat / 2
        if tt >= stop_at:
            break
        late = tt > stop_at - 8.5  # 클리프 구간
        if k % 2 == 0:
            add(out, kick * (0.22 if late else 0.15), tt)
        h = rng.normal(0, 1, hat_n); h = (h - lp(h, 3)) * env(hat_n, 0.0005, 0.01)
        add(out, h * (0.07 if late else 0.035), tt)
    fade = np.clip((stop_at - t) / 0.03, 0, 1)
    fin = np.clip(t / 0.8, 0, 1)
    return out * fade * fin


def build_audio(path, clips, starts, cues, end_start, lead):
    n = int(TOTAL * SR)
    voice = np.zeros(n, np.float32)
    fx = np.zeros(n, np.float32)
    for c, s in zip(clips, starts):
        add(voice, c, s + lead)
    for b, s, cu, i in zip(BEATS, starts, cues, range(len(BEATS))):
        for seg, delay, kind in b["sfx"]:
            at = s + (cu[seg] if seg is not None else 0) + delay
            if kind == "rumble":
                x = sfx(kind, dur=starts[i] + (end_start - s if i == len(BEATS) - 1 else
                                               (starts[i + 1] - s)) - at + 0.05)
            elif kind == "riser":
                x = sfx(kind, dur=end_start - at)
            else:
                x = sfx(kind)
            add(fx, x, at)
    add(fx, sfx("boom"), end_start)
    m = music(n, end_start)
    duck = lp(np.abs(voice), int(0.08 * SR))
    duck = 1 - 0.55 * np.clip(duck * 6, 0, 1)
    mix = voice * 1.0 + fx * 0.9 + m * duck
    fade_out = np.clip((TOTAL - np.arange(n) / SR) / 0.4, 0, 1)
    mix *= fade_out
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.95
    write_wav(path, np.stack([mix, mix], 1))


# ───────────────────────── 렌더 ─────────────────────────
def render(out, audio, starts, durs, cues, end_start):
    cmd = [FF, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", audio, "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", str(TOTAL), out]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    nf = int(TOTAL * FPS)
    for k in range(nf):
        t = k / FPS
        if t >= end_start:
            bg, fn, tl, d, cu = END_BG, s_end, t - end_start, TOTAL - end_start, []
        else:
            i = max(j for j, s in enumerate(starts) if s <= t)
            bg, fn, tl, d, cu = BEATS[i]["bg"], BEATS[i]["fn"], t - starts[i], durs[i], cues[i]
        F = np.empty((H, W, 3), np.float32)
        F[:] = bg
        fn(F, tl, d, cu)
        pr.stdin.write(np.clip(F, 0, 255).astype(np.uint8).tobytes())
        if k % 90 == 0:
            print(f"  {t:5.1f}s / {TOTAL:.0f}s", flush=True)
    pr.stdin.close()
    if pr.wait() != 0:
        sys.exit("ffmpeg 인코딩 실패")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="피자_인사이트_30초.mp4")
    ap.add_argument("--tmp", default=None)
    a = ap.parse_args()
    tmp = a.tmp or tempfile.mkdtemp()
    os.makedirs(tmp, exist_ok=True)
    print("1/3 내레이션 생성")
    clips, starts, durs, cues, end_start, lead = build_timeline(tmp)
    print("2/3 음악·효과음 믹스")
    audio = os.path.join(tmp, "mix.wav")
    build_audio(audio, clips, starts, cues, end_start, lead)
    print("3/3 프레임 렌더링")
    render(a.out, audio, starts, durs, cues, end_start)
    print("완료:", a.out)


if __name__ == "__main__":
    main()
