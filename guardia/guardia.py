import math
import os
import sys
import time

print("art by lucas_de\\/", flush=True)
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import numpy as np
import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1100, 840
FONT_SIZE = 9
FPS_RECORD = 30
DURATION = 84.0
WATERMARK = "lucas_de\\/"
MESSAGE = "GUARDIA "
RAMP = ".:;=+*xo%#&@"
GLITCH = "01<>{}[]#$%&*+=?!ABXZ"
BG = (0, 0, 0)
SCALE = 285
CY = H * 0.5
CAM = 5.0
LIGHT = np.array([-0.5, 0.6, 0.75])

# ------------------------------------------------------------------ timeline (s)
WM_END = 3.4            # watermark typed, then glitched away
RAIN = (3.2, 9.6)       # matrix rain, the logo condenses out of it
ASSEMBLE = 4.2
SPIN = (9.5, 14.0)      # full 3D turn of the logo
SCAN = (11.6, 13.8)     # laser scan
MORPH = 2.2             # length of one morph (plus up to MORPH_SPREAD of stagger)
MORPH_SPREAD = 0.9
# shape, time its morph starts
SEQ = [("logo", 0.0), ("knot", 14.0), ("fish", 19.5), ("helix", 26.0), ("jellies", 31.5),
       ("swarm", 37.0), ("space", 42.5), ("blackhole", 53.0), ("galaxy", 61.0), ("comet", 68.0),
       ("logo", 76.0)]
SPACE = 42.5            # leave the ocean for space
PULLBACK = (47.8, 51.0) # camera pulls back from Saturn to reveal the system
BH = 53.0               # black hole
GALAXY = 61.0           # spiral galaxy
COMET = 68.0            # comet streaks past an ice giant through an asteroid field
DISSOLVE = 80.4         # letter rain, watermark returns

RECORD = "--record" in sys.argv
OUT_DIR = "frames_guardia"
if RECORD:
    os.makedirs(OUT_DIR, exist_ok=True)

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("guardia.py")
MONO = "consolas,menlo,dejavusansmono,liberationmono,couriernew"
font = pygame.font.SysFont(MONO, FONT_SIZE, bold=True)
font_big = pygame.font.SysFont(MONO, 40, bold=True)
font_small = pygame.font.SysFont(MONO, 14, bold=True)
cw, ch = font.size("M")
COLS, ROWS = W // cw, H // ch

RAMP_ARR = np.array(list(RAMP))
LETTERS = np.array(list("GUARDIAlucasde"))
MSG_GRID = np.array([[MESSAGE[(r * COLS + c) % len(MESSAGE)] for c in range(COLS)]
                     for r in range(ROWS)])
BIN_GRID = np.array([["01"[(r * 7 + c * 3 + (r * c) % 5) % 2] for c in range(COLS)]
                     for r in range(ROWS)])
rng = np.random.default_rng(42)
SETTLE = 0.12 + rng.random((ROWS, COLS)) * 0.5


# ------------------------------------------------------------------ math helpers
def ss(a, b, x):
    k = np.clip((x - a) / (b - a), 0.0, 1.0)
    return k * k * (3 - 2 * k)


def sms(a, b, x):
    """smootherstep: zero velocity and acceleration at both ends, for silky morphs"""
    k = np.clip((x - a) / (b - a), 0.0, 1.0)
    return k * k * k * (k * (k * 6 - 15) + 10)


def fs(a, b, x):
    return float(ss(a, b, x))


def bump(a, b, c, d, x):
    return fs(a, b, x) * (1 - fs(c, d, x))


def rot(yaw, pitch, roll):
    c, s = math.cos(yaw), math.sin(yaw)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    c, s = math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    c, s = math.cos(roll), math.sin(roll)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def unit(n, g):
    return norm(g.normal(size=(n, 3)))


def palette(h):
    return 0.5 + 0.5 * np.cos(2 * math.pi * (np.asarray(h)[..., None] + np.array([0.0, 0.33, 0.67])))


CYAN = np.array([0.1, 0.95, 1.0])
BLUE = np.array([0.15, 0.4, 1.0])
FUCHSIA = np.array([1.0, 0.12, 0.8])
VIOLET = np.array([0.62, 0.2, 1.0])
WHITE = np.array([1.0, 1.0, 1.0])


def tube(c, T, r, th):
    """points on a tube of radius r around centreline c with tangent T"""
    T = norm(T)
    ref = np.where(np.abs(T[:, 1:2]) > 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    b1 = norm(np.cross(T, ref))
    b2 = np.cross(T, b1)
    nn = np.cos(th)[:, None] * b1 + np.sin(th)[:, None] * b2
    return c + r[:, None] * nn, nn


def cone(base, tip, rad, h, phi):
    ax = tip - base
    T = np.broadcast_to(norm(ax), (len(h), 3))
    c = base + ax * h[:, None]
    return tube(c, T, rad * (1 - h), phi)


def split(n, fracs):
    cnt = (np.array(fracs) / sum(fracs) * n).astype(int)
    cnt[-1] = n - cnt[:-1].sum()
    return np.split(np.arange(n), np.cumsum(cnt)[:-1])


# ------------------------------------------------------------------ the real logo
src = pygame.image.load(os.path.join(HERE, "guardia_logo.png")).convert()
A = pygame.surfarray.array3d(src).transpose(1, 0, 2).astype(float) / 255
IH, IW = A.shape[:2]
ink = A.min(2) < 0.80
ink[236:] = False
TEXT_ROW = 165

hmap = ink.astype(float)
for _ in range(4):
    hmap = (hmap + np.roll(hmap, 1, 0) + np.roll(hmap, -1, 0)
            + np.roll(hmap, 1, 1) + np.roll(hmap, -1, 1)) / 5
hmap *= ink
dhy, dhx = np.gradient(hmap)

yy, xx = np.nonzero(ink)
px = A[yy, xx]
cyan = (px[:, 2] - px[:, 0]) > 0.25
gold = ((px[:, 0] - px[:, 2]) > 0.10) & ~cyan
dark = ~cyan & ~gold
is_text = yy >= TEXT_ROW
PIX = 2.5 / IW
lnrm = norm(np.stack([-dhx[yy, xx] * 5, dhy[yy, xx] * 5, np.ones(len(xx))], -1))
lcol = px / np.maximum(px.max(1, keepdims=True), 0.05)
yfrac = yy / IH
silver = np.stack([0.78 + 0.15 * (1 - yfrac), 0.83 + 0.12 * (1 - yfrac), 0.95 + 0 * yfrac], -1)
lcol[dark] = silver[dark]
lcol[gold] = np.clip(lcol[gold] * np.array([1.0, 0.86, 0.62]) + 0.05, 0, 1)
lz = np.select([cyan & ~is_text, gold & ~is_text, is_text], [0.65, 0.32, -0.55], -0.02)
front = np.stack([(xx - IW / 2) * PIX, -(yy - 118) * PIX, hmap[yy, xx] * np.where(is_text, 0.06, 0.13)], -1)
bk = np.nonzero(~is_text)[0][::3]
LOGO_P = np.concatenate([front, front[bk] * np.array([1, 1, 0]) - np.array([0, 0, 0.11])])
LOGO_N = np.concatenate([lnrm, lnrm[bk] * np.array([1, 1, -1])])
LOGO_C = np.concatenate([lcol, lcol[bk] * 0.5])
LOGO_LZ = np.concatenate([lz, lz[bk]])
N = len(LOGO_P)
LOGO_X = LOGO_P[:, 0]


def logo_yaw(t):
    return 2 * math.pi * fs(SPIN[0], SPIN[1], t) + 0.22 * math.sin(t * 0.5) * fs(SPIN[1], SPIN[1] + 1.5, t)


def shape_logo(t):
    p = LOGO_P
    sheen = np.maximum(np.sin(LOGO_X * 3 + p[:, 1] * 2 - t * 2.4), 0) ** 12
    glow = 0.35 * sheen
    pitch = 0.15 * math.sin(math.pi * fs(SPIN[0], SPIN[1], t)) + 0.05 * math.sin(t * 0.7)
    R = rot(logo_yaw(t), pitch, 0.04 * math.sin(t * 1.1))
    return p @ R.T, LOGO_N @ R.T, LOGO_C, glow


# ------------------------------------------------------------------ torus knot
def build_knot(g):
    u = g.random(N) * 2 * math.pi
    th = g.random(N) * 2 * math.pi

    def curve(u):
        r = 0.55 + 0.24 * np.cos(3 * u)
        return np.stack([r * np.cos(2 * u), r * np.sin(2 * u), 0.26 * np.sin(3 * u)], -1) * 1.3

    T = curve(u + 1e-3) - curve(u - 1e-3)
    C = curve(u)

    def f(t):
        bulge = np.maximum(np.sin(u * 6 - t * 4.5), 0) ** 6
        p, n = tube(C, T, 0.09 + 0.07 * bulge, th + t * 0.8)
        col = BLUE * (0.7 + 0.3 * np.cos(th + t * 0.8))[:, None] + bulge[:, None] * (CYAN - BLUE)
        R = rot(t * 0.55, 0.55 + 0.2 * math.sin(t * 0.4), 0.2)
        return p @ R.T, n @ R.T, col, 0.7 * bulge
    return f


# ------------------------------------------------------------------ fish, drawn flat like a line sketch
def build_fish(g):
    segs = []  # (start, end, colour, part)  part: 0 body 1 tail 2 fins 3 eye 4 scales

    def poly(pts, col_, part, w=1.0):
        for a, b in zip(pts[:-1], pts[1:]):
            segs.append((np.array(a, float), np.array(b, float), col_, part, w))

    def arc(c, r, a0, a1, col_, part, n=18, w=1.0):
        a = np.linspace(a0, a1, n)
        poly(list(zip(c[0] + r * np.cos(a), c[1] + r * np.sin(a))), col_, part, w)

    def up(u):
        return 0.06 * u + 0.4 * np.sin(np.pi * u) ** 0.8

    def lo(u):
        return -(0.06 * u + 0.34 * np.sin(np.pi * u) ** 0.85)

    u = np.linspace(0, 1, 50)
    x = 0.9 - 1.5 * u
    poly(list(zip(x, up(u))), CYAN, 0)
    poly(list(zip(x, lo(u))), CYAN, 0)
    poly([(-0.6, 0.06), (-1.05, 0.45), (-0.9, 0.0), (-1.05, -0.45), (-0.6, -0.06)], FUCHSIA, 1)
    for ty in (0.32, 0.16, -0.16, -0.32):
        poly([(-0.63, ty * 0.12), (-0.97 + 0.05 * (abs(ty) < 0.2), ty)], FUCHSIA, 1, 0.6)
    poly([(0.3, 0.39), (0.05, 0.72), (-0.15, 0.64), (-0.33, 0.33)], FUCHSIA, 2)
    for k in np.linspace(0.1, 0.9, 5):
        bx = 0.28 - 0.58 * k
        poly([(bx, up((0.9 - bx) / 1.5) - 0.01), (0.05 - 0.2 * k + 0.05, 0.72 - 0.1 * k * k - 0.05)], FUCHSIA, 2, 0.5)
    poly([(0.35, -0.08), (0.06, -0.33), (0.0, -0.22), (0.3, -0.04)], FUCHSIA, 2)
    poly([(-0.2, -0.31), (-0.38, -0.53), (-0.52, -0.24)], FUCHSIA, 2)
    arc((0.6, 0.1), 0.08, 0, 2 * math.pi, WHITE, 3, 24)
    for r_ in (0.035, 0.018):
        arc((0.62, 0.1), r_, 0, 2 * math.pi, WHITE, 3, 12)
    arc((0.78, 0.0), 0.32, 0.72 * math.pi, 1.28 * math.pi, CYAN, 0)
    poly([(0.9, -0.02), (0.8, -0.07)], CYAN, 0)
    for row, cy in enumerate(np.arange(-0.24, 0.3, 0.1)):
        for cx in np.arange(-0.48, 0.42, 0.13) + 0.065 * (row % 2):
            uu = (0.9 - cx) / 1.5
            if lo(uu) + 0.08 < cy < up(uu) - 0.08:
                arc((cx, cy), 0.06, 0.5 * math.pi, 1.5 * math.pi, BLUE, 4, 7, 0.55)

    A_ = np.array([sg[0] for sg in segs])
    B_ = np.array([sg[1] for sg in segs])
    L = np.linalg.norm(B_ - A_, axis=1) * np.array([sg[4] for sg in segs])
    nbub = int(N * 0.05)
    nl = N - nbub
    pick = g.choice(len(segs), nl, p=L / L.sum())
    st = g.random(nl)
    d = B_[pick] - A_[pick]
    perp = norm(np.stack([-d[:, 1], d[:, 0]], -1))
    S = A_[pick] + d * st[:, None] + perp * (g.normal(size=nl) * 0.011)[:, None]
    part = np.array([sg[3] for sg in segs])[pick]
    lcol = np.array([sg[2] for sg in segs])[pick]
    bub_id = g.integers(0, 3, nbub)
    bub_a = g.random(nbub) * 2 * math.pi

    def f(t):
        x, y = S[:, 0].copy(), S[:, 1].copy()
        wag = 0.32 * math.sin(t * 4.2)
        k = np.clip((-0.42 - x) / 0.6, 0, 1) ** 1.3
        a = wag * k
        dx = x + 0.45
        x, y = -0.45 + dx * np.cos(a) - y * np.sin(a), dx * np.sin(a) + y * np.cos(a)
        y = y + 0.025 * np.sin(x * 5 - t * 4.2) * (0.6 - 0.4 * x)
        fin = part == 2
        y = np.where(fin & (S[:, 1] < -0.05), y + 0.03 * math.sin(t * 6) * (S[:, 1] + 0.05) * -4, y)
        blink = 0.12 if (t % 3.3) < 0.13 else 1.0
        eye = part == 3
        y = np.where(eye, 0.1 + (y - 0.1) * blink, y)
        rk = 0.03 + 0.018 * bub_id
        rise = (t * 0.35 + bub_id / 3) % 1
        bx = 1.0 + 0.05 * np.sin(t * 2 + bub_id * 2) + rise * 0.15 + rk * np.cos(bub_a)
        by = -0.02 + rise * 1.1 + rk * np.sin(bub_a)
        P = np.zeros((N, 3))
        P[:nl, 0], P[:nl, 1] = x, y
        P[nl:, 0], P[nl:, 1] = bx, by
        P[:, :2] = P[:, :2] * 1.2 + np.array([0.15 * math.sin(t * 0.4) - 0.1, 0.07 * math.sin(t * 0.9)])
        col = np.concatenate([lcol, np.tile([0.7, 0.95, 1.0], (nbub, 1))])
        glow = np.concatenate([np.select([eye, part == 4], [0.9, 0.15], 0.45),
                               0.5 * (1 - ss(0.75, 1.0, rise))])
        return P, np.tile([0, 0, 1.0], (N, 1)), col, glow
    return f


# ------------------------------------------------------------------ DNA double helix
def build_helix(g):
    strand_n = int(N * 0.66)
    u = g.random(N)
    side = g.integers(0, 2, N)
    D = unit(N, g)
    level = g.integers(0, 26, N)
    q = g.random(N)

    def f(t):
        P = np.zeros((N, 3))
        col = np.zeros((N, 3))
        ang0 = t * 1.4
        st = slice(0, strand_n)
        y = -1.15 + 2.3 * u[st]
        a = u[st] * 5 * math.pi + ang0 + side[st] * math.pi
        c = np.stack([0.48 * np.cos(a), y, 0.48 * np.sin(a)], -1)
        P[st] = c + D[st] * 0.06
        col[st] = np.where((side[st] == 0)[:, None], CYAN, FUCHSIA)
        rg = slice(strand_n, N)
        lv = (level[rg] + 0.5) / 26
        y = -1.15 + 2.3 * lv
        a = lv * 5 * math.pi + ang0
        c0 = np.stack([0.48 * np.cos(a), y, 0.48 * np.sin(a)], -1)
        P[rg] = c0 * (1 - 2 * q[rg][:, None]) * np.array([1, 0, 1]) + np.stack([0 * y, y, 0 * y], -1) + D[rg] * 0.025
        pair = level[rg] % 4
        base = np.array([VIOLET, BLUE, VIOLET * 0.8 + 0.2, BLUE * 0.8 + 0.2])
        col[rg] = np.where((q[rg] < 0.5)[:, None], base[pair], base[(pair + 2) % 4])
        Nn = norm(P * np.array([1, 0, 1]) + D * 0.3)
        sweep = ((t * 0.9) % 3.0) - 1.5
        glow = 1.1 * np.exp(-((P[:, 1] - sweep) / 0.14) ** 2)
        R = rot(0.0, 0.25, 0.35 * math.sin(t * 0.5))
        return P @ R.T, Nn @ R.T, col, glow
    return f


# ------------------------------------------------------------------ jellyfish
JELLIES = [((-0.85, 0.15, -0.2), 0.72, FUCHSIA, 0.0), ((0.82, 0.35, 0.0), 0.62, CYAN, 1.7),
           ((0.0, -0.3, 0.45), 0.5, VIOLET, 3.1)]
SWARM = [((-1.25, 0.5, -0.4), 0.42, BLUE, 0.0), ((-0.45, -0.1, 0.2), 0.38, FUCHSIA, 0.9),
         ((0.3, 0.62, -0.3), 0.45, CYAN, 1.8), ((1.1, -0.05, 0.1), 0.4, VIOLET, 2.6),
         ((-0.95, -0.75, 0.3), 0.34, CYAN, 3.4), ((0.6, -0.8, -0.2), 0.36, FUCHSIA, 4.3)]


def build_jellies(g, specs):
    groups = split(N, [sp[1] ** 2 for sp in specs])
    parts = [split(len(idx), [0.45, 0.4, 0.15]) for idx in groups]
    phi = np.sqrt(g.random(N)) * math.pi * 0.55
    th = g.random(N) * 2 * math.pi
    s = g.random(N)
    tk = g.integers(0, 10, N)
    D = unit(N, g)

    def f(t):
        P = np.zeros((N, 3))
        Nn = np.zeros((N, 3))
        col = np.zeros((N, 3))
        glow = np.zeros(N)
        for idx, (bl, te, ar), (c0, sc, neon, ph0) in zip(groups, parts, specs):
            tt = t + ph0
            ctr = np.array(c0) + np.array([0.07 * math.sin(t * 0.5 + ph0), 0.05 * math.sin(t * 0.7 + ph0), 0])
            pulse = 0.5 + 0.5 * math.sin(tt * 2.6)
            rx = 0.62 * (1 - 0.14 * pulse)
            ry = 0.5 * (1 + 0.12 * pulse)
            bob = 0.12 * math.sin(tt * 2.6 - 1.0)
            bell = idx[bl]
            ph, tb = phi[bell], th[bell]
            wav = 1 + 0.06 * np.sin(tb * 8) * (ph / 1.73)
            p = np.stack([rx * np.sin(ph) * np.cos(tb) * wav, 0.45 + ry * np.cos(ph) + bob,
                          rx * np.sin(ph) * np.sin(tb) * wav], -1)
            P[bell] = p * sc + ctr
            Nn[bell] = norm(np.stack([np.sin(ph) * np.cos(tb), np.cos(ph), np.sin(ph) * np.sin(tb)], -1))
            rim = ss(1.45, 1.73, ph)
            col[bell] = neon * (0.45 + 0.4 * ph / 1.73)[:, None] * (1 - rim[:, None]) + (neon * 0.5 + 0.5) * rim[:, None]
            glow[bell] = 0.9 * rim * (0.6 + 0.4 * np.sin(tb * 8 + tt * 4)) + 0.15
            tent = idx[te]
            q = s[tent]
            a0 = tk[tent] * 2 * math.pi / 10
            root = np.stack([rx * 0.93 * np.cos(a0), 0.45 + bob + ry * math.cos(1.6) + 0 * q, rx * 0.93 * np.sin(a0)], -1)
            sway = np.stack([0.14 * np.sin(5 * q - tt * 3 + tk[tent]) * q, -1.55 * q + 0.15 * pulse * q,
                             0.14 * np.cos(4 * q - tt * 2.6 + tk[tent]) * q], -1)
            p = root * (1 - 0.25 * q[:, None]) * np.array([1, 0, 1]) + np.array([0, 1, 0]) * root + sway + D[tent] * 0.012
            P[tent] = p * sc + ctr
            Nn[tent] = norm(D[tent] + np.array([0, 0, 1.0]))
            col[tent] = neon * (1 - 0.6 * q[:, None])
            glow[tent] = 0.7 * np.exp(-((q - ((tt * 0.7 + tk[tent] * 0.13) % 1)) / 0.04) ** 2)
            arms = idx[ar]
            q = s[arms]
            a0 = (tk[arms] % 4) * math.pi / 2 + 0.3 * q * math.sin(tt)
            frill = 0.07 * (1 + np.sin(q * 30 - tt * 4)) * (1 - q)
            c = np.stack([0.06 * np.cos(a0) + 0.08 * np.sin(4 * q - tt * 2) * q, 0.45 + bob - 1.0 * q,
                          0.06 * np.sin(a0) + 0.08 * np.cos(3 * q - tt * 2) * q], -1)
            P[arms] = (c + D[arms] * (0.02 + frill)[:, None]) * sc + ctr
            Nn[arms] = D[arms]
            col[arms] = neon * 0.6 + 0.4
            glow[arms] = 0.3
        R = rot(0.25 * math.sin(t * 0.3), 0.18, 0.0)
        return P @ R.T, Nn @ R.T, col, glow
    return f


# ------------------------------------------------------------------ space: Saturn and friends
def build_space(g):
    groups = split(N, [0.34, 0.44, 0.11, 0.06, 0.05])
    D = unit(N, g)
    rr = 0.8 + 0.62 * g.random(N)
    rr = np.where((rr > 1.08) & (rr < 1.13), rr + 0.06, rr)    # Cassini division
    rr = np.where((rr > 1.31) & (rr < 1.33), rr + 0.03, rr)    # Encke gap
    ra = g.random(N) * 2 * math.pi
    ry = g.normal(size=N) * 0.005
    ringlet = 0.55 + 0.45 * np.sin(rr * 75) ** 2 * (0.7 + 0.3 * np.sin(rr * 13))
    noise = g.random(N)
    SAT_R = 0.52
    TILT = rot(0.35, 0.42, 0.2)
    CS = np.array([0.1, 0.0, 0.0])

    def sphere(idx, ctr, r, spin, t):
        d = D[idx]
        ang = t * spin
        c, s_ = math.cos(ang), math.sin(ang)
        dl = np.stack([d[:, 0] * c - d[:, 2] * s_, d[:, 1], d[:, 0] * s_ + d[:, 2] * c], -1)
        return ctr + d * r, d, dl

    def f(t):
        P = np.zeros((N, 3))
        Nn = np.zeros((N, 3))
        col = np.zeros((N, 3))
        glow = np.zeros(N)
        sat, ring, blue, pink, moon = groups
        # Saturn: oblate, banded gold
        d = D[sat] * np.array([1, 0.9, 1])
        lat = np.arcsin(np.clip(D[sat][:, 1], -1, 1))
        band = 0.5 + 0.5 * np.sin(lat * 17 + 0.7 * np.sin(lat * 5 + 1.3))
        c_ = np.array([1.0, 0.86, 0.58]) * band[:, None] + np.array([0.85, 0.52, 0.28]) * (1 - band[:, None])
        c_ = c_ * (0.75 + 0.25 * np.cos(lat))[:, None]
        P[sat], Nn[sat], col[sat] = d * SAT_R @ TILT.T + CS, D[sat] @ TILT.T, c_
        glow[sat] = 0.05
        # rings: Keplerian, icy cyan to violet, ringlets, planet shadow
        a = ra[ring] + t * 0.32 / rr[ring] ** 1.5
        r = rr[ring]
        p = np.stack([r * np.cos(a), ry[ring], r * np.sin(a)], -1) @ TILT.T
        kk = ((r - 0.8) / 0.68)[:, None]
        c_ = (np.array([0.8, 0.93, 1.0]) * (1 - kk) + np.array([0.72, 0.5, 1.0]) * kk) * ringlet[ring][:, None]
        Ln = LIGHT / np.linalg.norm(LIGHT)
        along = p @ Ln
        dist = np.linalg.norm(p - along[:, None] * Ln, axis=1)
        shadow = (along < 0) & (dist < SAT_R)
        c_[shadow] *= 0.12
        P[ring], col[ring] = p + CS, c_
        Nn[ring] = np.tile([0, 1.0, 0], (len(ring), 1)) @ TILT.T
        glow[ring] = (0.18 + 0.3 * ringlet[ring]) * ~shadow
        # electric-blue planet with a fuchsia atmosphere
        p, n, dl = sphere(blue, np.array([-1.5, 0.78, -0.9]), 0.27, 0.35, t)
        lat = np.arcsin(np.clip(dl[:, 1], -1, 1))
        lon = np.arctan2(dl[:, 2], dl[:, 0])
        swirl = 0.5 + 0.5 * np.sin(lat * 9 + 1.5 * np.sin(lon * 2 + lat * 3))
        rim = (1 - np.abs(n[:, 2])) ** 3
        c_ = BLUE * (0.6 + 0.4 * swirl[:, None]) + CYAN * 0.25 * swirl[:, None]
        P[blue], Nn[blue] = p, n
        col[blue] = c_ * (1 - rim[:, None]) + FUCHSIA * rim[:, None]
        glow[blue] = 0.8 * rim + 0.1
        # small fuchsia gas giant with violet bands
        p, n, dl = sphere(pink, np.array([1.6, -0.82, -0.6]), 0.17, 0.5, t)
        lat = np.arcsin(np.clip(dl[:, 1], -1, 1))
        band = 0.5 + 0.5 * np.sin(lat * 11)
        rim = (1 - np.abs(n[:, 2])) ** 3
        P[pink], Nn[pink] = p, n
        col[pink] = (FUCHSIA * band[:, None] + VIOLET * (1 - band[:, None])) * (1 - rim[:, None]) + CYAN * rim[:, None]
        glow[pink] = 0.7 * rim + 0.1
        # cratered moon orbiting Saturn
        ang = t * 0.55
        mc = CS + np.array([1.62 * math.cos(ang), 0, 1.62 * math.sin(ang)]) @ TILT.T
        p, n, dl = sphere(moon, mc, 0.075, 0.2, t)
        crater = (np.sin(dl[:, 0] * 23) * np.sin(dl[:, 1] * 19) * np.sin(dl[:, 2] * 17)) > 0.25
        P[moon], Nn[moon] = p, n
        col[moon] = np.where(crater[:, None], [0.45, 0.47, 0.55], [0.8, 0.82, 0.9])
        # camera: slow drift, then pull back to reveal the system
        zoom = 1.3 - 0.55 * float(sms(PULLBACK[0], PULLBACK[1], t))
        R = rot(0.18 * math.sin(t * 0.17) + 0.03 * (t - SPACE), 0.06, 0.0)
        return P @ R.T * zoom, Nn @ R.T, col, glow
    return f


# ------------------------------------------------------------------ black hole
def build_blackhole(g):
    groups = split(N, [0.62, 0.2, 0.08, 0.1])
    r = 0.42 + 1.15 * g.random(N) ** 1.4
    a0 = g.random(N) * 2 * math.pi
    yj = g.normal(size=N)
    th = g.random(N) * 2 * math.pi
    q = g.random(N)
    SH = 0.36  # shadow radius on screen

    def heat(k):
        k = np.clip(k, 0, 1)[:, None]
        return (WHITE * np.clip(1 - 3 * k, 0, 1) + FUCHSIA * np.clip(1 - np.abs(k * 3 - 1), 0, 1)
                + VIOLET * np.clip(1 - np.abs(k * 3 - 2), 0, 1) + BLUE * np.clip(k * 3 - 2, 0, 1))

    def f(t):
        P = np.zeros((N, 3))
        Nn = np.tile([0, 0, 1.0], (N, 1))
        col = np.zeros((N, 3))
        glow = np.zeros(N)
        disk, lens, photon, jets = groups
        zoom = 0.85 + 0.3 * float(sms(BH, BH + 7.5, t))
        # accretion disk, nearly edge-on, inner parts orbit faster
        rr = r[disk]
        ang = a0[disk] + t * 1.1 / rr ** 1.5
        p = np.stack([rr * np.cos(ang), yj[disk] * 0.012 * rr, rr * np.sin(ang)], -1)
        R = rot(0.25 * math.sin(t * 0.2), 0.2, -0.12)
        p = p @ R.T
        k = (rr - 0.42) / 1.15
        swirl = 0.65 + 0.35 * np.sin(rr * 38 + ang * 3 - t * 2.5)
        beam = 1 + 0.65 * np.sin(ang)               # the approaching side is brighter
        c = heat(k) * (swirl * beam * (1.1 - 0.5 * k))[:, None]
        hidden = (p[:, 2] < 0) & (np.hypot(p[:, 0], p[:, 1]) < SH * 1.05)
        c[hidden] = 0
        P[disk], col[disk] = p, c
        Nn[disk] = np.tile([0, 1.0, 0], (len(disk), 1)) @ R.T
        glow[disk] = (0.9 * (1 - k) ** 2 + 0.2) * ~hidden
        # far side of the disk, bent over and under the shadow by gravity
        rl = SH * 1.12 + 0.3 * q[lens] ** 1.6
        al = th[lens] + t * 0.6
        P[lens] = np.stack([rl * np.cos(al) * 1.05, rl * np.sin(al) * 0.92, -0.05 + 0 * al], -1) @ rot(0, 0, -0.12).T
        kl = (rl - SH) / 0.35
        col[lens] = (heat(kl) * (0.55 + 0.45 * np.abs(np.sin(al)))[:, None]
                     * (0.75 + 0.25 * np.sin(al * 5 - t * 3))[:, None])
        glow[lens] = 0.6 * (1 - kl)
        # photon ring
        ap = th[photon]
        P[photon] = np.stack([SH * np.cos(ap), SH * np.sin(ap), 0 * ap], -1) * (1 + 0.012 * yj[photon][:, None])
        col[photon] = np.array([0.85, 0.95, 1.0]) * (0.8 + 0.2 * np.sin(ap * 7 + t * 6))[:, None]
        glow[photon] = 1.3
        # relativistic jets
        sgn = np.where(q[jets] < 0.5, 1.0, -1.0)
        s_ = (q[jets] * 2 % 1 + t * 0.6) % 1
        spread = 0.015 + 0.08 * s_
        P[jets] = np.stack([yj[jets] * spread + 0.06 * s_ * sgn * math.sin(t), sgn * (0.3 + 1.4 * s_),
                            np.cos(th[jets]) * spread], -1) @ rot(0, 0, -0.12).T
        col[jets] = CYAN * (1 - s_)[:, None] + BLUE * s_[:, None]
        glow[jets] = 0.9 * (1 - s_)
        return P * zoom, Nn, col, glow
    return f


# ------------------------------------------------------------------ spiral galaxy
def build_galaxy(g):
    core_n = int(N * 0.15)
    r = g.random(N) ** 0.7 * 1.55
    arm = g.integers(0, 3, N)
    noise = g.normal(size=N)
    yj = g.normal(size=N)
    D = unit(N, g)
    knot_ = g.random(N) < 0.05        # star-forming knots
    dust = g.random(N) < 0.18

    def f(t):
        ang = arm * 2 * math.pi / 3 + np.log(r + 0.15) * 2.7 + noise * 0.3 / (r + 0.35) + t * 0.45 / (r + 0.4)
        P = np.stack([r * np.cos(ang), yj * 0.05 * (1 - r / 1.7), r * np.sin(ang)], -1)
        P[:core_n] = D[:core_n] * np.abs(yj[:core_n, None]) * 0.13 * np.array([1, 0.55, 1])
        k = np.clip(r / 1.55, 0, 1)[:, None]
        col = (VIOLET * 0.5 + 0.5 * WHITE) * (1 - k) + BLUE * k
        col = col * (0.6 + 0.4 * np.abs(np.sin(ang * 0.5)))[:, None]
        col[knot_] = FUCHSIA
        col[dust] *= 0.25
        col[:core_n] = [1.0, 0.85, 0.95]
        glow = 0.2 + 1.1 * np.exp(-r * 3) + knot_ * (0.7 + 0.5 * np.sin(t * 5 + noise * 4))
        glow[:core_n] = 1.0
        tilt = 1.15 - 0.55 * float(sms(GALAXY, GALAXY + 6.5, t))
        R = rot(0.15 * t, tilt, 0.3)
        zoom = 0.9 + 0.15 * float(sms(GALAXY, GALAXY + 6.5, t))
        return P @ R.T * zoom, np.tile([0, 1.0, 0], (N, 1)) @ R.T, col, glow
    return f


# ------------------------------------------------------------------ comet, ice giant, asteroids
AST = [((1.25, 0.55, 0.3), 0.11), ((0.55, -0.85, 0.5), 0.08), ((-0.25, 0.95, -0.2), 0.06),
       ((1.5, -0.35, -0.1), 0.07), ((0.1, -0.2, 0.9), 0.05), ((-1.5, 0.9, 0.2), 0.09)]


def build_comet(g):
    groups = split(N, [0.36, 0.02, 0.07, 0.26, 0.11, 0.18])
    D = unit(N, g)
    q = g.random(N)
    gj = g.normal(size=(N, 3))
    aid = g.integers(0, len(AST), N)
    actr = np.array([a[0] for a in AST])
    arad = np.array([a[1] for a in AST])

    def f(t):
        P = np.zeros((N, 3))
        Nn = np.tile([0, 0, 1.0], (N, 1))
        col = np.zeros((N, 3))
        glow = np.zeros(N)
        planet, nucleus, coma, dtail, itail, rocks = groups
        # ice giant, lower left
        d = D[planet]
        ang = t * 0.25
        dl = np.stack([d[:, 0] * math.cos(ang) - d[:, 2] * math.sin(ang), d[:, 1],
                       d[:, 0] * math.sin(ang) + d[:, 2] * math.cos(ang)], -1)
        lat = np.arcsin(np.clip(dl[:, 1], -1, 1))
        lon = np.arctan2(dl[:, 2], dl[:, 0])
        band = 0.5 + 0.5 * np.sin(lat * 8 + 1.2 * np.sin(lon * 3 + lat * 4))
        rim = (1 - np.abs(d[:, 2])) ** 3
        P[planet], Nn[planet] = np.array([-1.05, -0.8, -0.6]) + d * 0.8, d
        col[planet] = (CYAN * band[:, None] + BLUE * (1 - band[:, None])) * (1 - rim[:, None]) + VIOLET * rim[:, None]
        glow[planet] = 0.7 * rim
        # comet head sweeps right to left
        u = float(sms(COMET - 0.5, COMET + 8.5, t))
        head = np.array([2.0 - 3.6 * u, 0.85 - 0.55 * u + 0.15 * math.sin(u * 3), 0.3])
        P[nucleus], Nn[nucleus] = head + D[nucleus] * 0.035, D[nucleus]
        col[nucleus], glow[nucleus] = WHITE, 1.5
        P[coma] = head + D[coma] * (0.04 + 0.12 * q[coma] ** 2)[:, None]
        Nn[coma] = D[coma]
        col[coma] = CYAN * 0.5 + 0.5
        glow[coma] = 1.0 * (1 - q[coma])
        # curved dust tail and straight ion tail, both pointing away from the light
        tdir = norm(np.array([1.0, 0.45, -0.3]))
        s_ = q[dtail] ** 1.3 * 1.9
        curve = np.array([0.0, -0.18, 0.0]) * (s_ ** 2)[:, None]
        P[dtail] = head + tdir * s_[:, None] + curve + gj[dtail] * (0.015 + 0.13 * s_)[:, None]
        k = (s_ / 1.9)[:, None]
        col[dtail] = (CYAN * (1 - k) + BLUE * k) * (1 - 0.6 * k)
        glow[dtail] = 0.8 * (1 - k[:, 0]) ** 2
        s_ = q[itail] * 2.3
        flick = 0.02 * np.sin(s_ * 9 - t * 7)
        P[itail] = (head + norm(np.array([1.0, 0.6, -0.2])) * s_[:, None] + gj[itail] * 0.012
                    + np.array([0, 1.0, 0]) * flick[:, None])
        col[itail] = FUCHSIA * (1 - 0.6 * s_ / 2.3)[:, None]
        glow[itail] = 0.6 * (1 - s_ / 2.3)
        # tumbling lumpy asteroids
        a_ = aid[rocks]
        d = D[rocks]
        lump = 1 + 0.28 * np.sin(d[:, 0] * 5 + a_) * np.sin(d[:, 1] * 4) + 0.15 * np.sin(d[:, 2] * 7)
        spin = t * (0.6 + 0.2 * a_)
        cs, sn = np.cos(spin), np.sin(spin)
        dr = np.stack([d[:, 0] * cs - d[:, 1] * sn, d[:, 0] * sn + d[:, 1] * cs, d[:, 2]], -1)
        drift = np.stack([np.full(len(a_), -0.05 * (t - COMET)), 0.02 * np.sin(t + a_), np.zeros(len(a_))], -1)
        P[rocks], Nn[rocks] = actr[a_] + dr * (arad[a_] * lump)[:, None] + drift, dr
        col[rocks] = np.array([0.6, 0.55, 0.75]) * (0.7 + 0.3 * lump)[:, None]
        R = rot(0.1 * math.sin(t * 0.25), 0.05, 0.0)
        return P @ R.T, Nn @ R.T, col, glow
    return f


# ------------------------------------------------------------------ shape registry
class Shape:
    """Points are reordered by height so morphs sweep across the shapes, not a random scatter."""

    def __init__(self, fn, tref):
        self.fn = fn
        p = fn(tref)[0]
        self.perm = np.argsort(p[:, 1] + 0.03 * p[:, 0], kind="stable")

    def __call__(self, t):
        return tuple(a[self.perm] if np.ndim(a) else a for a in self.fn(t))


g0 = np.random.default_rng(7)
SHAPES = {
    "logo": Shape(shape_logo, 9.0),
    "knot": Shape(build_knot(g0), 17.0),
    "fish": Shape(build_fish(g0), 23.0),
    "helix": Shape(build_helix(g0), 29.0),
    "jellies": Shape(build_jellies(g0, JELLIES), 34.0),
    "swarm": Shape(build_jellies(g0, SWARM), 40.0),
    "space": Shape(build_space(g0), 49.0),
    "blackhole": Shape(build_blackhole(g0), 57.0),
    "galaxy": Shape(build_galaxy(g0), 64.5),
    "comet": Shape(build_comet(g0), 72.0),
}

PID = rng.integers(0, 10_000, N)
RND = rng.random(N)
JIT = rng.normal(size=(N, 3))
REST = SHAPES["logo"](9.0)[0]
YN = (REST[:, 1] - REST[:, 1].min()) / np.ptp(REST[:, 1])
XN = (REST[:, 0] - REST[:, 0].min()) / np.ptp(REST[:, 0])
START = REST.copy()
START[:, 1] = 1.9 + RND * 3.2
START[:, 2] = rng.normal(size=N) * 0.35
DELAY = ASSEMBLE + 2.6 * (1 - YN) + 0.9 * rng.random(N)
FLIGHT = 1.4
MORPH_DELAY = MORPH_SPREAD * (0.7 * np.arange(N) / N + 0.3 * RND)
DROP = DISSOLVE + 1.0 * XN + 0.8 * RND
MORPH_TOTAL = MORPH + MORPH_SPREAD


def morph_state(t):
    """which shape(s) are on screen and how far the current morph is"""
    i = max(k for k, (_, t0) in enumerate(SEQ) if t0 <= t)
    name, t0 = SEQ[i]
    if i > 0 and t < t0 + MORPH_TOTAL:
        return SEQ[i - 1][0], name, t0
    return None, name, t0


def transition_energy(t):
    a, _, t0 = morph_state(t)
    if a is None:
        return 0.0
    return math.sin(math.pi * min(1.0, (t - t0) / MORPH_TOTAL))


LIGHT_N = LIGHT / np.linalg.norm(LIGHT)
VIEW = np.array([0.0, 0.0, 1.0])
HALF = (LIGHT_N + VIEW) / np.linalg.norm(LIGHT_N + VIEW)


def scene(t):
    if t < ASSEMBLE - 0.1:
        return None
    a_name, b_name, t0 = morph_state(t)
    pos, nrm, col, glow = SHAPES[b_name](t)
    flying = np.zeros(N, bool)
    if a_name is not None:
        pa, na, ca, ga = SHAPES[a_name](t)
        e = sms(t0 + MORPH_DELAY, t0 + MORPH_DELAY + MORPH, t)
        arc = np.sin(math.pi * e)[:, None]
        d = pos - pa
        # each point flies a curved, swirling path instead of a straight line
        swirl = np.stack([-d[:, 2], 0.25 * JIT[:, 1], d[:, 0]], -1) * 0.38 + JIT * 0.07
        pos = pa * (1 - e[:, None]) + pos * e[:, None] + swirl * arc
        nrm = norm(na * (1 - e[:, None]) + nrm * e[:, None] + 1e-4)
        hue = np.where((RND < 0.5)[:, None], CYAN, FUCHSIA)
        col = ca * (1 - e[:, None]) + col * e[:, None]
        col = col * (1 - 0.3 * arc) + hue * 0.3 * arc
        glow = ga * (1 - e) + glow * e + 0.5 * arc[:, 0]
        flying = (e > 0.06) & (e < 0.94)

    # condense out of the rain
    if t < DELAY.max() + FLIGHT:
        e = ss(DELAY, DELAY + FLIGHT, t)
        pos = START * (1 - e[:, None]) + pos * e[:, None]
        pos[:, 0] += (1 - e) * 0.03 * np.sin(t * 6 + RND * 20)
        flying |= e < 0.97
        fade = np.where(t > DELAY - 0.05, 1.0, 0.0)
    else:
        fade = np.ones(N)

    # short glitch-slice burst as every morph kicks off
    if a_name is not None and t - t0 < 0.45:
        gg = np.random.default_rng(int(t * 20))
        band_id = np.floor((pos[:, 1] + 3) * 10).astype(int) % 60
        offs = gg.normal(size=60) * 0.12 * (gg.random(60) < 0.3)
        pos[:, 0] += offs[band_id]
        hit = offs[band_id] != 0
        col = np.where(hit[:, None], col * 0.3 + FUCHSIA, col)

    # final letter rain
    if t > DISSOLVE:
        tau = np.maximum(0.0, t - DROP)
        pos[:, 1] -= 1.8 * tau ** 2
        pos[:, 0] += JIT[:, 0] * 0.25 * tau
        pos[:, 2] += JIT[:, 2] * 0.4 * tau
        flying |= tau > 0
        fade = fade * (1 - ss(0.6, 1.6, tau))

    # laser scan on the logo
    scanned = np.zeros(N, bool)
    if SCAN[0] < t < SCAN[1]:
        u = (t - SCAN[0]) / (SCAN[1] - SCAN[0])
        d = np.abs(pos[:, 1] - (1.3 - 2.7 * (1 - abs(2 * u - 1))))
        hit = np.exp(-(d / 0.04) ** 2)
        scanned = d < 0.06
        glow = glow + 1.2 * hit
        col = col * (1 - hit[:, None]) + np.array([0.3, 1.0, 0.85]) * hit[:, None]

    # slow cinematic camera drift
    R = rot(0.1 * math.sin(t * 0.23), 0.05 * math.sin(t * 0.31), 0.0)
    pos = pos @ R.T * (1 + 0.04 * math.sin(t * 0.37))
    return pos, nrm @ R.T, col, glow, fade, flying, scanned


def rasterize(t):
    sc = scene(t)
    if sc is None:
        return None
    pos, nrm, col, glow, fade, flying, scanned = sc
    depth = pos[:, 2]
    nd = nrm[:, 2]
    nrm = np.where((nd < 0)[:, None], -nrm, nrm)
    nd = np.abs(nd)
    diff = np.maximum(nrm @ LIGHT_N, 0)
    spec = np.maximum(nrm @ HALF, 0) ** 22
    lum = (0.32 + 0.9 * diff + 0.3 * (1 - nd) ** 2) + 0.6 * glow
    dof = 0.5 + 0.5 * np.clip((depth + 1.2) / 2.0, 0, 1)
    rgb = (col * lum[:, None] * 1.12 + (0.45 * spec)[:, None]) * (dof * fade)[:, None]
    lumt = np.clip((lum * (0.35 + 0.65 * col.max(1)) + 0.45 * spec) * dof * fade, 0, 1)

    near = depth < CAM - 0.15
    k = CAM / np.maximum(CAM - depth, 0.1)
    cc = np.floor((W / 2 + pos[:, 0] * k * SCALE) / cw).astype(int)
    rr = np.floor((CY - pos[:, 1] * k * SCALE) / ch).astype(int)
    ok = near & (cc >= 0) & (cc < COLS) & (rr >= 0) & (rr < ROWS) & (fade > 0)
    idx = np.nonzero(ok)[0]
    ids = rr[idx] * COLS + cc[idx]
    order = np.argsort(-depth[idx], kind="stable")
    uid, first = np.unique(ids[order], return_index=True)
    win = idx[order[first]]

    n = ROWS * COLS
    Lg, Cg, Dg = np.zeros(n), np.zeros((n, 3)), np.full(n, -3.0)
    Fg, Sg, Pg = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, int)
    Lg[uid], Cg[uid], Dg[uid] = lumt[win], rgb[win], depth[win]
    Fg[uid], Sg[uid], Pg[uid] = flying[win], scanned[win], PID[win]
    sh = (ROWS, COLS)
    return (Lg.reshape(sh), Cg.reshape(sh + (3,)), Dg.reshape(sh), Fg.reshape(sh),
            Sg.reshape(sh), Pg.reshape(sh))


def edge_chars(filled, D):
    F = np.where(filled, D, -3.0)
    Fp = np.pad(F, 1, constant_values=-3.0)
    nbs = [Fp[1:-1, 2:], Fp[1:-1, :-2], Fp[2:, 1:-1], Fp[:-2, 1:-1]]
    jump = np.max([np.abs(F - nb) for nb in nbs], axis=0)
    edge = filled & (jump > 0.28) & ((jump > 1.8) | (F >= np.max(nbs, axis=0)))
    gx = (Fp[1:-1, 2:] - Fp[1:-1, :-2]) / cw
    gy = (Fp[2:, 1:-1] - Fp[:-2, 1:-1]) / ch
    a = np.degrees(np.arctan2(-gx, -gy)) % 180
    ech = np.select([(a < 22.5) | (a >= 157.5), a < 67.5, a < 112.5], ["-", "/", "|"], "\\")
    return edge, ech


# ------------------------------------------------------------------ background layers
RAIN_COLS = np.arange(0, COLS, 2)
RAIN_SPEED = 18 + rng.random(len(RAIN_COLS)) * 32
RAIN_OFF = rng.random(len(RAIN_COLS)) * ROWS * 1.5
RAIN_LEN = 16


def rain_layer(t, gen):
    env = bump(RAIN[0], RAIN[0] + 0.8, RAIN[1] - 1.4, RAIN[1], t)
    out = []
    if env <= 0:
        return out
    heads = ((t - RAIN[0]) * RAIN_SPEED + RAIN_OFF).astype(int) - 20
    for c, hd in zip(RAIN_COLS.tolist(), heads.tolist()):
        for j in range(RAIN_LEN):
            r = hd - j
            if 0 <= r < ROWS:
                f = (1 - j / RAIN_LEN) * env
                rgb = (int(40 * f), int(255 * f), int(230 * f)) if j else tuple(int(v * env) for v in (220, 255, 250))
                out.append((GLITCH[gen.integers(len(GLITCH))], c, r, rgb))
    return out


NBUB = 110
BUB_X = rng.uniform(0, W, NBUB)
BUB_SPD = rng.uniform(35, 120, NBUB)
BUB_OFF = rng.uniform(0, H + 60, NBUB)
BUB_PH = rng.uniform(0, 2 * math.pi, NBUB)
NSNOW = 160
SNOW = np.stack([rng.uniform(0, W, NSNOW), rng.uniform(0, H, NSNOW), rng.uniform(6, 18, NSNOW)], -1)
WEED_X = np.sort(rng.choice(np.arange(3, COLS - 3), 22, replace=False))
WEED_H = rng.integers(10, 30, 22)
WEED_PH = rng.uniform(0, 2 * math.pi, 22)
WEED_COL = [(BLUE, CYAN, VIOLET, FUCHSIA)[i % 4] for i in range(22)]
RAYS = [(0.12, 0.0), (0.33, 1.3), (0.55, 2.1), (0.74, 3.7), (0.92, 4.4)]
_sea = [0.0, 0.0]  # integrated travel (speeds up during morphs), last t


def sea_layer(t):
    """underwater backdrop: light shafts, rising bubbles, swaying neon seaweed, marine snow"""
    env = fs(RAIN[1] - 2.0, RAIN[1], t) * (1 - fs(SPACE, SPACE + 2.0, t))
    warp = transition_energy(t)
    dt = t - _sea[1]
    if 0 < dt < 0.5:
        _sea[0] += dt * (1 + 4 * warp)
    _sea[1] = t
    if env <= 0:
        return []
    out = []
    # light shafts from the surface
    for k, (x0, ph) in enumerate(RAYS):
        a = 0.5 + 0.5 * math.sin(t * 0.6 + ph)
        base = x0 * COLS + 6 * math.sin(t * 0.25 + ph)
        for r in range(0, int(ROWS * 0.8), 2):
            f = env * a * (1 - r / (ROWS * 0.8)) * 0.35
            if f < 0.03:
                continue
            for w in range(-3, 4):
                c = int(base + r * 0.45 + w)
                if 0 <= c < COLS and (r * 31 + c * 17 + int(t * 6)) % 4 == 0:
                    v = f * (1 - abs(w) / 4)
                    out.append(("/", c, r, (int(60 * v), int(150 * v), int(255 * v))))
    # marine snow drifting down
    for x, y, sp in SNOW.tolist():
        yy_ = (y + t * sp) % H
        xx_ = x + 6 * math.sin(t * 0.5 + y)
        out.append((".", int(xx_ // cw), int(yy_ // ch), (int(70 * env), int(90 * env), int(130 * env))))
    # bubbles
    trav = _sea[0]
    for i in range(NBUB):
        y = H - ((trav * BUB_SPD[i] + BUB_OFF[i]) % (H + 60)) + 30
        x = BUB_X[i] + 9 * math.sin(trav * 2.2 + BUB_PH[i])
        c, r = int(x // cw), int(y // ch)
        if 0 <= c < COLS and 0 <= r < ROWS:
            big = BUB_SPD[i] > 95
            v = env * (0.5 + 0.5 * (y / H))
            out.append(("O" if big else "o" if BUB_SPD[i] > 60 else ".", c, r,
                        (int(150 * v), int(230 * v), int(255 * v))))
    # neon seaweed rising from the floor
    for x0, hgt, ph, neon in zip(WEED_X.tolist(), WEED_H.tolist(), WEED_PH.tolist(), WEED_COL):
        prev = 0.0
        for j in range(hgt):
            q = j / hgt
            sw = 3.5 * q ** 1.5 * math.sin(t * 1.3 - j * 0.22 + ph)
            chh = "(" if sw > prev else ")"
            prev = sw
            c, r = int(x0 + sw), ROWS - 1 - j
            if 0 <= c < COLS:
                v = env * (0.25 + 0.45 * q)
                rgb = tuple(int(255 * v * x) for x in neon)
                out.append((chh, c, r, rgb))
    return out


NST = 420
ST_X = rng.uniform(0, W, NST)
ST_Y = rng.uniform(0, H, NST)
ST_B = rng.random(NST) ** 2
ST_PH = rng.uniform(0, 2 * math.pi, NST)
ST_COL = np.array([WHITE, WHITE, WHITE, CYAN * 0.6 + 0.4, FUCHSIA * 0.6 + 0.4, VIOLET * 0.6 + 0.4])[rng.integers(0, 6, NST)]


def space_layer(t):
    """deep-space backdrop: twinkling parallax stars and shooting stars"""
    env = fs(SPACE + 0.5, SPACE + 2.5, t) * (1 - fs(DISSOLVE + 1, DISSOLVE + 2.5, t))
    if env <= 0:
        return []
    out = []
    xs = (ST_X - (t - SPACE) * (3 + 10 * ST_B)) % W
    tw = 0.65 + 0.35 * np.sin(t * (2 + 3 * ST_B) + ST_PH)
    bs = env * (0.2 + 0.8 * ST_B) * tw
    for x, y, b0, b, c in zip(xs.tolist(), ST_Y.tolist(), ST_B.tolist(), bs.tolist(), ST_COL.tolist()):
        chh = "*" if b0 > 0.7 else "+" if b0 > 0.35 else "."
        out.append((chh, int(x // cw), int(y // ch), (int(255 * b * c[0]), int(255 * b * c[1]), int(255 * b * c[2]))))
    k = int(t / 2.4)
    s_ = (t % 2.4) / 0.6
    if s_ < 1:
        gk = np.random.default_rng(k)
        x0, y0 = gk.uniform(0.1, 0.8) * COLS, gk.uniform(0.05, 0.45) * ROWS
        hx, hy = x0 + s_ * 45, y0 + s_ * 13
        for i in range(14):
            c, r = int(hx - i * 1.7), int(hy - i * 0.5)
            if 0 <= c < COLS and 0 <= r < ROWS:
                v = env * (1 - i / 14) * (1 - s_ * 0.5)
                out.append(("*" if i == 0 else "-", c, r, (int(255 * v), int(220 * v), int(255 * v))))
    return out


# ------------------------------------------------------------------ text overlays
glyph_cache = {}


def glyph(c, rgb, f=font):
    key = (c, rgb, id(f))
    s = glyph_cache.get(key)
    if s is None:
        s = glyph_cache[key] = f.render(c, True, rgb)
    return s


def big_text(t, gen):
    out = []
    bw, bh = font_big.size("M")
    x0_ = W // 2 - bw * len(WATERMARK) // 2
    y0_ = int(CY - bh / 2)
    blink = (t * 2.2) % 1 < 0.55
    base = (220, 235, 255)
    if t < 0.4:
        if blink:
            out.append(("_", x0_, y0_, base))
    elif t < 2.0:
        n = min(len(WATERMARK), int((t - 0.4) / 1.3 * len(WATERMARK)) + 1)
        for i, ch_ in enumerate(WATERMARK[:n]):
            if t - 0.4 - i * 1.3 / len(WATERMARK) < 0.18:
                ch_ = GLITCH[gen.integers(len(GLITCH))]
            out.append((ch_, x0_ + i * bw, y0_, base))
        out.append(("_", x0_ + n * bw, y0_, base))
    elif t < 2.8:
        for i, ch_ in enumerate(WATERMARK + ("_" if blink else "")):
            pulse = 0.75 + 0.25 * math.sin(t * 12 + i)
            out.append((ch_, x0_ + i * bw, y0_, (int(80 * pulse), int(230 * pulse), 255)))
    elif t < WM_END:
        f = 1 - (t - 2.8) / (WM_END - 2.8)
        for i, ch_ in enumerate(WATERMARK):
            if gen.random() < 0.6:
                ch_ = GLITCH[gen.integers(len(GLITCH))]
            if gen.random() < f:
                v = int(255 * f)
                dx = int(gen.normal() * 8 * (1 - f))
                out.append((ch_, x0_ + i * bw + dx, y0_, (int(v * 0.4), int(v * 0.9), v)))
    elif t > DISSOLVE + 1.2:
        a = fs(DISSOLVE + 1.2, DISSOLVE + 1.8, t)
        n = min(len(WATERMARK), int((t - DISSOLVE - 1.2) / 0.8 * len(WATERMARK)) + 1)
        for i, ch_ in enumerate(WATERMARK[:n] + ("_" if blink else "")):
            out.append((ch_, x0_ + i * bw, y0_, (int(220 * a), int(235 * a), int(255 * a))))
    return out


def signature():
    sw, shh = font_small.size(WATERMARK)
    return [(glyph(WATERMARK, (70, 90, 110), font_small), (W - sw - 12, H - shh - 8))]


def main():
    gen = np.random.default_rng(5)
    first_on = np.full((ROWS, COLS), np.inf)
    start = time.perf_counter()
    frame, last_t = 0, 0.0
    clock = pygame.time.Clock()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                start = time.perf_counter()
        t = frame / FPS_RECORD if RECORD else (time.perf_counter() - start) % DURATION
        if t < last_t:
            first_on[:] = np.inf
        last_t = t

        screen.fill(BG)
        blits = []
        for c, cc, rr, rgb in space_layer(t) + sea_layer(t) + rain_layer(t, gen):
            blits.append((glyph(c, rgb), (cc * cw, rr * ch)))
        res = rasterize(t)
        if res is not None:
            Lg, Cg, Dg, Fg, Sg, Pg = res
            filled = Lg > 0.03
            edge, ech = edge_chars(filled, Dg)
            edge &= ~Fg
            chars = RAMP_ARR[np.clip((Lg * (len(RAMP) - 1)).astype(int), 0, len(RAMP) - 1)]
            chars = np.where(Lg > 0.82, MSG_GRID, chars)
            chars = np.where(edge, ech, chars)
            chars = np.where(Sg, BIN_GRID, chars)
            chars = np.where(Fg, LETTERS[Pg % len(LETTERS)], chars)
            colr = np.where(edge[..., None], Cg * 1.25 + np.array([0.06, 0.12, 0.22]), Cg)
            first_on[filled & np.isinf(first_on)] = t
            first_on[~filled] = np.inf
            glitching = filled & ~Fg & ((t - first_on) < SETTLE) & (t < 12)
            colr = np.where(glitching[..., None], colr * 0.5 + 0.45, colr)
            colr = (np.clip(colr, 0, 1) * 255 // 10 * 10).astype(int)
            rs, cs_ = np.nonzero(filled)
            gl = glitching[rs, cs_].tolist()
            chs = chars[rs, cs_].tolist()
            cols_ = colr[rs, cs_].tolist()
            for r_, c_, g_, chh, rgb in zip(rs.tolist(), cs_.tolist(), gl, chs, cols_):
                if g_:
                    chh = GLITCH[gen.integers(len(GLITCH))]
                if chh == " " or (rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 0):
                    continue
                blits.append((glyph(chh, tuple(rgb)), (c_ * cw, r_ * ch)))
        for c, x, y, rgb in big_text(t, gen):
            blits.append((glyph(c, rgb, font_big), (x, y)))
        if t > WM_END:
            blits += signature()
        screen.blits(blits, doreturn=False)

        for div in (5, 18):
            small = pygame.transform.smoothscale(screen, (W // div, H // div))
            screen.blit(pygame.transform.smoothscale(small, (W, H)), (0, 0),
                        special_flags=pygame.BLEND_ADD)
        pygame.display.flip()

        if RECORD:
            pygame.image.save(screen, f"{OUT_DIR}/f_{frame:04d}.png")
            if frame + 1 >= int(DURATION * FPS_RECORD):
                pygame.quit()
                return
        else:
            clock.tick(60)
        frame += 1


if __name__ == "__main__":
    main()
