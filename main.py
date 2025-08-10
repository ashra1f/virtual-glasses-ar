import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import math
from pathlib import Path

# -------------------------
# Utilitaires overlay PNG
# -------------------------
def load_glass_images(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)  # BGRA
        if img is None:
            continue
        imgs.append((Path(p).name, img))
    return imgs

def rotate_image_with_alpha(img_bgra, angle_deg):
    (h, w) = img_bgra.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    # compute new bounding dimensions
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to consider translation
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    rotated = cv2.warpAffine(img_bgra, M, (nW, nH), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def overlay_png(bg_bgr, fg_bgra, x, y):
    """Overlay fg_bgra onto bg_bgr with top-left corner at (x,y). Handles clipping."""
    bh, bw = bg_bgr.shape[:2]
    fh, fw = fg_bgra.shape[:2]
    if x >= bw or y >= bh or x + fw <= 0 or y + fh <= 0:
        return bg_bgr  # completely outside
    # clipping
    x1 = max(x, 0); y1 = max(y, 0)
    x2 = min(x + fw, bw); y2 = min(y + fh, bh)
    fx1 = x1 - x; fy1 = y1 - y
    fx2 = fx1 + (x2 - x1); fy2 = fy1 + (y2 - y1)
    # split fg channels
    fg = fg_bgra[fy1:fy2, fx1:fx2, :3].astype(float)
    alpha = fg_bgra[fy1:fy2, fx1:fx2, 3].astype(float) / 255.0
    alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    roi = bg_bgr[y1:y2, x1:x2].astype(float)
    out = (alpha * fg) + ((1 - alpha) * roi)
    bg_bgr[y1:y2, x1:x2] = out.astype(np.uint8)
    return bg_bgr

# -------------------------
# Face mesh + main loop
# -------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             max_num_faces=2,
                             refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# landmarks used for eyes outer corners (MediaPipe indices)
# left eye outer: 33, right eye outer: 263
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# smoothing state (exponential moving average)
class SmoothState:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.x = None
        self.y = None
        self.angle = None
        self.scale = None

    def update(self, x, y, angle, scale):
        if self.x is None:
            self.x, self.y, self.angle, self.scale = x, y, angle, scale
        else:
            a = self.alpha
            # angle smoothing: handle wrap-around
            # convert angles to complex representation for smooth interpolation
            ca = complex(math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle)))
            na = complex(math.cos(math.radians(angle)), math.sin(math.radians(angle)))
            inter = (a * na) + ((1-a) * ca)
            interp_angle = math.degrees(math.atan2(inter.imag, inter.real))
            self.x = a * x + (1 - a) * self.x
            self.y = a * y + (1 - a) * self.y
            self.angle = interp_angle
            self.scale = a * scale + (1 - a) * self.scale
        return self.x, self.y, self.angle, self.scale

# Load glasses
glasses_dir = "glasses"  # change if needed
glasses = load_glass_images(glasses_dir)
if not glasses:
    raise SystemExit(f"No PNG found in folder '{glasses_dir}'. Put some transparent PNGs there.")

idx = 0
state = SmoothState(alpha=0.6)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

print("Controls: [n] next glasses, [p] prev, [s] save photo, [q/ESC] quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # get eye outer corners
            lm = face_landmarks.landmark
            left = lm[LEFT_EYE_OUTER]
            right = lm[RIGHT_EYE_OUTER]
            lx, ly = int(left.x * w), int(left.y * h)
            rx, ry = int(right.x * w), int(right.y * h)

            # center between eyes
            cx = (lx + rx) // 2
            cy = (ly + ry) // 2

            # distance between eyes
            eye_dist = math.hypot(rx - lx, ry - ly)

            # angle between eyes
            angle_rad = math.atan2(ry - ly, rx - lx)
            angle_deg = math.degrees(angle_rad)

            # estimate scale (glass image width relative to eye distance)
            target_glass_width = int(eye_dist * 2.4)  # tweak factor for coverage

            # smoothing
            sx, sy, sangle, sscale = state.update(cx, cy, angle_deg, target_glass_width)

            # prepare glass image
            name, gimg = glasses[idx]
            # rotate
            rotated = rotate_image_with_alpha(gimg, sangle)
            # scale to target width
            gh, gw = rotated.shape[:2]
            if gw == 0 or gh == 0:
                continue
            scale_factor = sscale / gw
            new_w = max(1, int(gw * scale_factor))
            new_h = max(1, int(gh * scale_factor))
            resized = cv2.resize(rotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # top-left coordinate to place (centered at sx, sy)
            top_left_x = int(sx - new_w // 2)
            top_left_y = int(sy - new_h // 2) + int(eye_dist * 0.12)  # tweak vertical offset

            # overlay
            overlayed = overlay_png(frame, resized, top_left_x, top_left_y)
            frame = overlayed

            # debug: show landmarks and bbox
            cv2.circle(frame, (lx, ly), 2, (0,255,0), -1)
            cv2.circle(frame, (rx, ry), 2, (0,255,0), -1)
            cv2.putText(frame, f"{name}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    else:
        # slowly decay smoothing if face lost
        pass

    cv2.imshow("Virtual Glasses AR", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('n'):
        idx = (idx + 1) % len(glasses)
    elif key == ord('p'):
        idx = (idx - 1) % len(glasses)
    elif key == ord('s'):
        # save snapshot
        out_name = f"snap_{idx}_{np.random.randint(1_000_000)}.png"
        cv2.imwrite(out_name, frame)
        print("Saved", out_name)

cap.release()
cv2.destroyAllWindows()
