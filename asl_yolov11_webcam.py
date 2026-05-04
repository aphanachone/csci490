"""
ASL Sign Detection - YOLOv11 webcam app.

Run:
    python asl_yolo_webcam.py
    Ensure best.pt is in the same folder
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import tkinter as tk
from collections import Counter, deque
from pathlib import Path
from tkinter import messagebox, ttk

import cv2

# Fail if Ultralytics isn't installed
try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'ultralytics'. Install with:\n"
        "    pip install ultralytics opencv-python"
    ) from exc

# Finds best.pt in same folder
DEFAULT_WEIGHT_CANDIDATES = [
    Path("best.pt"),
]

# Path for persistent user preferences, kept next to script
SETTINGS_PATH = Path(__file__).parent / "settings.json"

# Default settings used when no settings.json exists yet
DEFAULTS = {
    "camera": 0, #0 is default camera index for laptops; 1, 2, 3 are external
    "conf": 0.50,
    "mirror": False,
    "show_fps": True,
    "save_on_exit": True,
}

# Fixed inference values not exposed in settings UI.
IOU_THRESHOLD = 0.45      # NMS overlap threshold for duplicate boxes
MAX_DETECTIONS = 2        # cap on detections per frame (two hands)
SMOOTH_WINDOW = 7         # rolling window size for stable-label vote
MIN_STABLE_VOTES = 4      # votes needed within window to declare a stable label
INFERENCE_IMGSZ = 640     # input resolution YOLO resizes frames to
WARMUP_FRAMES = 30        # frames discarded so camera sensor stabilizes

# Tkinter palette
UI_BG          = "#1e1e1e"   # main background
UI_BG_RAISED   = "#2a2a2a"   # raised panels (info card, dialog body)
UI_BG_INPUT    = "#333333"   # form field background
UI_BORDER      = "#3a3a3a"   # subtle dividers
UI_TEXT        = "#ffffff"   # primary text
UI_TEXT_MUTED  = "#a0a0a0"   # secondary text
UI_TEXT_FAINT  = "#707070"   # tertiary text (class preview, hints)

UI_PRIMARY     = "#4CAF50"   # primary action (green)
UI_PRIMARY_HOV = "#5BC25F"
UI_DANGER      = "#d32f2f"   # warning action (red)
UI_DANGER_HOV  = "#e53935"
UI_NEUTRAL     = "#4a4a4a"   # neutral action (gray)
UI_NEUTRAL_HOV = "#5a5a5a"

# Typography
FONT_FAMILY = "Segoe UI"
FONT_TITLE   = (FONT_FAMILY, 22, "bold")
FONT_SUB     = (FONT_FAMILY, 11)
FONT_BODY    = (FONT_FAMILY, 10)
FONT_LABEL   = (FONT_FAMILY, 10)
FONT_BTN     = (FONT_FAMILY, 11, "bold")
FONT_BTN_SM  = (FONT_FAMILY, 10)
FONT_VALUE   = (FONT_FAMILY, 10, "bold")

# Layout spacing
PAD_XL = 28
PAD_LG = 18
PAD_MD = 12
PAD_SM = 8
PAD_XS = 4
BTN_WIDTH = 22
BTN_HEIGHT = 2
WINDOW_W = 480
WINDOW_H = 540
DIALOG_W = 460
DIALOG_H = 460

# OpenCV overlay colors (BGR order)
BOX_COLOR    = (54, 205, 88)     # detection box
TEXT_COLOR   = (255, 255, 255)
TEXT_SHADOW  = (0, 0, 0)
STABLE_COLOR = (0, 200, 255)
DIM_COLOR    = (180, 180, 180)
QUIT_FILL    = (47, 47, 211)     # red
QUIT_HOVER   = (53, 57, 229)     # red hover

# === Reusable Tkinter widgets ==============================================
# Helper functions that wrap tk.Button with established colors above

def primary_button(parent, text: str, command) -> tk.Button:
    # Green primary action button, used for main affirmative action
    return tk.Button(
        parent, text=text, command=command,
        font=FONT_BTN, width=BTN_WIDTH, height=BTN_HEIGHT,
        bg=UI_PRIMARY, fg=UI_TEXT,
        activebackground=UI_PRIMARY_HOV, activeforeground=UI_TEXT,
        relief="flat", cursor="hand2", borderwidth=0,
    )


def neutral_button(parent, text: str, command,
                   small: bool = False) -> tk.Button:
    # Gray secondary action button
    return tk.Button(
        parent, text=text, command=command,
        font=FONT_BTN_SM if small else FONT_BTN,
        width=14 if small else BTN_WIDTH,
        height=1 if small else BTN_HEIGHT,
        bg=UI_NEUTRAL, fg=UI_TEXT,
        activebackground=UI_NEUTRAL_HOV, activeforeground=UI_TEXT,
        relief="flat", cursor="hand2", borderwidth=0,
    )


def danger_button(parent, text: str, command,
                  small: bool = False) -> tk.Button:
    # Red destructive action button for Quit
    return tk.Button(
        parent, text=text, command=command,
        font=FONT_BTN_SM if small else FONT_BTN,
        width=14 if small else BTN_WIDTH,
        height=1 if small else BTN_HEIGHT,
        bg=UI_DANGER, fg=UI_TEXT,
        activebackground=UI_DANGER_HOV, activeforeground=UI_TEXT,
        relief="flat", cursor="hand2", borderwidth=0,
    )


def field_label(parent, text: str) -> tk.Label:
    # Muted label that sits above a form field
    return tk.Label(
        parent, text=text, font=FONT_LABEL,
        bg=UI_BG_RAISED, fg=UI_TEXT_MUTED, anchor="w",
    )


def section_divider(parent) -> tk.Frame:
    # Thin horizontal line that separates groups of controls
    return tk.Frame(parent, bg=UI_BORDER, height=1)


def configure_ttk_styles(root: tk.Misc) -> None:
    # Apply dark theme to ttk widgets used in dialog
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        "Dark.TCombobox",
        fieldbackground=UI_BG_INPUT, background=UI_BG_INPUT,
        foreground=UI_TEXT, arrowcolor=UI_TEXT,
        bordercolor=UI_BORDER, lightcolor=UI_BG_INPUT, darkcolor=UI_BG_INPUT,
        selectbackground=UI_BG_INPUT, selectforeground=UI_TEXT,
    )
    style.map(
        "Dark.TCombobox",
        fieldbackground=[("readonly", UI_BG_INPUT)],
        selectbackground=[("readonly", UI_BG_INPUT)],
    )
    style.configure(
        "Dark.Horizontal.TScale",
        background=UI_BG_RAISED, troughcolor=UI_BG_INPUT,
        bordercolor=UI_BG_RAISED, lightcolor=UI_PRIMARY, darkcolor=UI_PRIMARY,
    )


# === Settings persistence ==================================================
def load_settings() -> dict:
    # Read settings.json if present, fall back to defaults on missing or corrupt file
    settings = DEFAULTS.copy()
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH) as f:
                saved = json.load(f)
            # Only copy keys we know about, to ignore stale keys from older versions
            for key in DEFAULTS:
                if key in saved:
                    settings[key] = saved[key]
        except (json.JSONDecodeError, OSError):
            pass
    return settings


def save_settings(settings: dict) -> None:
    # Write settings for next launch
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)
    except OSError:
        pass


# === Weight finding ========================================================
def find_default_weights() -> Path | None:
    # Search standard locations for trained model file
    for candidate in DEFAULT_WEIGHT_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    return None


# === Tkinter: settings dialog ==============================================
class SettingsDialog(tk.Toplevel):
    """Modal settings window that updates passed-in settings dict on Save."""

    def __init__(self, parent: tk.Tk, settings: dict):
        super().__init__(parent)
        self.title("Settings")
        self.resizable(False, False)
        self.configure(bg=UI_BG)

        # Makes settings modal child of parent window so it blocks interactions until dismissed
        self.transient(parent)
        self.grab_set()

        # Center dialog over parent window
        self.update_idletasks()
        px = parent.winfo_x() + (parent.winfo_width() - DIALOG_W) // 2
        py = parent.winfo_y() + (parent.winfo_height() - DIALOG_H) // 2
        self.geometry(f"{DIALOG_W}x{DIALOG_H}+{px}+{py}")

        self.settings = settings
        configure_ttk_styles(self)
        self._build_ui()

        # Treat OS close button (X) same as Cancel
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.wait_window(self)

    def _build_ui(self) -> None:
        # Header bar with dialog title
        header = tk.Frame(self, bg=UI_BG, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header, text="Settings", font=FONT_TITLE,
            bg=UI_BG, fg=UI_TEXT,
        ).pack(pady=PAD_LG)

        # Body card holds all form controls on a slightly raised background
        body = tk.Frame(self, bg=UI_BG_RAISED)
        body.pack(fill="both", expand=True, padx=PAD_XL, pady=(0, PAD_LG))

        inner = tk.Frame(body, bg=UI_BG_RAISED)
        inner.pack(fill="both", expand=True, padx=PAD_LG, pady=PAD_LG)

        # Camera index field - dropdown of likely camera indices
        field_label(inner, "Camera index").pack(anchor="w", pady=(0, PAD_XS))
        self.cam_var = tk.StringVar(value=str(self.settings["camera"]))
        cam_box = ttk.Combobox(
            inner, textvariable=self.cam_var,
            values=["0", "1", "2", "3"],
            state="readonly", width=10, style="Dark.TCombobox",
        )
        cam_box.pack(anchor="w", pady=(0, PAD_MD))

        # Confidence threshold field with a live readout
        conf_header = tk.Frame(inner, bg=UI_BG_RAISED)
        conf_header.pack(fill="x", pady=(0, PAD_XS))
        field_label(conf_header, "Confidence threshold").pack(side="left")
        self.conf_var = tk.DoubleVar(value=self.settings["conf"])
        self.conf_label = tk.Label(
            conf_header, text=f"{self.conf_var.get():.2f}",
            font=FONT_VALUE, bg=UI_BG_RAISED, fg=UI_PRIMARY, width=5,
        )
        self.conf_label.pack(side="right")

        ttk.Scale(
            inner, from_=0.10, to=0.95, orient="horizontal",
            variable=self.conf_var, command=self._update_conf_label,
            style="Dark.Horizontal.TScale",
        ).pack(fill="x", pady=(0, PAD_MD))

        # Divider between group and toggles
        section_divider(inner).pack(fill="x", pady=PAD_SM)

        # Boolean toggles for on/off preferences
        self.mirror_var = tk.BooleanVar(value=self.settings["mirror"])
        self.fps_var = tk.BooleanVar(value=self.settings["show_fps"])
        self.save_var = tk.BooleanVar(value=self.settings["save_on_exit"])

        self._checkbox(inner, "Mirror view", self.mirror_var)
        self._checkbox(inner, "Show FPS overlay", self.fps_var)
        self._checkbox(inner, "Save preferences between launches", self.save_var)

        # Footer row with buttons
        footer = tk.Frame(self, bg=UI_BG)
        footer.pack(side="bottom", fill="x", pady=(0, PAD_LG))

        btn_row = tk.Frame(footer, bg=UI_BG)
        btn_row.pack()

        neutral_button(btn_row, "Reset", self._reset, small=True).pack(
            side="left", padx=PAD_XS)
        neutral_button(btn_row, "Cancel", self._on_cancel, small=True).pack(
            side="left", padx=PAD_XS)

        # Save button
        tk.Button(
            btn_row, text="Save", command=self._on_save,
            font=FONT_BTN_SM, width=14, height=1,
            bg=UI_PRIMARY, fg=UI_TEXT,
            activebackground=UI_PRIMARY_HOV, activeforeground=UI_TEXT,
            relief="flat", cursor="hand2", borderwidth=0,
        ).pack(side="left", padx=PAD_XS)

    def _checkbox(self, parent, text: str, var: tk.BooleanVar) -> None:
        # Themed checkbox row that matches dark dialog body
        cb = tk.Checkbutton(
            parent, text=text, variable=var,
            font=FONT_BODY,
            bg=UI_BG_RAISED, fg=UI_TEXT, selectcolor=UI_BG_INPUT,
            activebackground=UI_BG_RAISED, activeforeground=UI_TEXT,
            highlightthickness=0, borderwidth=0, anchor="w",
        )
        cb.pack(fill="x", pady=2)

    def _update_conf_label(self, _value: str) -> None:
        # Live update confidence number as slider moves
        self.conf_label.config(text=f"{self.conf_var.get():.2f}")

    def _reset(self) -> None:
        # Restore default values without closing dialog
        self.cam_var.set(str(DEFAULTS["camera"]))
        self.conf_var.set(DEFAULTS["conf"])
        self.conf_label.config(text=f"{DEFAULTS['conf']:.2f}")
        self.mirror_var.set(DEFAULTS["mirror"])
        self.fps_var.set(DEFAULTS["show_fps"])
        self.save_var.set(DEFAULTS["save_on_exit"])

    def _on_save(self) -> None:
        # Commit dialog's values back into shared settings dict
        self.settings["camera"] = int(self.cam_var.get())
        self.settings["conf"] = round(self.conf_var.get(), 2)
        self.settings["mirror"] = bool(self.mirror_var.get())
        self.settings["show_fps"] = bool(self.fps_var.get())
        self.settings["save_on_exit"] = bool(self.save_var.get())

        # Save to disk immediately if user opted in
        if self.settings["save_on_exit"]:
            save_settings(self.settings)
        self.destroy()

    def _on_cancel(self) -> None:
        # Close dialog without applying any changes
        self.destroy()


# === Tkinter: start screen =================================================
def show_start_screen(weights_path: Path, num_classes: int,
                      class_preview: str, settings: dict) -> str:
    """Show the start screen until user picks Begin Test or Quit.

    Returns 'start' to begin a detection session, or 'quit' to exit app.
    """
    root = tk.Tk()
    root.title("ASL Sign Detection")
    root.resizable(False, False)
    root.configure(bg=UI_BG)
    configure_ttk_styles(root)

    # Center window on user's screen
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(
        f"{WINDOW_W}x{WINDOW_H}+{(sw - WINDOW_W) // 2}+{(sh - WINDOW_H) // 2}")

    # Mutable container so nested callbacks can write choice
    choice = {"value": "quit"}

    def on_start():
        choice["value"] = "start"
        root.destroy()

    def on_settings():
        SettingsDialog(root, settings)

    def on_quit():
        choice["value"] = "quit"
        root.destroy()

    # Header section with title and subtitle
    header = tk.Frame(root, bg=UI_BG)
    header.pack(fill="x", pady=(PAD_XL + 12, 0))
    tk.Label(
        header, text="ASL Sign Detection",
        font=FONT_TITLE, bg=UI_BG, fg=UI_TEXT,
    ).pack()
    tk.Label(
        header, text="YOLOv11  -  Letters A-Z",
        font=FONT_SUB, bg=UI_BG, fg=UI_TEXT_MUTED,
    ).pack(pady=(PAD_XS, 0))

    # Info card showing which model file and which classes loaded
    card_outer = tk.Frame(root, bg=UI_BG)
    card_outer.pack(fill="x", padx=PAD_XL, pady=PAD_LG)
    card = tk.Frame(card_outer, bg=UI_BG_RAISED)
    card.pack(fill="x")
    card_inner = tk.Frame(card, bg=UI_BG_RAISED)
    card_inner.pack(fill="x", padx=PAD_LG, pady=PAD_MD)

    tk.Label(
        card_inner, text="MODEL", font=(FONT_FAMILY, 8, "bold"),
        bg=UI_BG_RAISED, fg=UI_TEXT_FAINT,
    ).pack(anchor="w")
    tk.Label(
        card_inner, text=weights_path.name, font=FONT_BODY,
        bg=UI_BG_RAISED, fg=UI_TEXT,
    ).pack(anchor="w", pady=(0, PAD_SM))

    tk.Label(
        card_inner, text=f"CLASSES  ({num_classes})",
        font=(FONT_FAMILY, 8, "bold"),
        bg=UI_BG_RAISED, fg=UI_TEXT_FAINT,
    ).pack(anchor="w")
    tk.Label(
        card_inner, text=class_preview, font=FONT_BODY,
        bg=UI_BG_RAISED, fg=UI_TEXT_MUTED,
        wraplength=WINDOW_W - 2 * PAD_XL - 2 * PAD_LG, justify="left",
    ).pack(anchor="w")

    # Stack of three action buttons - main entry point for app
    btns = tk.Frame(root, bg=UI_BG)
    btns.pack(pady=(0, PAD_LG))

    primary_button(btns, "Begin Test", on_start).pack(pady=PAD_XS)
    neutral_button(btns, "Settings", on_settings).pack(pady=PAD_XS)
    danger_button(btns, "Quit", on_quit).pack(pady=PAD_XS)

    # Treat OS close button (X) same as Quit
    root.protocol("WM_DELETE_WINDOW", on_quit)
    root.mainloop()
    return choice["value"]


# === Detection helpers =====================================================
def open_camera(index: int) -> cv2.VideoCapture:
    # On Windows, prefer DirectShow over default Media Foundation backend
    if platform.system() == "Windows" and hasattr(cv2, "CAP_DSHOW"):
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def warm_up(cap: cv2.VideoCapture, frames: int) -> None:
    # Discard first few frames while camera sensor adjusts auto-exposure
    for _ in range(frames):
        cap.read()


def draw_text(frame, text, org, scale=0.7, color=TEXT_COLOR):
    # Draw text twice: once thick in shadow color, once thin in main color
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_DUPLEX, scale,
                TEXT_SHADOW, max(2, int(scale * 4)), cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_DUPLEX, scale,
                color, max(1, int(scale * 2)), cv2.LINE_AA)


def draw_detection(frame, xyxy, label: str, conf: float) -> None:
    # Draw bounding box and a filled label pill above it
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

    # Compute label size so we can size pill to fit
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
    pad = 4

    # Place pill above box, but flip it below if it would clip off-screen
    by1, by2 = y1 - th - 2 * pad, y1
    if by1 < 0:
        by1, by2 = y1, y1 + th + 2 * pad

    cv2.rectangle(frame, (x1, by1), (x1 + tw + 2 * pad, by2), BOX_COLOR, -1)
    cv2.putText(frame, text, (x1 + pad, by2 - pad),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


def draw_quit_button(frame, state: dict) -> None:
    # Draw QUIT pill in top-right corner.
    h, w = frame.shape[:2]
    bw, bh, margin = 160, 56, 16
    x1, y1 = w - bw - margin, margin
    x2, y2 = x1 + bw, y1 + bh

    # Cache button rect so mouse callback can do hit-testing
    state["quit_rect"] = (x1, y1, x2, y2)

    fill = QUIT_HOVER if state.get("hover") else QUIT_FILL
    cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)

    # Center label text inside pill
    label = "QUIT"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    cv2.putText(frame, label,
                (x1 + (bw - tw) // 2, y1 + (bh + th) // 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


class StableLabeler:

    def __init__(self, window: int, min_votes: int):
        self.window = max(1, window)
        self.min_votes = max(1, min_votes)
        self.history: deque[str | None] = deque(maxlen=self.window)

    def update(self, label: str | None) -> tuple[str | None, int]:
        # Add new prediction to rolling window and find top vote
        self.history.append(label)
        labels = [l for l in self.history if l]
        if not labels:
            return None, 0
        top, votes = Counter(labels).most_common(1)[0]
        # Only return a stable label if it has enough support
        return (top, votes) if votes >= self.min_votes else (None, votes)

    def __len__(self) -> int:
        return len(self.history)


# === Detection loop ========================================================
def run_detection(model, names: dict, settings: dict) -> None:
    # Open camera and bail out if index is wrong
    cap = open_camera(settings["camera"])
    if not cap.isOpened():
        messagebox.showerror(
            "Camera error",
            f"Could not open camera index {settings['camera']}.\n"
            f"Try a different camera in Settings (1, 2, or 3).",
        )
        return

    # Reduce internal buffering so displayed frame is closer to "now"
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"Warming up camera ({WARMUP_FRAMES} frames)...", end="", flush=True)
    warm_up(cap, WARMUP_FRAMES)
    print(" ready!")

    # Shared mutable state that mouse callback can write to
    state = {"quit_clicked": False, "quit_rect": None, "hover": False}

    def on_mouse(event, x, y, flags, param):
        # Hit-test cursor against QUIT button rect.
        # Sets hover for visual feedback and quit_clicked when actually clicked.
        rect = state["quit_rect"]
        inside = False
        if rect is not None:
            x1, y1, x2, y2 = rect
            inside = x1 <= x <= x2 and y1 <= y <= y2
            state["hover"] = inside
        if inside and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP):
            state["quit_clicked"] = True

    window_name = "ASL Sign Detection - YOLOv11  (Q=quit  S=save)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    smoother = StableLabeler(SMOOTH_WINDOW, MIN_STABLE_VOTES)
    fps_hist: deque[float] = deque(maxlen=30)
    consecutive_failures = 0

    # Snapshot settings at session start
    conf = settings["conf"]
    mirror = settings["mirror"]
    show_fps = settings["show_fps"]

    try:
        while True:
            t0 = time.perf_counter()

            # Read next frame retry briefly on transient failures
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print("Too many consecutive frame failures, aborting session.")
                    break
                time.sleep(0.05)
                continue
            consecutive_failures = 0

            if mirror:
                frame = cv2.flip(frame, 1)

            # Run YOLO inference on current frame
            result = model.predict(
                source=frame,
                conf=conf,
                iou=IOU_THRESHOLD,
                imgsz=INFERENCE_IMGSZ,
                max_det=MAX_DETECTIONS,
                verbose=False,
            )[0]

            annotated = frame.copy()
            primary_label, primary_conf = None, 0.0

            # Draw each detection and remember highest-confidence one
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.cpu()
                xyxy = boxes.xyxy.numpy()
                confs = boxes.conf.numpy()
                cls_ids = boxes.cls.numpy().astype(int)
                for coords, c, cls_id in zip(xyxy, confs, cls_ids):
                    label = names.get(int(cls_id), str(cls_id))
                    draw_detection(annotated, coords, label, float(c))
                    if c > primary_conf:
                        primary_conf, primary_label = float(c), label

            stable_sign, votes = smoother.update(primary_label)

            # HUD line 1: raw per-frame prediction
            if primary_label:
                draw_text(annotated,
                          f"Instant: {primary_label}  ({primary_conf:.2f})",
                          (16, 34), 0.8)
            else:
                draw_text(annotated, "Instant: --", (16, 34), 0.8, DIM_COLOR)

            # HUD line 2: smoothed stable label
            if stable_sign:
                draw_text(annotated,
                          f"Stable:  {stable_sign}  [{votes}/{len(smoother)}]",
                          (16, 72), 1.0, STABLE_COLOR)
            else:
                draw_text(annotated,
                          f"Stable:  ...  [{votes}/{len(smoother)}]",
                          (16, 72), 1.0, DIM_COLOR)

            # HUD footer: FPS readout plus current setting values
            fps_hist.append(1.0 / max(time.perf_counter() - t0, 1e-6))
            if show_fps:
                fps = sum(fps_hist) / len(fps_hist)
                draw_text(annotated,
                          f"FPS: {fps:.1f}   conf>={conf:.2f}   "
                          f"mirror: {'ON' if mirror else 'OFF'}",
                          (16, annotated.shape[0] - 16), 0.6, DIM_COLOR)

            # Draw QUIT button last so detection boxes never overlap it
            draw_quit_button(annotated, state)
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF

            # Multiple ways to exit session: clicking QUIT, closing window via OS, or pressing Q
            if state["quit_clicked"]:
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if key == ord("q"):
                break
            elif key == ord("s"):
                # Save current annotated frame to ./captures/ for screenshots
                out_dir = Path("captures")
                out_dir.mkdir(exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                out = out_dir / f"asl_{stamp}.jpg"
                cv2.imwrite(str(out), annotated)
                print(f"Saved: {out}")
    finally:
        # Always release camera and destroy windows even if an exception
        # bubbled up. Without this camera stays locked on Windows.
        cap.release()
        cv2.destroyAllWindows()
        # Pump waitKey a few times so window actually closes on Windows
        for _ in range(4):
            cv2.waitKey(1)
        print("Session ended.")


# === Entry point ===========================================================
def parse_args() -> argparse.Namespace:
    # Minimal CLI for development overrides; end users should use Settings instead
    p = argparse.ArgumentParser(
        description="Live ASL sign detection - YOLOv11")
    p.add_argument("--weights", default="",
                   help="Path to trained YOLO weights (best.pt)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve weights path before showing UI
    if args.weights:
        weights = Path(args.weights)
    else:
        weights = find_default_weights()
        if weights is None:
            raise SystemExit(
                "Could not find a trained model. Looked in:\n  - "
                + "\n  - ".join(str(p) for p in DEFAULT_WEIGHT_CANDIDATES)
                + "\n\nDrop best.pt next to the script, or pass --weights <path>."
            )
    if not weights.exists():
        raise SystemExit(f"Weights file not found: {weights}")

    print(f"Loading YOLO weights: {weights}")
    model = YOLO(str(weights))

    # Normalize class-name container into a dict
    # Different Ultralytics versions return either a list or a dict here.
    names = model.names
    if isinstance(names, (list, tuple)):
        names = {i: n for i, n in enumerate(names)}

    # Build a class preview string for start screen info card
    class_list = [names[i] for i in sorted(names)]
    preview = ", ".join(class_list[:10]) + (" ..." if len(class_list) > 10 else "")
    print(f"Loaded {len(names)} classes: {', '.join(class_list)}")

    # Load saved preferences, falling back to defaults on first launch
    settings = load_settings()

    # Main loop: bounce between start screen and detection sessions
    while True:
        choice = show_start_screen(weights, len(names), preview, settings)
        if choice != "start":
            break
        run_detection(model, names, settings)

    # Persist any unsaved settings on exit if user opted in
    if settings.get("save_on_exit", True):
        save_settings(settings)
    print("Goodbye!")


if __name__ == "__main__":
    main()
