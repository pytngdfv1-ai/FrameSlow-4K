import os, threading, time
import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton
from kivy.utils import platform

OUT_DIR = "/storage/emulated/0/DCIM/Frames4K"
if platform != "android":
    OUT_DIR = os.path.expanduser("~/Frames4K")

PERMS = []
if platform == "android":
    from android.permissions import request_permissions, Permission
    PERMS = ["android.permission.READ_MEDIA_VIDEO",
             "android.permission.READ_MEDIA_IMAGES",
             Permission.READ_EXTERNAL_STORAGE,
             Permission.WRITE_EXTERNAL_STORAGE]


def scan_media(path):
    """Hace visible el archivo en la galería."""
    if platform != "android":
        return
    try:
        from jnius import autoclass
        PA = autoclass("org.kivy.android.PythonActivity")
        MSC = autoclass("android.media.MediaScannerConnection")
        MSC.scanFile(PA.mActivity, [path], None, None)
    except Exception as e:
        print("scan error", e)


class Viewer(BoxLayout):
    def __init__(self, **kw):
        super().__init__(orientation="vertical", spacing=4, padding=4, **kw)
        self.cap = None
        self.n = 0
        self.fps = 0
        self.idx = 0
        self.start = 0
        self.end = 0
        self.busy = False
        self.path = None

        self.img = Image(allow_stretch=True, keep_ratio=True, size_hint_y=0.5)
        self.info = Label(text="Selecciona un video", size_hint_y=0.07)
        self.slider = Slider(min=0, max=1, value=0, size_hint_y=0.07)
        self.slider.bind(value=self.on_slider)
        self.panel = BoxLayout(orientation="vertical", spacing=4)
        self.add_widget(self.img)
        self.add_widget(self.panel)
        self.info.halign = "center"
        self.info.valign = "middle"
        self.info.bind(size=lambda w, s: setattr(w, "text_size", s))
        self.panel.add_widget(self.info)
        self.panel.add_widget(self.slider)

        row = BoxLayout(size_hint_y=0.08, spacing=4)
        row.add_widget(Button(text="Video", on_release=lambda *_: self.pick()))
        for t, d in (("-10", -10), ("-1", -1), ("+1", 1), ("+10", 10)):
            row.add_widget(Button(text=t, on_release=lambda _, d=d: self.step(d)))
        self.panel.add_widget(row)

        row2 = BoxLayout(size_hint_y=0.08, spacing=4)
        row2.add_widget(Button(text="Inicio", on_release=lambda *_: self.set_start()))
        row2.add_widget(Button(text="Fin", on_release=lambda *_: self.set_end()))
        self.fmt = ToggleButton(text="PNG", state="normal")
        self.fmt.bind(state=lambda w, s: setattr(w, "text", "JPG100" if s == "down" else "PNG"))
        row2.add_widget(self.fmt)
        self.panel.add_widget(row2)

        row3 = BoxLayout(size_hint_y=0.08, spacing=4)
        row3.add_widget(Button(text="Guardar frame", on_release=lambda *_: self.save_one()))
        row3.add_widget(Button(text="Guardar rango", on_release=lambda *_: self.save_range()))
        row3.add_widget(Button(text="Auto nítidos", on_release=lambda *_: self.auto_sharp()))
        self.panel.add_widget(row3)
        self.bind(size=self.relayout)
        self.relayout()

    def relayout(self, *a):
        land = self.width > self.height
        self.orientation = "horizontal" if land else "vertical"
        self.img.size_hint = (0.62, 1) if land else (1, 0.5)
        self.panel.size_hint = (0.38, 1) if land else (1, 0.5)

    # ---------- util ----------
    @mainthread
    def msg(self, t):
        self.info.text = t

    def ext(self):
        return ("jpg", [cv2.IMWRITE_JPEG_QUALITY, 100]) if self.fmt.state == "down" \
            else ("png", [cv2.IMWRITE_PNG_COMPRESSION, 1])

    def read(self, i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = self.cap.read()
        return f if ok else None

    # ---------- carga ----------
    def pick(self):
        if platform == "android":
            request_permissions(PERMS)
        try:
            from plyer import filechooser
            filechooser.open_file(on_selection=self.on_pick, filters=["*.mp4", "*.mov", "*.mkv", "*.3gp"])
        except Exception as e:
            self.msg("Error selector: %s" % e)

    @mainthread
    def on_pick(self, sel):
        if not sel:
            return
        self.load(sel[0])

    def load(self, path):
        if self.cap:
            self.cap.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.msg("No se pudo abrir el video")
            return
        self.cap, self.path = cap, path
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 0
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.start, self.end = 0, max(self.n - 1, 0)
        self.slider.max = max(self.n - 1, 1)
        self.show(0)

    # ---------- visor ----------
    def on_slider(self, _, v):
        if self.cap and int(v) != self.idx:
            self.show(int(v))

    def step(self, d):
        if self.cap:
            self.show(min(max(self.idx + d, 0), self.n - 1))

    def show(self, i):
        f = self.read(i)
        if f is None:
            return
        self.idx = i
        if int(self.slider.value) != i:
            self.slider.unbind(value=self.on_slider)
            self.slider.value = i
            self.slider.bind(value=self.on_slider)
        p = cv2.resize(f, (960, int(960 * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA)
        p = cv2.flip(cv2.cvtColor(p, cv2.COLOR_BGR2RGB), 0)
        tex = Texture.create(size=(p.shape[1], p.shape[0]), colorfmt="rgb")
        tex.blit_buffer(p.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self.img.texture = tex
        self.info.text = "%d/%d  %.0ffps  %dx%d  [%d-%d]" % (
            i, self.n - 1, self.fps, self.w, self.h, self.start, self.end)

    def set_start(self):
        self.start = self.idx
        self.show(self.idx)

    def set_end(self):
        self.end = self.idx
        self.show(self.idx)

    # ---------- guardado ----------
    def name(self, i):
        base = os.path.splitext(os.path.basename(self.path))[0]
        e, _ = self.ext()
        return os.path.join(OUT_DIR, "%s_f%06d.%s" % (base, i, e))

    def write(self, f, i):
        os.makedirs(OUT_DIR, exist_ok=True)
        p = self.name(i)
        cv2.imwrite(p, f, self.ext()[1])  # resolución original, sin reescalar
        scan_media(p)
        return p

    def save_one(self):
        if not self.cap:
            return
        f = self.read(self.idx)
        if f is not None:
            self.msg("Guardado: " + self.write(f, self.idx))

    def run_bg(self, fn):
        if self.busy or not self.cap:
            return
        self.busy = True

        def w():
            try:
                fn()
            except Exception as e:
                self.msg("Error: %s" % e)
            self.busy = False
        threading.Thread(target=w, daemon=True).start()

    def save_range(self):
        a, b = sorted((self.start, self.end))

        def job():
            cap = cv2.VideoCapture(self.path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            for i in range(a, b + 1):
                ok, f = cap.read()
                if not ok:
                    break
                self.write(f, i)
                self.msg("Guardando %d/%d" % (i - a + 1, b - a + 1))
            cap.release()
            self.msg("Rango guardado en DCIM/Frames4K")
        self.run_bg(job)

    # ---------- nitidez ----------
    def auto_sharp(self, top=10):
        a, b = sorted((self.start, self.end))

        def job():
            cap = cv2.VideoCapture(self.path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            sc = []
            for i in range(a, b + 1):
                ok, f = cap.read()
                if not ok:
                    break
                g = cv2.cvtColor(cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                sc.append((cv2.Laplacian(g, cv2.CV_64F).var(), i))
                if i % 20 == 0:
                    self.msg("Analizando %d/%d" % (i - a, b - a))
            cap.release()
            # mejores N con separación mínima entre frames
            gap = max(int(self.fps / 10), 2)
            sel = []
            for s, i in sorted(sc, reverse=True):
                if all(abs(i - j) >= gap for j in sel):
                    sel.append(i)
                if len(sel) >= top:
                    break
            cap = cv2.VideoCapture(self.path)
            for k, i in enumerate(sorted(sel)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, f = cap.read()
                if ok:
                    self.write(f, i)
                self.msg("Nítido %d/%d (frame %d)" % (k + 1, len(sel), i))
            cap.release()
            self.msg("%d frames nítidos guardados" % len(sel))
        self.run_bg(job)


class FramesApp(App):
    def build(self):
        if platform == "android":
            request_permissions(PERMS)
        return Viewer()

    def on_stop(self):
        v = self.root
        if v.cap:
            v.cap.release()


if __name__ == "__main__":
    FramesApp().run()
