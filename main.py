import os, threading, time
import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.slider import Slider
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

Window.clearcolor = (0, 0, 0, 1)


def scan_media(path):
    if platform != "android":
        return
    try:
        from jnius import autoclass
        PA = autoclass("org.kivy.android.PythonActivity")
        MSC = autoclass("android.media.MediaScannerConnection")
        MSC.scanFile(PA.mActivity, [path], None, None)
    except Exception as e:
        print("scan error", e)


def open_cap(path):
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    except Exception:
        pass
    return cap


def detect_rotation(path, cap):
    if platform == "android":
        try:
            from jnius import autoclass
            MMR = autoclass("android.media.MediaMetadataRetriever")
            r = MMR()
            r.setDataSource(path)
            v = r.extractMetadata(24)
            r.release()
            if v is not None:
                return int(v) % 360
        except Exception as e:
            print("rot jnius", e)
    try:
        v = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        if v:
            return int(v) % 360
    except Exception:
        pass
    return 0


def rotate(f, deg):
    deg %= 360
    if deg == 90:
        return cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(f, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return f


def frame_score(f):
    h, w = f.shape[:2]
    g = cv2.cvtColor(cv2.resize(f, (960, max(int(960 * h / w), 1)), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F).var()
    clip = (np.count_nonzero(g < 8) + np.count_nonzero(g > 247)) / g.size
    return lap * (1.0 - min(clip, 0.9))


class TButton(Button):
    """Botón semitransparente."""
    def __init__(self, **kw):
        kw.setdefault("background_normal", "")
        kw.setdefault("background_down", "")
        kw.setdefault("background_color", (0, 0, 0, 0.5))
        kw.setdefault("color", (1, 1, 1, 0.95))
        kw.setdefault("font_size", "12sp")
        kw.setdefault("bold", True)
        self.base = kw["background_color"]
        super().__init__(**kw)
        self.bind(state=self._st)
        self.bind(size=lambda w, s: setattr(w, "text_size", s))
        self.halign = "center"
        self.valign = "middle"
        self.shorten = True

    def _st(self, w, s):
        w.background_color = (1, 1, 1, 0.35) if s == "down" else self.base


class InfoLabel(Label):
    def __init__(self, **kw):
        super().__init__(**kw)
        with self.canvas.before:
            Color(0, 0, 0, 0.35)
            self._r = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._u, size=self._u)

    def _u(self, *a):
        self._r.pos, self._r.size = self.pos, self.size


class Viewer(FloatLayout):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.cap = None
        self.n = 0
        self.fps = 0
        self.idx = 0
        self.start = 0
        self.end = 0
        self.busy = False
        self.path = None
        self.meta_rot = 0
        self.man_rot = 0
        self.playing = False
        self.play_gen = 0
        self.speeds = [1, 2, 4, 8, 16]
        self.si = 0
        self.best = None
        self.use_jpg = False
        self.swap = (platform == "android")  # OpenCV en Android entrega R/B invertidos
        self.tex = None
        self._pend = None
        self._sched = False

        # --- video a pantalla completa ---
        self.img = Image(size_hint=(1, 1), allow_stretch=True, keep_ratio=True)
        self.add_widget(self.img)

        # --- info arriba ---
        self.info = InfoLabel(text="Toca 'Abrir' para elegir un video", size_hint=(1, None),
                              height=dp(28), pos_hint={"x": 0, "top": 0.97}, font_size="12sp")
        self.add_widget(self.info)

        # --- ocultar/mostrar controles ---
        self.hide_btn = TButton(text="UI", size_hint=(None, None), size=(dp(46), dp(32)),
                                pos_hint={"right": 0.99, "top": 0.92})
        self.hide_btn.bind(on_release=lambda *_: self.toggle_ui())
        self.add_widget(self.hide_btn)

        # --- panel de controles semitransparente abajo ---
        self.panel = BoxLayout(orientation="vertical", size_hint=(1, None), height=dp(190),
                               pos_hint={"x": 0, "y": 0}, spacing=dp(3), padding=dp(3))
        self.slider = Slider(min=0, max=1, value=0, size_hint_y=None, height=dp(36), opacity=0.85)
        self.slider.bind(value=self.on_slider)
        self.panel.add_widget(self.slider)

        def row(items):
            r = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(3))
            for t, fn in items:
                b = TButton(text=t)
                b.bind(on_release=lambda *_, fn=fn: fn())
                r.add_widget(b)
            self.panel.add_widget(r)
            return r

        r1 = row([("Abrir", self.pick), ("-10", lambda: self.step(-10)), ("-1", lambda: self.step(-1)),
                  ("Play", self.toggle_play), ("+1", lambda: self.step(1)), ("+10", lambda: self.step(10)),
                  ("x1", self.cycle_speed)])
        self.bplay = r1.children[3]
        self.bspd = r1.children[0]
        r2 = row([("Ini", self.set_start), ("Fin", self.set_end), ("Giro", self.rot90),
                  ("Col", self.toggle_color), ("PNG", self.toggle_fmt)])
        self.bfmt = r2.children[0]
        row([("Foto", self.save_one), ("Rango", self.save_range),
             ("Mejor", self.find_best), ("Top10", self.auto_sharp)])
        self.add_widget(self.panel)

    # ---------- UI ----------
    def toggle_ui(self):
        vis = self.panel.opacity > 0
        for w in (self.panel, self.info):
            w.opacity = 0 if vis else 1
            w.disabled = vis

    def toggle_color(self):
        self.swap = not self.swap
        if self.cap and not self.playing:
            self.show(self.idx)

    def toggle_fmt(self):
        self.use_jpg = not self.use_jpg
        self.bfmt.text = "JPG100" if self.use_jpg else "PNG"

    @mainthread
    def msg(self, t):
        self.info.text = t

    def ext(self):
        return ("jpg", [cv2.IMWRITE_JPEG_QUALITY, 100]) if self.use_jpg \
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
            filechooser.open_file(on_selection=self.on_pick,
                                  filters=["*.mp4", "*.mov", "*.mkv", "*.3gp"])
        except Exception as e:
            self.msg("Error selector: %s" % e)

    @mainthread
    def on_pick(self, sel):
        if sel:
            self.load(sel[0])

    def load(self, path):
        self.stop_play()
        if self.cap:
            self.cap.release()
        cap = open_cap(path)
        if not cap.isOpened():
            self.msg("No se pudo abrir el video")
            return
        self.cap, self.path = cap, path
        self.meta_rot = detect_rotation(path, cap)
        self.man_rot = 0
        self.best = None
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 0
        self.start, self.end = 0, max(self.n - 1, 0)
        self.slider.max = max(self.n - 1, 1)
        self.show(0)

    # ---------- visor ----------
    def on_slider(self, _, v):
        if self.cap and int(v) != self.idx:
            self.stop_play()
            self.show(int(v))

    def step(self, d):
        if self.cap:
            self.stop_play()
            self.show(min(max(self.idx + d, 0), self.n - 1))

    def orient(self, f):
        return rotate(f, self.meta_rot + self.man_rot)

    def bgr(self, f):
        """Frame girado y en orden BGR correcto (para guardar)."""
        f = self.orient(f)
        if self.swap:
            f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        return f

    def prep(self, f):
        """Se puede llamar desde cualquier hilo."""
        f = self.orient(f)
        oh, ow = f.shape[:2]
        sc = 1280.0 / max(ow, oh)
        p = cv2.resize(f, (int(ow * sc), int(oh * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else f
        if not self.swap:
            p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        p = cv2.flip(p, 0)
        return p.tobytes(), p.shape[1], p.shape[0], ow, oh

    def present(self, pr, i):
        """Solo hilo principal."""
        buf, w, h, ow, oh = pr
        if self.tex is None or tuple(self.tex.size) != (w, h):
            self.tex = Texture.create(size=(w, h), colorfmt="rgb")
            self.img.texture = self.tex
        self.tex.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")
        self.img.canvas.ask_update()
        self.idx = i
        if int(self.slider.value) != i:
            self.slider.unbind(value=self.on_slider)
            self.slider.value = i
            self.slider.bind(value=self.on_slider)
        mark = "  *MEJOR*" if self.best == i else ""
        self.info.text = "%d/%d  %.0ffps  %dx%d  rot%d  col%s  [%d-%d]%s" % (
            i, self.n - 1, self.fps, ow, oh, (self.meta_rot + self.man_rot) % 360,
            "B" if self.swap else "A", self.start, self.end, mark)

    def show(self, i):
        f = self.read(i)
        if f is not None:
            self.present(self.prep(f), i)

    def set_start(self):
        self.start = self.idx
        self.show(self.idx)

    def set_end(self):
        self.end = self.idx
        self.show(self.idx)

    def rot90(self):
        self.man_rot = (self.man_rot + 90) % 360
        if self.cap and not self.playing:
            self.show(self.idx)

    # ---------- reproducción (hilo aparte: no bloquea la pantalla) ----------
    def toggle_play(self):
        if not self.cap:
            self.msg("Primero elige un video")
            return
        if self.playing:
            self.stop_play()
        else:
            self.playing = True
            self.play_gen += 1
            self.bplay.text = "Pausa"
            threading.Thread(target=self.play_loop, args=(self.play_gen, self.idx), daemon=True).start()

    def stop_play(self):
        self.playing = False
        self.play_gen += 1
        try:
            self.bplay.text = "Play"
        except Exception:
            pass

    def cycle_speed(self):
        self.si = (self.si + 1) % len(self.speeds)
        self.bspd.text = "x%d" % self.speeds[self.si]

    def post(self, pr, i):
        self._pend = (pr, i)
        if not self._sched:
            self._sched = True
            Clock.schedule_once(self._apply)

    def _apply(self, dt):
        self._sched = False
        p, self._pend = self._pend, None
        if p and self.playing:
            self.present(*p)

    @mainthread
    def _ended(self, gen):
        if gen == self.play_gen:
            self.stop_play()

    def play_loop(self, gen, start):
        cap = None
        try:
            cap = open_cap(self.path)
            i = start + 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            while self.playing and gen == self.play_gen:
                t0 = time.time()
                if self.n and i >= self.n:
                    break
                ok, f = cap.read()
                if not ok:
                    break
                self.post(self.prep(f), i)
                sp = self.speeds[self.si]
                for _ in range(sp - 1):
                    cap.grab()
                i += sp
                dt = 0.04 - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
        except Exception as e:
            self.msg("Error play: %s" % e)
        finally:
            if cap:
                cap.release()
        self._ended(gen)

    # ---------- mejor frame ----------
    @mainthread
    def goto(self, i):
        self.best = i
        self.show(i)

    def find_best(self):
        if not self.cap:
            return
        self.stop_play()
        a, b = sorted((self.start, self.end))
        stride = 1 if (b - a) <= 600 else max(int((b - a) / 600), 2)

        def job():
            cap = open_cap(self.path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            best, bi, i = -1, a, a
            while i <= b:
                ok, f = cap.read()
                if not ok:
                    break
                s_ = frame_score(f)
                if s_ > best:
                    best, bi = s_, i
                for _ in range(stride - 1):
                    cap.grab()
                if ((i - a) // stride) % 20 == 0:
                    self.msg("Buscando mejor frame %d%%" % (100 * (i - a) / max(b - a, 1)))
                i += stride
            if stride > 1:
                lo, hi = max(bi - stride, a), min(bi + stride, b)
                cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
                for j in range(lo, hi + 1):
                    ok, f = cap.read()
                    if not ok:
                        break
                    s_ = frame_score(f)
                    if s_ > best:
                        best, bi = s_, j
            cap.release()
            self.goto(bi)
        self.run_bg(job)

    # ---------- guardado ----------
    def name(self, i):
        base = os.path.splitext(os.path.basename(self.path))[0]
        return os.path.join(OUT_DIR, "%s_f%06d.%s" % (base, i, self.ext()[0]))

    def write(self, f, i):
        os.makedirs(OUT_DIR, exist_ok=True)
        p = self.name(i)
        cv2.imwrite(p, self.bgr(f), self.ext()[1])
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
        if not self.cap:
            return
        self.stop_play()
        a, b = sorted((self.start, self.end))

        def job():
            cap = open_cap(self.path)
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

    def auto_sharp(self, top=10):
        if not self.cap:
            return
        self.stop_play()
        a, b = sorted((self.start, self.end))

        def job():
            cap = open_cap(self.path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            sc = []
            for i in range(a, b + 1):
                ok, f = cap.read()
                if not ok:
                    break
                sc.append((frame_score(f), i))
                if i % 20 == 0:
                    self.msg("Analizando %d/%d" % (i - a, b - a))
            cap.release()
            gap = max(int(self.fps / 10), 2)
            sel = []
            for s_, i in sorted(sc, reverse=True):
                if all(abs(i - j) >= gap for j in sel):
                    sel.append(i)
                if len(sel) >= top:
                    break
            cap = open_cap(self.path)
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

    def on_pause(self):
        self.root.stop_play()
        return True

    def on_stop(self):
        v = self.root
        v.stop_play()
        if v.cap:
            v.cap.release()


if __name__ == "__main__":
    FramesApp().run()
