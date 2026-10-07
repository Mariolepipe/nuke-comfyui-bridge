"""
comfy_bridge.nuke_panel
-----------------------
Select a node, pick a workflow, press Generate: the frames go to ComfyUI, one file per frame
comes back and is loaded as a Read. Any API workflow dropped in comfy_bridge/workflows is listed
(see client.py for the convention).
"""

import os
import re
import threading
import time

import nuke

try:
    from PySide6 import QtCore, QtWidgets
except ImportError:            # Nuke < 16
    from PySide2 import QtCore, QtWidgets

from . import client as cc

WORKFLOW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")   # API-format workflows
DEFAULT_URL = os.environ.get("COMFY_BRIDGE_URL", "http://127.0.0.1:8188")
DIRECT_EXT = (".png", ".jpg", ".jpeg")       # sent as is; anything else is rendered by Nuke first

_panel = None


def _viewer_display_view():
    """(display, view) of the active viewer, e.g. ('sRGB - Display', 'ACES 1.0 - SDR Video')."""
    try:
        vp = nuke.activeViewer().node()["viewerProcess"].value()
    except Exception:
        return None, None
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", vp or "")
    return (m.group(2).strip(), m.group(1).strip()) if m else (None, vp or None)


def render_source(node, first, last, out_dir, log=print):
    """Render `node` to 8-bit PNGs as the viewer shows it: the models expect display-referred
    images, a linear or log plate looks wrong to them."""
    path = os.path.join(out_dir, "_render", "src.%04d.png").replace("\\", "/")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp, src = [], node
    try:
        display, view = _viewer_display_view()
        if display or view:
            d = nuke.nodes.OCIODisplay()
            tmp.append(d)
            d.setInput(0, node)
            if display:
                d["display"].setValue(display)
            if view:
                d["view"].setValue(view)
            src = d
            log("colour: viewer transform %s / %s" % (view, display))
        w = nuke.nodes.Write(file=path, file_type="png")
        tmp.append(w)
        for knob, val in (("datatype", "8 bit"), ("raw", src is not node)):   # raw: already display-referred
            try:
                w[knob].setValue(val)
            except Exception:                   # knob names differ between Nuke versions
                pass
        w.setInput(0, src)
        nuke.execute(w, first, last, 1)
    finally:
        for n in reversed(tmp):
            nuke.delete(n)
    return path


def _source_name(node):
    """'plate_%04d.exr' / 'plate.####.exr' -> 'plate' (no frame token: Nuke would expand it in paths)."""
    if node.Class() != "Read":
        return node.name()
    base = os.path.basename(node["file"].value().replace("\\", "/"))
    base = re.sub(r"[._-]?(%0?\d*d|#+|\$F\d*)", "", base)
    base = os.path.splitext(base)[0].split(".")[0].strip("._-")
    return base or node.name()


class _Signals(QtCore.QObject):
    progress = QtCore.Signal(int, int)
    log = QtCore.Signal(str)
    finished = QtCore.Signal(str)               # error message, "" on success


class ComfyBridgePanel(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(ComfyBridgePanel, self).__init__(parent)
        self.setWindowTitle("ComfyUI Bridge")
        self.setObjectName("ComfyBridgePanel")
        self.setMinimumWidth(480)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.node = None
        self.cancel_event = None
        self.workflows = cc.list_workflows(WORKFLOW_DIR)
        self.wf = None
        self.sig = _Signals()
        self.sig.progress.connect(self._on_progress)
        self.sig.log.connect(self._log)
        self.sig.finished.connect(self._on_finished)

        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        lay.addLayout(form)

        self.url = QtWidgets.QLineEdit(DEFAULT_URL)
        self.status = QtWidgets.QLabel("")
        form.addRow("ComfyUI", self.url)
        form.addRow("", self.status)

        src = QtWidgets.QHBoxLayout()
        self.src_label = QtWidgets.QLabel("-")
        btn_pick = QtWidgets.QPushButton("Use selected")
        src.addWidget(self.src_label, 1)
        src.addWidget(btn_pick)
        form.addRow("source", src)
        rng = QtWidgets.QHBoxLayout()
        self.first, self.last = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for s in (self.first, self.last):
            s.setRange(-100000, 100000)
            rng.addWidget(s)
        form.addRow("frames", rng)
        self.out_dir = QtWidgets.QLineEdit()
        form.addRow("output", self.out_dir)
        self.skip = QtWidgets.QCheckBox("skip if already on disk")
        self.skip.setChecked(True)
        form.addRow("", self.skip)

        self.wf_combo = QtWidgets.QComboBox()
        self.wf_combo.addItems([w.label for w in self.workflows])
        names = [w.name for w in self.workflows]
        if "nukebridge_all" in names:                   # default workflow
            self.wf_combo.setCurrentIndex(names.index("nukebridge_all"))
        self.wf_desc = QtWidgets.QLabel("")
        self.wf_desc.setWordWrap(True)
        form.addRow("workflow", self.wf_combo)
        form.addRow("", self.wf_desc)

        self.params = {}
        box = QtWidgets.QGroupBox("Settings")
        self.pform = QtWidgets.QFormLayout(box)
        lay.addWidget(box)

        self.bar = QtWidgets.QProgressBar()
        self.btn_run = QtWidgets.QPushButton("Generate")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        run = QtWidgets.QHBoxLayout()
        run.addWidget(self.btn_run, 1)
        run.addWidget(self.btn_cancel)
        self.logbox = QtWidgets.QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setFixedHeight(120)
        lay.addWidget(self.bar)
        lay.addLayout(run)
        lay.addWidget(self.logbox)

        btn_pick.clicked.connect(lambda: self.pick())
        self.btn_run.clicked.connect(self.run)
        self.btn_cancel.clicked.connect(self.cancel)
        self.wf_combo.currentIndexChanged.connect(self._workflow_changed)
        self._workflow_changed()
        self.pick(quiet=True)

    def _workflow_changed(self, *_):
        while self.pform.rowCount():
            self.pform.removeRow(0)
        self.params = {}
        i = self.wf_combo.currentIndex()
        self.wf = self.workflows[i] if 0 <= i < len(self.workflows) else None
        if not self.wf:
            self.wf_desc.setText("No API workflow found in %s" % WORKFLOW_DIR)
            return
        self.wf_desc.setText(self.wf.description)
        for label, (nid, key, default) in self.wf.params().items():
            w = self._widget(nid, key, default)
            self.pform.addRow(label, w)
            self.params[label] = w

    def _widget(self, nid, key, default):
        try:          # model files etc.: list what is installed in ComfyUI
            choices = cc.ComfyClient(self.url.text(), timeout=3).combo_choices(self.wf.graph[nid]["class_type"], key)
        except cc.ComfyError:
            choices = []
        if choices:
            w = QtWidgets.QComboBox()
            w.addItems([str(c) for c in choices])
            w.setCurrentText(str(default))
        elif isinstance(default, bool):
            w = QtWidgets.QCheckBox()
            w.setChecked(default)
        elif isinstance(default, int):
            w = QtWidgets.QSpinBox()
            w.setRange(-1000000, 1000000)
            w.setValue(default)
        elif isinstance(default, float):
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(-1e6, 1e6)
            w.setDecimals(3)
            w.setValue(default)
        else:
            w = QtWidgets.QLineEdit(str(default))
        return w

    def _overrides(self):
        out = {}
        for label, (nid, key, default) in self.wf.params().items():
            w = self.params[label]
            if isinstance(w, QtWidgets.QComboBox):
                v = w.currentText()
                if isinstance(default, int) and not isinstance(default, bool):
                    v = int(v)
            elif isinstance(w, QtWidgets.QCheckBox):
                v = w.isChecked()
            elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                v = w.value()
            else:
                v = w.text()
            out[(nid, key)] = v
        return out

    def pick(self, quiet=False):
        sel = nuke.selectedNodes()
        if not sel:
            if not quiet:
                nuke.message("Select a Read (or any node) first.")
            return
        # several nodes selected: take the last one clicked (nuke.selectedNode), never the Viewer itself
        others = [n for n in sel if n.Class() != "Viewer"]
        try:
            last = nuke.selectedNode()
        except ValueError:
            last = None
        node = last if (last is not None and last.Class() != "Viewer") else (others or sel)[0]
        if node.Class() == "Viewer":            # only the viewer selected: use what it is showing
            idx = 0
            try:
                v = nuke.activeViewer()
                if v and v.node() is node:
                    idx = v.activeInput() or 0
            except Exception:
                pass
            node = node.input(idx) or node.input(0)
            if node is None:
                if not quiet:
                    nuke.message("The Viewer has no input. Select the plate Read.")
                return
        self.node = node
        self.src_label.setText(self.node.name())
        if not quiet:
            self._log("source: %s" % self.node.name())
        self.first.setValue(int(self.node.firstFrame()))
        self.last.setValue(int(self.node.lastFrame()))
        script = nuke.root().name()
        root = os.path.dirname(script) if script != "Root" else os.path.expanduser("~")
        self.out_dir.setText(os.path.join(root, "comfy_passes").replace("\\", "/"))

    def run(self):
        if not self.node or not self.wf:
            return nuke.message("Select a node and a workflow first.")
        try:
            self.cl = cc.ComfyClient(self.url.text().strip())
            self.cl.ping()
            self.status.setText("connected")
        except cc.ComfyError as e:
            self.status.setText("<font color='#e66'>%s</font>" % e)
            return
        if "rgba.red" not in self.node.channels():
            return self._log("ERROR: %s has no rgb channels (a pass Read?). Select the plate." % self.node.name())
        frames = list(range(self.first.value(), self.last.value() + 1))
        name = _source_name(self.node)
        out_dir = os.path.join(self.out_dir.text(), name, self.wf.name).replace("\\", "/")
        p = self.wf.pass_name
        self.out_pattern = "%s/%s/%s_%s.%%04d.%s" % (out_dir, p, name, p, self.wf.ext)
        src = self.node["file"].value() if self.node.Class() == "Read" else ""
        if not src.lower().endswith(DIRECT_EXT):
            self._log("rendering %d frames as seen in the viewer..." % len(frames))
            QtWidgets.QApplication.processEvents()
            try:
                src = render_source(self.node, frames[0], frames[-1], out_dir, log=self._log)
            except Exception as e:
                return self._log("ERROR: source render failed: %s" % e)
            self._log("rendered to %s" % os.path.dirname(src))
        self.range = (frames[0], frames[-1])
        try:                                    # re-read the JSON: picks up edits without reopening the panel
            wf = cc.Workflow(self.wf.path)
        except Exception as e:
            return self._log("ERROR: %s" % e)
        self.run_wf = wf
        self.cancel_event = threading.Event()
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.t0 = time.time()
        args = dict(name=name, overrides=self._overrides(), skip_existing=self.skip.isChecked(),
                    cancel=self.cancel_event, on_progress=self.sig.progress.emit, on_log=self.sig.log.emit)

        def worker():
            try:
                cc.process(self.cl, wf, src, frames, self.out_pattern, **args)
                self.sig.finished.emit("")
            except cc.Cancelled:
                self.sig.finished.emit("cancelled")
            except Exception as e:      # noqa - shown in the panel
                self.sig.finished.emit(str(e))

        threading.Thread(target=worker, daemon=True).start()

    def cancel(self):
        if self.cancel_event:
            self.cancel_event.set()

    def _log(self, msg):
        self.logbox.appendPlainText(msg)

    def _on_progress(self, done, total):
        self.bar.setMaximum(max(total, 1))
        self.bar.setValue(done)
        self.bar.setFormat("%d / %d   (%.0fs)" % (done, total, time.time() - self.t0))

    def _on_finished(self, error):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if error:
            return self._log("ERROR: " + error)
        self._log("done in %.0fs" % (time.time() - self.t0))
        r = nuke.nodes.Read(file=self.out_pattern, first=self.range[0], last=self.range[1],
                            origfirst=self.range[0], origlast=self.range[1])
        r["raw"].setValue(True)                  # data passes: no colour transform
        r["label"].setValue("ComfyUI: " + self.run_wf.label)
        r.setXYpos(self.node.xpos() + 150, self.node.ypos())


def show():
    global _panel
    # close every bridge panel already open, even one created by an older (reloaded) version of this module
    for w in QtWidgets.QApplication.topLevelWidgets():
        if w.objectName() == "ComfyBridgePanel":
            try:
                w.close()
                w.deleteLater()
            except Exception:
                pass
    _panel = ComfyBridgePanel(QtWidgets.QApplication.activeWindow())
    _panel.show()
    return _panel
