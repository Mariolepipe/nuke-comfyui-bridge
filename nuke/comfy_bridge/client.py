"""
comfy_bridge.client
-------------------
Minimal ComfyUI HTTP client. Standard library only, so it runs in Nuke's Python.

Any ComfyUI workflow saved with "Export (API)" into comfy_bridge/workflows works if it has:
  * a LoadImagesFromFolderKJ node titled NUKE_INPUT (frames are read from a local folder)
  * one save node whose filename_prefix starts with "nuke_" and that writes one file per frame
    ("nuke_geo" -> pass "geo"; an EXR save node gives .exr files, anything else .png)
  * optionally, a top-level "_nuke_bridge" entry:
      {"name": "shown in the panel", "description": "...", "chunk": 25, "overlap": 2,
       "params": {"label shown in the panel": ["node_id", "input_name"], ...}}
"""

import copy
import json
import os
import re
import shutil
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request

INPUT_TITLE = "NUKE_INPUT"
PASS_PREFIX = "nuke_"


class ComfyError(RuntimeError):
    pass


class Cancelled(Exception):
    pass


class ComfyClient(object):
    def __init__(self, url="http://127.0.0.1:8188", timeout=30):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    def _request(self, method, path, data=None, headers=None, params=None):
        full = self.url + path + ("?" + urllib.parse.urlencode(params) if params else "")
        req = urllib.request.Request(full, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            raise ComfyError("HTTP %s on %s: %s" % (e.code, path, e.read().decode("utf-8", "replace")[:2000]))
        except urllib.error.URLError as e:
            raise ComfyError("Cannot reach ComfyUI at %s (%s)" % (self.url, e.reason))

    def _get(self, path, params=None):
        return json.loads(self._request("GET", path, params=params).decode("utf-8"))

    def _post(self, path, payload):
        raw = self._request("POST", path, data=json.dumps(payload).encode("utf-8"),
                            headers={"Content-Type": "application/json"})
        return json.loads(raw.decode("utf-8")) if raw else {}

    def ping(self):
        return self._get("/system_stats")

    def combo_choices(self, node_class, input_name):
        """Options of a COMBO input (e.g. the installed model files)."""
        spec = self._get("/object_info/" + urllib.parse.quote(node_class)).get(node_class, {}).get("input", {})
        entry = spec.get("required", {}).get(input_name) or spec.get("optional", {}).get(input_name)
        if not entry:
            return []
        if isinstance(entry[0], list):                          # old format: [[a, b], {...}]
            return entry[0]
        if len(entry) > 1 and isinstance(entry[1], dict):       # new format: ["COMBO", {"options": [...]}]
            return entry[1].get("options", [])
        return []

    def queue(self, prompt):
        res = self._post("/prompt", {"prompt": prompt, "client_id": self.client_id})
        if res.get("node_errors"):
            raise ComfyError("Workflow rejected: %s" % json.dumps(res["node_errors"])[:2000])
        return res["prompt_id"]

    def wait(self, prompt_id, cancel=None, poll=1.0, timeout=6 * 3600):
        start = time.time()
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            entry = self._get("/history/" + prompt_id).get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    raise ComfyError("ComfyUI error: %s" % (msgs[0][1].get("exception_message", "").strip()
                                                            if msgs else ""))
                if status.get("completed", True):
                    return entry
            if time.time() - start > timeout:
                raise ComfyError("Timed out waiting for prompt %s" % prompt_id)
            time.sleep(poll)

    def download(self, info, dest):
        data = self._request("GET", "/view", params={"filename": info["filename"],
                                                     "subfolder": info.get("subfolder", ""),
                                                     "type": info.get("type", "output")})
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest + ".part", "wb") as f:
            f.write(data)
        os.replace(dest + ".part", dest)

    def interrupt(self, prompt_id):
        try:
            self._post("/queue", {"delete": [prompt_id]})
            self._post("/interrupt", {})
        except ComfyError:
            pass


class Workflow(object):
    def __init__(self, path):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r") as f:
            data = json.load(f)
        if "nodes" in data and "links" in data:
            raise ComfyError("%s is a UI workflow; in ComfyUI use Export (API)" % self.name)
        self.meta = data.pop("_nuke_bridge", {})
        self.label = self.meta.get("name", self.name)        # shown in the panel; files use self.name
        self.graph = data
        self.input_id = self.save_id = None
        for nid, node in data.items():
            if not isinstance(node, dict):
                continue
            if node.get("_meta", {}).get("title") == INPUT_TITLE:
                self.input_id = nid
            if str(node.get("inputs", {}).get("filename_prefix", "")).startswith(PASS_PREFIX):
                self.save_id = nid
        if self.input_id is None or self.save_id is None:
            raise ComfyError("%s: needs a node titled %s and a save node with a '%s' prefix"
                             % (self.name, INPUT_TITLE, PASS_PREFIX))
        prefix = data[self.save_id]["inputs"]["filename_prefix"]
        self.pass_name = prefix[len(PASS_PREFIX):].split("/")[0] or "out"
        self.ext = "exr" if "EXR" in data[self.save_id].get("class_type", "").upper() else "png"

    @property
    def description(self):
        return self.meta.get("description", "")

    def params(self):
        """{label: (node_id, input_name, default)} shown in the panel."""
        return {label: (nid, key, self.graph[nid]["inputs"].get(key))
                for label, (nid, key) in self.meta.get("params", {}).items() if nid in self.graph}

    def build(self, folder, count, tag, overrides=None):
        g = copy.deepcopy(self.graph)
        g[self.input_id]["inputs"].update({"folder": folder, "image_load_cap": count, "start_index": 0})
        g[self.save_id]["inputs"]["filename_prefix"] = "nuke_bridge/%s" % tag
        for (nid, key), value in (overrides or {}).items():
            if nid in g:
                g[nid]["inputs"][key] = value
        return g


def list_workflows(folder):
    out = []
    for fn in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if fn.lower().endswith(".json"):
            try:
                out.append(Workflow(os.path.join(folder, fn)))
            except (ComfyError, ValueError, KeyError) as e:
                print("[comfy_bridge] skipped %s: %s" % (fn, e))
    return out


_FRAME_RE = re.compile(r"(%0?(\d*)d|#+)")


def frame_path(pattern, frame):
    """'plate.%04d.png' / 'plate.####.png' -> 'plate.1001.png'"""
    def repl(m):
        tok = m.group(1)
        return str(frame).zfill(len(tok) if tok.startswith("#") else int(m.group(2) or 0))
    return _FRAME_RE.sub(repl, pattern, count=1)


def _stage(src_pattern, frames, folder):
    """Hard-link (or copy) the frames into `folder`, named so they sort in order."""
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    ext = os.path.splitext(src_pattern)[1]
    for i, f in enumerate(frames):
        src = frame_path(src_pattern, f)
        if not os.path.exists(src):
            raise ComfyError("missing source frame: %s" % src)
        dst = os.path.join(folder, "f%06d%s" % (i, ext))
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _output_files(node_output):
    for value in node_output.values():
        if isinstance(value, list) and value and isinstance(value[0], dict) and "filename" in value[0]:
            return value
    return []


def process(client, wf, src_pattern, frames, out_pattern, name="nuke", overrides=None, skip_existing=True,
            cancel=None, on_progress=None, on_log=None):
    """Send the frames in chunks, download one output file per frame to out_pattern."""
    log = on_log or (lambda m: None)
    progress = on_progress or (lambda d, t: None)
    frames = list(frames)
    total = len(frames)
    if skip_existing and all(os.path.exists(frame_path(out_pattern, f)) for f in frames):
        log("all frames already on disk, skipped")
        progress(total, total)
        return
    chunk = int(wf.meta.get("chunk", 0)) or total
    # overlap: extra frames on each side of a chunk, processed but not kept, so the temporal
    # stabilization sees real neighbours at chunk boundaries
    overlap = int(wf.meta.get("overlap", 0))
    stage = os.path.join(os.path.dirname(os.path.dirname(out_pattern)), "_src")
    progress(0, total)
    try:
        for c0 in range(0, total, chunk):
            part = frames[c0:c0 + chunk]
            lo, hi = max(0, c0 - overlap), min(total, c0 + chunk + overlap)
            sent = frames[lo:hi]
            _stage(src_pattern, sent, stage)
            t0 = time.time()
            log("frames %d-%d (+%d overlap)..." % (part[0], part[-1], len(sent) - len(part)))
            pid = client.queue(wf.build(stage.replace("\\", "/"), len(sent), "%s_%04d" % (name, part[0]), overrides))
            try:
                entry = client.wait(pid, cancel=cancel)
            except Cancelled:
                client.interrupt(pid)
                raise
            files = _output_files(entry.get("outputs", {}).get(wf.save_id, {}))
            if len(files) != len(sent):
                raise ComfyError("got %d file(s) for %d frame(s)" % (len(files), len(sent)))
            for f, info in zip(part, files[c0 - lo:c0 - lo + len(part)]):
                client.download(info, frame_path(out_pattern, f))
            progress(c0 + len(part), total)
            log("  done in %.0fs (%.2fs/frame)" % (time.time() - t0, (time.time() - t0) / len(part)))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
