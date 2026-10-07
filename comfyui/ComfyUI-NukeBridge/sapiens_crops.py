"""
Sapiens2NormalCrops - run Sapiens2 normals on per-person crops instead of the whole frame.

Why: Sapiens2 takes a 1024x768 portrait input. A 2048x870 plate is squeezed into 768x326,
so a face 120 px tall in the plate is ~45 px for the model -> soft, mushy faces.
Cropping each person (and optionally their head) brings the model back to its native
scale: ~3x more pixels on the body, ~6-8x on the face.

Crops are computed once per batch (union of the person mask over all frames), so the
framing does not jitter frame to frame. Each crop is pasted back with a feathered edge,
head crops over body crops. Uses the SAPIENS2_MODEL from ComfyUI-Sapiens2-Easy without
importing that pack.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy import ndimage as _ndi
except Exception:  # scipy missing -> one crop around all people
    _ndi = None

ASPECT = 768.0 / 1024.0  # Sapiens2 input width / height


def _bbox(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _fit(box, W, H, pad):
    """pad the box, grow it to the Sapiens aspect, keep it inside the frame when possible."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w, h = w * (1 + 2 * pad), h * (1 + 2 * pad)
    if w / h > ASPECT:
        h = w / ASPECT
    else:
        w = h * ASPECT
    w, h = min(w, W), min(h, H)  # bigger than the frame: the Sapiens pipeline pads the rest
    x0 = int(round(min(max(cx - w / 2, 0), W - w)))
    y0 = int(round(min(max(cy - h / 2, 0), H - h)))
    return [x0, y0, x0 + int(round(w)), y0 + int(round(h))]


def _feather(box, W, H, f):
    """alpha ramp on the crop sides that are not on the frame border."""
    x0, y0, x1, y1 = box
    h, w = y1 - y0, x1 - x0
    f = max(int(f), 1)
    ax = np.ones(w, np.float32)
    ay = np.ones(h, np.float32)
    r = np.clip((np.arange(w, dtype=np.float32) + 0.5) / f, 0, 1)
    if x0 > 0:
        ax = np.minimum(ax, r)
    if x1 < W:
        ax = np.minimum(ax, r[::-1])
    r = np.clip((np.arange(h, dtype=np.float32) + 0.5) / f, 0, 1)
    if y0 > 0:
        ay = np.minimum(ay, r)
    if y1 < H:
        ay = np.minimum(ay, r[::-1])
    a = ay[:, None] * ax[None, :]
    return a * a * (3 - 2 * a)  # smoothstep


# Sapiens2 segmentation classes that make up a head: eyeglass, face/neck, hair, lips, teeth, tongue
HEAD_CLASSES = (2, 3, 4, 24, 25, 26, 27, 28)


def compute_crops(union, W, H, pad=0.08, head=True, head_frac=0.3, min_area=0.002, head_union=None):
    """-> list of crop boxes, body crops first then head crops.
    head_union: pixels labelled as head (face, hair...) over the batch. When given, each head crop is
    fitted on those pixels; otherwise it falls back to the top head_frac of the body box."""
    if _ndi is not None:
        lab, n = _ndi.label(union)
        comps = [lab == k for k in range(1, n + 1)]
    else:
        comps = [union]
    comps = [c for c in comps if c.sum() >= min_area * W * H]
    bodies, heads = [], []
    for c in comps:
        b = _bbox(c)
        bodies.append(_fit(b, W, H, pad))
        if head:
            hb = None
            if head_union is not None:
                hp = c & head_union
                if hp.sum() >= 0.0002 * W * H:          # a real head, not a few stray pixels
                    hb = _bbox(hp)
            if hb is None:                              # no head labels: top part of the body box
                top = b[1] + max(int((b[3] - b[1]) * head_frac), 8)
                hm = c.copy()
                hm[top:] = False
                hb = _bbox(hm)
            if hb is not None:
                heads.append(_fit(hb, W, H, pad * 2))
    compute_crops.last_n_body = len(bodies)
    compute_crops.last_comps = comps
    return bodies + heads


def track_heads(comps, head_frames, W, H, pad, smooth=1.5):
    """Per-frame head crops that follow the head: fixed size (largest head of the batch + margin),
    centre tracked frame by frame and smoothed in time. -> list (one per person) of per-frame boxes,
    or None for a person with no head pixels at all."""
    B = head_frames.shape[0]
    tracks = []
    for c in comps:
        boxes = [_bbox(head_frames[i] & c) for i in range(B)]
        valid = [b for b in boxes if b is not None and (b[2] - b[0]) * (b[3] - b[1]) >= 0.0001 * W * H]
        if not valid:
            tracks.append(None)
            continue
        sw = max(b[2] - b[0] for b in valid)
        sh = max(b[3] - b[1] for b in valid)
        cx = np.array([np.nan if b is None else (b[0] + b[2]) / 2.0 for b in boxes])
        cy = np.array([np.nan if b is None else (b[1] + b[3]) / 2.0 for b in boxes])
        idx = np.arange(B)
        ok = ~np.isnan(cx)
        cx = np.interp(idx, idx[ok], cx[ok])           # frames without a head: nearest known position
        cy = np.interp(idx, idx[ok], cy[ok])
        if smooth > 0 and B > 1:                        # gaussian smoothing in time: no jitter
            r = int(3 * smooth)
            k = np.exp(-0.5 * (np.arange(-r, r + 1) / smooth) ** 2)
            k /= k.sum()
            cx = np.convolve(np.pad(cx, r, mode="edge"), k, mode="valid")
            cy = np.convolve(np.pad(cy, r, mode="edge"), k, mode="valid")
        tracks.append([_fit([x - sw / 2, y - sh / 2, x + sw / 2, y + sh / 2], W, H, pad) for x, y in zip(cx, cy)])
    return tracks


def _infer(model, crop_rgb):
    """crop_rgb: (h, w, 3) float 0..1 torch -> (3, h, w) unit normals, Sapiens2 convention."""
    rgb = (crop_rgb.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    data = model.model.pipeline(dict(img=rgb[:, :, ::-1].copy()))
    data = model.model.data_preprocessor(data)
    if model.dtype != torch.float32:
        data["inputs"] = data["inputs"].to(dtype=model.dtype)
    with torch.inference_mode():
        n = model.model(data["inputs"]).float()
    n = n / torch.norm(n, dim=1, keepdim=True).clamp(min=1e-8)
    pad = data["data_samples"]["meta"].get("padding_size", (0, 0, 0, 0))
    if isinstance(pad, torch.Tensor):
        pad = pad.detach().cpu().tolist()
    l, r, t, b = (int(v) for v in pad)
    n = n[:, :, t:n.shape[2] - b, l:n.shape[3] - r]
    n = F.interpolate(n, size=crop_rgb.shape[:2], mode="bilinear", align_corners=False)[0]
    return n.cpu()


def _soft_mask(m, erode, blur):
    t = m[:, None].float()
    if erode > 0:
        t = -F.max_pool2d(-t, 2 * erode + 1, stride=1, padding=erode)
    if blur > 0:
        k = 2 * blur + 1
        t = F.avg_pool2d(F.avg_pool2d(t, k, 1, blur, count_include_pad=False), k, 1, blur, count_include_pad=False)
    return t[:, 0].clamp(0, 1)


# ---------------- debug views ----------------
def _draw_boxes(img, boxes, n_body, mask):
    """plate + person mask tint + crop boxes (green = body, amber = head) with their number."""
    from PIL import Image, ImageDraw
    a = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    a = a.copy()
    a[mask] = (a[mask] * 0.6 + np.array([60, 90, 255]) * 0.4).astype(np.uint8)   # blue tint = person mask
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    lw = max(2, im.width // 400)
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        col = (61, 220, 151) if k < n_body else (255, 186, 73)
        d.rectangle((x0, y0, x1 - 1, y1 - 1), outline=col, width=lw)
        d.rectangle((x0, y0, x0 + 34, y0 + 26), fill=col)
        d.text((x0 + 8, y0 + 6), str(k + 1), fill=(0, 0, 0))
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)


def _tile_sheet(crops, th=384):
    """one tile per crop: plate crop on top, Sapiens normals of that crop below (3:4, fixed size)."""
    tw = int(th * ASPECT)
    cols = []
    for rgb, n in crops:
        r = F.interpolate(rgb.permute(2, 0, 1)[None].float(), size=(th, tw), mode="bilinear", align_corners=False)[0]
        nn = F.interpolate(((n + 1) * 0.5)[None].float(), size=(th, tw), mode="bilinear", align_corners=False)[0]
        col = torch.cat((r, nn), 1)
        col = F.pad(col, (4, 4, 4, 4), value=0.05)
        cols.append(col)
    if not cols:
        return torch.zeros(2 * th, tw, 3)
    return torch.cat(cols, 2).permute(1, 2, 0).clamp(0, 1)


class Sapiens2NormalCrops:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),

                "padding": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01,
                                      "tooltip": "Extra margin around each person, fraction of its size."}),
                "head_pass": ("BOOLEAN", {"default": True,
                                          "tooltip": "Second, tighter crop on each head (top of the person) "
                                                     "for sharper faces."}),
                "head_fraction": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.6, "step": 0.05,
                                            "tooltip": "Fallback only: top part of each person used as the head crop when no head pixels (face, hair) are labelled."}),
                "feather_px": ("INT", {"default": 24, "min": 1, "max": 256,
                                       "tooltip": "Blend width when pasting crops back."}),
                "mask_erode_px": ("INT", {"default": 3, "min": 0, "max": 64,
                                          "tooltip": "Shrink the output mask so the silhouette edge stays MoGe."}),
                "mask_blur_px": ("INT", {"default": 6, "min": 0, "max": 64,
                                         "tooltip": "Soften the output mask: no hard seam Sapiens/MoGe."}),
            },
            "optional": {
                "labels": ("SAPIENS2_LABELS", {"tooltip": "Sapiens2 Segmentation 'labels' output (recommended): "
                                                          "every body part counts as person, whatever the parts widget."}),
                "person_mask": ("MASK", {"tooltip": "Or any person matte (e.g. your own roto). Used if labels "
                                                    "is not connected."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("normal_map", "soft_mask", "debug_boxes", "debug_crops")
    FUNCTION = "run"
    CATEGORY = "NukeBridge"
    DESCRIPTION = "Sapiens2 normals run per person crop (+ head crop) at native resolution, pasted back."

    def run(self, model, image, padding=0.08, head_pass=True, head_fraction=0.3,
            feather_px=24, mask_erode_px=3, mask_blur_px=6, labels=None, person_mask=None):
        if getattr(model, "task", "normal") != "normal":
            raise ValueError("Sapiens2NormalCrops needs a Sapiens2 'normal' model, got %r" % model.task)
        B, H, W = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        head_np = None
        if labels is not None and isinstance(labels, dict) and "class_ids" in labels:
            cid = torch.as_tensor(labels["class_ids"])
            if cid.ndim == 2:
                cid = cid[None]
            pm = (cid > 0).float()
            hd = torch.zeros_like(pm)
            for k in HEAD_CLASSES:
                hd = torch.maximum(hd, (cid == k).float())
            if hd.shape[1:] != (H, W):
                hd = F.interpolate(hd[:, None], size=(H, W), mode="nearest")[:, 0]
            head_np = hd.cpu().numpy() > 0                 # (B, H, W)
            if head_np.shape[0] == 1 and B > 1:
                head_np = np.repeat(head_np, B, 0)
        elif person_mask is not None:
            pm = person_mask
        else:
            raise ValueError("Sapiens2NormalCrops: connect 'labels' (Sapiens2 Segmentation) or a 'person_mask'.")
        if pm.shape[1:] != (H, W):
            pm = F.interpolate(pm[:, None].float(), size=(H, W), mode="nearest")[:, 0]
        pm_np = pm.cpu().numpy() > 0.5
        if pm_np.shape[0] == 1 and B > 1:
            pm_np = np.repeat(pm_np, B, 0)
        boxes = compute_crops(pm_np.any(0), W, H, padding, head_pass, head_fraction,
                              head_union=None if head_np is None else head_np.any(0))
        n_body = compute_crops.last_n_body
        # per-frame box lists: bodies are fixed over the batch, heads follow the head when labels are known
        frame_boxes = [list(boxes) for _ in range(B)]
        if head_pass and head_np is not None:
            tracks = track_heads(compute_crops.last_comps, head_np, W, H, padding * 2)
            fallback = boxes[n_body:]
            for i in range(B):
                if len(fallback) == len(tracks):
                    heads_i = [t[i] if t is not None else fb for t, fb in zip(tracks, fallback)]
                else:                                   # can't pair them safely: tracked heads only
                    heads_i = [t[i] for t in tracks if t is not None]
                frame_boxes[i] = boxes[:n_body] + heads_i
        print("[NukeBridge] Sapiens2NormalCrops: %d crops (%d body, %d head), frame 0: %s"
              % (len(frame_boxes[0]), n_body, len(frame_boxes[0]) - n_body, frame_boxes[0]))
        _alpha_cache = {}

        def alpha_of(b):
            key = tuple(b)
            if key not in _alpha_cache:
                _alpha_cache[key] = torch.from_numpy(_feather(b, W, H, feather_px))
            return _alpha_cache[key]

        target = torch.device(model.device) if getattr(model, "device", None) is not None else torch.device("cuda")
        if target.type == "cuda":
            model.model.to(target)
        out, tiles = [], []
        try:
            for i in range(B):
                acc = torch.zeros(3, H, W)
                acc[2] = 1e-4  # neutral where no crop lands
                crops_i = []
                for b in frame_boxes[i]:
                    x0, y0, x1, y1 = b
                    a = alpha_of(b)
                    n = _infer(model, image[i, y0:y1, x0:x1, :3])
                    crops_i.append((image[i, y0:y1, x0:x1, :3], n))
                    region = acc[:, y0:y1, x0:x1]
                    acc[:, y0:y1, x0:x1] = region * (1 - a) + n * a
                acc = acc / torch.norm(acc, dim=0, keepdim=True).clamp(min=1e-8)
                out.append(((acc + 1) * 0.5).permute(1, 2, 0))
                tiles.append(_tile_sheet(crops_i))
        finally:
            if target.type == "cuda":
                model.model.to("cpu")
                torch.cuda.empty_cache()

        soft = _soft_mask(torch.from_numpy(pm_np.astype(np.float32)), mask_erode_px, mask_blur_px)
        boxes_img = torch.stack([_draw_boxes(image[i, ..., :3], frame_boxes[i], n_body, pm_np[i]) for i in range(B)], 0)
        return (torch.stack(out, 0).clamp(0, 1), soft, boxes_img, torch.stack(tiles, 0))


class Sapiens2NormalTiles:
    """Debug / comparison: Sapiens2 normals on the WHOLE frame at native resolution, using overlapping
    3:4 tiles (no person mask). Shows what Sapiens2 does on the background."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("SAPIENS2_MODEL",),
            "image": ("IMAGE",),
            "tile_height": ("INT", {"default": 0, "min": 0, "max": 4096,
                                    "tooltip": "0 = full image height. Smaller = more tiles, more resolution."}),
            "overlap": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.75, "step": 0.05}),
            "feather_px": ("INT", {"default": 48, "min": 1, "max": 512}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("normal_map", "debug_tiles")
    FUNCTION = "run"
    CATEGORY = "NukeBridge/debug"

    def run(self, model, image, tile_height=0, overlap=0.25, feather_px=48):
        B, H, W = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        th = H if tile_height <= 0 else min(tile_height, H)
        tw = min(W, int(round(th * ASPECT)))

        def starts(total, size):
            if size >= total:
                return [0]
            n = int(math.ceil((total - size) / (size * (1 - overlap)))) + 1
            return [int(round(k * (total - size) / (n - 1))) for k in range(n)]

        boxes = [(x, y, x + tw, y + th) for y in starts(H, th) for x in starts(W, tw)]
        print("[NukeBridge] Sapiens2NormalTiles: %d tiles of %dx%d" % (len(boxes), tw, th))
        alphas = [torch.from_numpy(_feather(b, W, H, feather_px)) for b in boxes]
        target = torch.device(model.device) if getattr(model, "device", None) is not None else torch.device("cuda")
        if target.type == "cuda":
            model.model.to(target)
        out, dbg = [], []
        try:
            for i in range(B):
                acc = torch.zeros(3, H, W)
                acc[2] = 1e-4
                for b, a in zip(boxes, alphas):
                    x0, y0, x1, y1 = b
                    n = _infer(model, image[i, y0:y1, x0:x1, :3])
                    acc[:, y0:y1, x0:x1] = acc[:, y0:y1, x0:x1] * (1 - a) + n * a
                acc = acc / torch.norm(acc, dim=0, keepdim=True).clamp(min=1e-8)
                out.append(((acc + 1) * 0.5).permute(1, 2, 0))
                dbg.append(_draw_boxes(image[i, ..., :3], boxes, len(boxes), np.zeros((H, W), bool)))
        finally:
            if target.type == "cuda":
                model.model.to("cpu")
                torch.cuda.empty_cache()
        return (torch.stack(out, 0).clamp(0, 1), torch.stack(dbg, 0))
