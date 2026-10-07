"""
ComfyUI-NukeBridge - Mario Falco
Save MoGe geometry as a multi-channel float EXR ready for Nuke.

Channels (Nuke camera convention: +X right, +Y up, camera looks down -Z):
  depth.Z   camera-space Z distance, metric (MoGe-2/3), positive values  -> ZDefocus "direct"
  N.R/G/B   camera-space normals, -1..1 (facing camera = +Z)             -> NormalLight / ReLight
            = detail_normals (Sapiens2) inside detail_mask + MoGe elsewhere, optical-flow stabilized,
              when no normals_override is given
  P.R/G/B   camera-space position                                       -> PositionToPoints / ReLight
  Nmoge.R/G/B  MoGe normals, written only when normals_override is used (detail transfer)
  Ndetail.R/G/B  detail source for NormalDetail: detail_normals (e.g. Sapiens2, humans) inside
               detail_mask, MoGe normals elsewhere; axis signs auto-aligned to N
  depth_moge.Z MoGe depth, written only when depth_override is used (P is then rebuilt
               from the override depth + MoGe intrinsics, so depth.Z and P always agree)
  A         person matte: Sapiens2 segmentation (person_labels, any person part) or alpha_mask;
            without either, the MoGe validity mask (then written as moge_valid.A with comparison layers)
Header    : moge/fov_x, moge/fov_y (degrees), moge/focal_px
"""

import math
import os

import numpy as np
import torch

import folder_paths

from .exr_writer import write_exr, HALF, FLOAT

# MoGe is OpenCV (X right, Y down, Z forward) -> Nuke/OpenGL camera space
_CV_TO_GL = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def _np(t):
    return t.detach().float().cpu().numpy()


def _resize_hw(a, H, W):
    """Bilinear resize of an (h, w) or (h, w, c) float array to (H, W)."""
    if a.shape[:2] == (H, W):
        return a.astype(np.float32)
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    t = t[None, None] if t.ndim == 2 else t.permute(2, 0, 1)[None]
    t = torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0]
    return (t[0] if a.ndim == 2 else t.permute(1, 2, 0)).numpy()


class SaveMoGeEXR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moge_geometry": ("MOGE_GEOMETRY",),
                "filename_prefix": ("STRING", {"default": "nuke_geo"}),
                "half_float_normals": ("BOOLEAN", {"default": True,
                                       "tooltip": "Store N and A as 16-bit half (depth and P stay 32-bit)."}),
            },
            "optional": {
                "normals_override": ("IMAGE", {"tooltip": "0..1 encoded normal maps (e.g. NormalCrafter) "
                                               "written to N instead of MoGe's. Same batch size."}),
                "override_flip_x": ("BOOLEAN", {"default": False}),
                "override_flip_y": ("BOOLEAN", {"default": False}),
                "override_flip_z": ("BOOLEAN", {"default": False}),
                "detail_normals": ("IMAGE", {"tooltip": "0..1 encoded high-detail normals (e.g. Sapiens2 on "
                                             "people). Axis signs are auto-aligned to N. Written to Ndetail."}),
                "detail_mask": ("MASK", {"tooltip": "Where detail_normals are valid (e.g. Sapiens2 person mask). "
                                         "MoGe normals are used outside it."}),
                "depth_override": ("DEPTHS", {"tooltip": "Metric depth in metres per frame (e.g. Video "
                                              "Depth Anything metric). Written to depth.Z; P is rebuilt from it."}),
                "comparison_layers": ("BOOLEAN", {"default": True,
                                      "tooltip": "Also write the raw passes (Nmoge, Nraw, depth_moge.Z) for A/B checks."}),
                "stabilize_radius": ("INT", {"default": 0, "min": 0, "max": 6,
                                     "tooltip": "Detail normals: optical-flow temporal average over +-N frames "
                                                "(0 = off, 2 = 5-frame window)."}),
                "stabilize_sigma": ("FLOAT", {"default": 0.04, "min": 0.005, "max": 0.5, "step": 0.005,
                                    "tooltip": "Plate mismatch tolerance. Lower = fewer ghosts, less smoothing."}),
                "plate": ("IMAGE", {"tooltip": "Source frames, used for the optical flow of the stabilization."}),
                "stabilize_max_angle": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 90.0, "step": 0.5,
                                        "tooltip": "Anti-ghost: a neighbour frame whose normal differs more than this "
                                                   "(degrees) is ignored there. Lower = fewer ghosts. 0 = off."}),
                "person_labels": ("SAPIENS2_LABELS", {"tooltip": "Sapiens2 segmentation labels: every person pixel "
                                                      "(any body part) is written to A."}),
                "use_alpha_mask": ("BOOLEAN", {"default": True,
                                   "tooltip": "On: A = alpha_mask (e.g. the LTX Alpha Gen matte). Off: A = Sapiens2 "
                                              "segmentation, and the alpha_mask branch is not computed at all."}),
                "alpha_mask": ("MASK", {"lazy": True,
                                        "tooltip": "Any mask to write to A (used instead of person_labels)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("N_final", "N_before_stab", "N_moge", "sapiens_mask")
    FUNCTION = "save"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # always write: never hand back cached files from a previous run
        return float("nan")

    OUTPUT_NODE = True
    CATEGORY = "image/geometry estimation"

    def check_lazy_status(self, use_alpha_mask=True, **kwargs):
        # only ask ComfyUI to compute the (slow) matte branch when it is going to be used.
        # An unconnected input is absent from kwargs; a connected one not computed yet is None.
        if use_alpha_mask and "alpha_mask" in kwargs and kwargs["alpha_mask"] is None:
            return ["alpha_mask"]
        return []
    DESCRIPTION = "Write MoGe depth / normals / position / mask as one float EXR per frame (Nuke conventions)."

    def save(self, moge_geometry, filename_prefix, half_float_normals=True, normals_override=None,
             override_flip_x=False, override_flip_y=False, override_flip_z=False, depth_override=None,
             detail_normals=None, detail_mask=None, comparison_layers=True,
             stabilize_radius=0, stabilize_sigma=0.04, plate=None,
             stabilize_max_angle=15.0, person_labels=None, alpha_mask=None, use_alpha_mask=True):
        g = moge_geometry
        matte = alpha_mask if use_alpha_mask else None
        if matte is None and isinstance(person_labels, dict) and "class_ids" in person_labels:
            matte = (torch.as_tensor(person_labels["class_ids"]) > 0).float()
            if matte.ndim == 2:
                matte = matte[None]
        ref = next(g[k] for k in ("depth", "points", "image") if k in g)
        B, H, W = int(ref.shape[0]), int(ref.shape[1]), int(ref.shape[2])
        out_dir, fname, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), W, H)
        npt = HALF if half_float_normals else FLOAT
        if normals_override is not None and int(normals_override.shape[0]) != B:
            raise ValueError("normals_override has %d frames, geometry has %d" % (normals_override.shape[0], B))
        flip = np.array([-1.0 if override_flip_x else 1.0, -1.0 if override_flip_y else 1.0,
                         -1.0 if override_flip_z else 1.0], dtype=np.float32)

        if depth_override is not None:
            depth_override = _np(depth_override) if torch.is_tensor(depth_override) \
                else np.asarray(depth_override, dtype=np.float32)
            if depth_override.ndim == 4:
                depth_override = depth_override[..., 0]
            if depth_override.shape[0] != B:
                raise ValueError("depth_override has %d frames, geometry has %d" % (depth_override.shape[0], B))
        if detail_normals is not None and int(detail_normals.shape[0]) != B:
            raise ValueError("detail_normals has %d frames, geometry has %d" % (detail_normals.shape[0], B))
        detail_signs = None

        # ---- normals for the whole batch (temporal stabilization needs every frame) ----
        def _override(i):
            n = _resize_hw(_np(normals_override[i][..., :3]) * 2.0 - 1.0, H, W) * flip
            return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)

        def _moge_n(i):
            nm = _np(g["normal"][i]) * _CV_TO_GL
            return _resize_hw(np.where(np.isfinite(nm), nm, 0.0), H, W)

        detail_raw = detail_stab = None
        blend_masks = []
        if detail_normals is not None:
            detail_raw = np.empty((B, H, W, 3), np.float32)
            for i in range(B):
                nd = _resize_hw(_np(detail_normals[i][..., :3]) * 2.0 - 1.0, H, W)
                m = np.ones((H, W), np.float32) if detail_mask is None else \
                    _resize_hw(_np(detail_mask[min(i, detail_mask.shape[0] - 1)]).astype(np.float32), H, W)
                m = np.clip(m, 0.0, 1.0)
                blend_masks.append(m)
                base = _moge_n(i) if "normal" in g else (_override(i) if normals_override is not None else None)
                ref = _override(i) if normals_override is not None else base
                if detail_signs is None and ref is not None:
                    # pick each axis sign so the detail normals agree with the reference normals
                    sel = m > 0.5
                    if sel.sum() < 100:
                        sel = np.ones((H, W), bool)
                    corr = (nd[sel] * ref[sel]).sum(axis=0)
                    detail_signs = np.where(corr < 0, -1.0, 1.0).astype(np.float32)
                if detail_signs is not None:
                    nd = nd * detail_signs
                mix = nd if base is None else nd * m[..., None] + base * (1.0 - m[..., None])
                detail_raw[i] = mix / np.maximum(np.linalg.norm(mix, axis=-1, keepdims=True), 1e-6)
            detail_stab = detail_raw
            if stabilize_radius > 0 and B > 1:
                guide = plate if plate is not None else g.get("image")
                if guide is None:
                    print("[NukeBridge] stabilize: no plate / image, skipped")
                else:
                    guide = guide[..., :3].float().cpu()
                    if tuple(guide.shape[1:3]) != (H, W):
                        guide = torch.nn.functional.interpolate(guide.permute(0, 3, 1, 2), size=(H, W),
                                                                mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                    from .temporal import stabilize
                    detail_stab = stabilize(detail_raw, guide, stabilize_radius, stabilize_sigma,
                                            max_angle=stabilize_max_angle)

        uu, vv = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5, np.arange(H, dtype=np.float32) + 0.5)

        files = []
        prev = {"N": [], "raw": [], "moge": []}
        for i in range(B):
            ch = {}
            mask = _np(g["mask"][i]).astype(np.float32) if "mask" in g else np.ones((H, W), np.float32)

            if "depth" in g:
                z = _np(g["depth"][i])
                valid = np.isfinite(z) & (z > 0)
                ch["depth.Z"] = (np.where(valid, z, 0.0).astype(np.float32), FLOAT)
                mask = mask * valid

            if depth_override is not None:
                d = depth_override[i].astype(np.float32)
                if d.shape != (H, W):
                    t = torch.from_numpy(d)[None, None]
                    d = torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear",
                                                        align_corners=False)[0, 0].numpy()
                dvalid = np.isfinite(d) & (d > 0)
                d = np.where(dvalid, d, 0.0).astype(np.float32)
                if "depth.Z" in ch:
                    ch["depth_moge.Z"] = ch["depth.Z"]
                ch["depth.Z"] = (d, FLOAT)
                mask = mask * dvalid

            if depth_override is not None and "intrinsics" in g:
                # pinhole unprojection with MoGe's (fixed-FOV) intrinsics, OpenCV -> Nuke
                K = _np(g["intrinsics"][i])
                fx, fy = float(K[0, 0]) * W, float(K[1, 1]) * H
                cx, cy = float(K[0, 2]) * W, float(K[1, 2]) * H
                d = ch["depth.Z"][0]
                pc = np.stack(((uu - cx) / fx * d, (vv - cy) / fy * d, d), axis=-1) * _CV_TO_GL
                for k, c in enumerate("RGB"):
                    ch["P." + c] = (pc[..., k].astype(np.float32), FLOAT)
            elif "points" in g:
                p = _np(g["points"][i]) * _CV_TO_GL
                p = np.where(np.isfinite(p), p, 0.0)
                for k, c in enumerate("RGB"):
                    ch["P." + c] = (p[..., k], FLOAT)

            def put(layer, n):
                for k, c in enumerate("RGB"):
                    ch["%s.%s" % (layer, c)] = (n[..., k].astype(np.float32), npt)

            if normals_override is not None:           # N = video model (e.g. NormalCrafter)
                put("N", _override(i))
                if detail_stab is not None:
                    put("Ndetail", detail_stab[i])
                elif "normal" in g:
                    put("Nmoge", _moge_n(i))           # detail source for NormalDetail
            elif detail_stab is not None:              # N = detail normals (Sapiens2 + MoGe), stabilized
                put("N", detail_stab[i])
                if comparison_layers and detail_stab is not detail_raw:
                    put("Nraw", detail_raw[i])
            elif "normal" in g:
                put("N", _moge_n(i))
            if comparison_layers and "normal" in g and "Nmoge.R" not in ch:
                put("Nmoge", _moge_n(i))

            if matte is not None:
                ch["A"] = (np.clip(_resize_hw(_np(matte[min(i, matte.shape[0] - 1)]).astype(np.float32), H, W),
                                   0.0, 1.0), npt)
                if comparison_layers:
                    ch["moge_valid.A"] = (mask.astype(np.float32), npt)
            else:
                ch["A"] = (mask.astype(np.float32), npt)
            # previews for the debug workflow (0..1 encoded, like any normal map)
            nf = np.stack([ch["N." + c][0] for c in "RGB"], -1) if "N.R" in ch else np.zeros((H, W, 3), np.float32)
            prev["N"].append(nf)
            prev["raw"].append(detail_raw[i] if detail_raw is not None else nf)
            prev["moge"].append(_moge_n(i) if "normal" in g else nf)

            attrs = {"moge/source": "ComfyUI-NukeBridge SaveMoGeEXR v11"}
            if detail_signs is not None:
                attrs["detail/axis_signs"] = "%+d %+d %+d" % tuple(int(v) for v in detail_signs)
            if "depth_moge.Z" in ch:
                a, b = ch["depth.Z"][0], ch["depth_moge.Z"][0]
                both = (a > 0) & (b > 0)
                if both.any():
                    attrs["depth/override_over_moge"] = float(np.median(a[both] / b[both]))
            if "intrinsics" in g:
                K = _np(g["intrinsics"][i])
                attrs["moge/fov_x"] = math.degrees(2.0 * math.atan(0.5 / float(K[0, 0])))
                attrs["moge/fov_y"] = math.degrees(2.0 * math.atan(0.5 / float(K[1, 1])))
                attrs["moge/focal_px"] = float(K[0, 0]) * W

            if not comparison_layers and "depth_moge.Z" in ch:
                del ch["depth_moge.Z"]
            if detail_stab is not None and detail_stab is not detail_raw:
                attrs["detail/stabilize_radius"] = float(stabilize_radius)
                attrs["detail/stabilize_max_angle"] = float(stabilize_max_angle)

            name = "%s_%05d_.exr" % (fname, counter)
            write_exr(os.path.join(out_dir, name), ch, attrs)
            files.append({"filename": name, "subfolder": subfolder, "type": "output",
                          "fov_x": round(attrs.get("moge/fov_x", 0.0), 3)})
            counter += 1

        enc = lambda lst: torch.from_numpy(np.clip((np.stack(lst, 0) + 1.0) * 0.5, 0, 1).astype(np.float32))
        bm = torch.from_numpy(np.stack(blend_masks, 0)) if blend_masks else torch.zeros(B, H, W)
        # "files" (not "images") so the frontend does not try to preview an EXR
        return {"ui": {"files": files}, "result": (enc(prev["N"]), enc(prev["raw"]), enc(prev["moge"]), bm)}


def _font(size):
    from PIL import ImageFont
    for name in ("arialbd.ttf", "arial.ttf", "segoeui.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _legend(im, report, max_deg):
    """big readable report + gradient bar with degree ticks, top-left on a dark box."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    s = max(18, im.height // 28)
    f, fs = _font(s), _font(int(s * 0.75))
    x, y, bw, bh = int(s * 0.8), int(s * 0.6), int(s * 16), int(s * 0.7)
    d.rectangle((x - s // 2, y - s // 3, x + bw + s // 2, y + int(s * 3.6)), fill=(0, 0, 0))
    d.text((x, y), report, font=f, fill=(255, 255, 255))
    by = y + int(s * 1.5)
    for i in range(bw):
        t = i / (bw - 1)
        c = (int(255 * min(1, t * 3)), int(255 * min(1, max(0, t * 3 - 1))), int(255 * min(1, max(0, t * 3 - 2))))
        d.line((x + i, by, x + i, by + bh), fill=c)
    d.rectangle((x, by, x + bw, by + bh), outline=(120, 120, 120))
    for t in (0.0, 1 / 3, 2 / 3, 1.0):
        tx = x + int(t * (bw - 1))
        d.line((tx, by + bh, tx, by + bh + s // 4), fill=(200, 200, 200))
        lab = "%g\u00b0%s" % (round(t * max_deg, 1), "+" if t == 1.0 else "")
        d.text((tx, by + bh + s // 3), lab, font=fs, fill=(220, 220, 220), anchor="ma")


class NormalAngleDiff:
    """Debug: angle between two normal maps. Black = identical -> red -> yellow -> white = max_degrees or more."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"normals_a": ("IMAGE",), "normals_b": ("IMAGE",),
                             "max_degrees": ("FLOAT", {"default": 20.0, "min": 0.5, "max": 180.0, "step": 0.5})},
                "optional": {"mask": ("MASK", {"tooltip": "Only measure inside this mask (shown dimmed outside)."})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("heatmap", "report")
    FUNCTION = "run"
    CATEGORY = "NukeBridge/debug"
    OUTPUT_NODE = True

    def run(self, normals_a, normals_b, max_degrees=20.0, mask=None):
        from PIL import Image, ImageDraw
        a = normals_a[..., :3].float() * 2 - 1
        b = normals_b[..., :3].float() * 2 - 1
        if b.shape[1:3] != a.shape[1:3]:
            b = torch.nn.functional.interpolate(b.permute(0, 3, 1, 2), size=a.shape[1:3], mode="bilinear",
                                                align_corners=False).permute(0, 2, 3, 1)
        a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ang = torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1, 1)))           # B,H,W degrees
        m = torch.ones_like(ang) if mask is None else \
            torch.nn.functional.interpolate(mask[:, None].float(), size=ang.shape[1:], mode="bilinear")[:, 0]
        sel = m > 0.5
        mean = float(ang[sel].mean()) if sel.any() else float("nan")
        vals = ang[sel].numpy()
        p95 = float(np.percentile(vals[::max(1, vals.size // 2000000)], 95)) if vals.size > 10 else float("nan")
        report = "mean %.2f deg | 95%% < %.2f deg (inside mask)" % (mean, p95)
        print("[NukeBridge] NormalAngleDiff:", report)
        heat = (ang / max_degrees).clamp(0, 1)
        # classic heat ramp: black -> red -> yellow -> white
        rgb = torch.stack(((heat * 3).clamp(0, 1), (heat * 3 - 1).clamp(0, 1), (heat * 3 - 2).clamp(0, 1)), -1)
        inside = (m > 0.5).float()[..., None]
        rgb = rgb * inside + 0.12 * (1 - inside)                                   # outside mask: flat grey
        out = []
        for k in range(rgb.shape[0]):
            im = Image.fromarray((rgb[k].numpy() * 255).astype(np.uint8))
            _legend(im, report, max_degrees)
            out.append(torch.from_numpy(np.asarray(im).astype(np.float32) / 255))
        return {"ui": {"text": [report]}, "result": (torch.stack(out, 0), report)}


class PreviewAndSave:
    """Preview in the UI like PreviewImage, and also write the frames as PNGs to <folder>/<run_tag>/."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "run_tag": ("STRING", {"forceInput": True}),
            "name": ("STRING", {"default": "step"}),
            "folder": ("STRING", {"default": "", "tooltip": "Empty = ComfyUI/output/nukebridge_debug"}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "NukeBridge/debug"

    def run(self, images, run_tag, name, folder):
        import uuid
        from PIL import Image
        tag = "".join(c if c.isalnum() or c in "-_." else "_" for c in (run_tag or "run")) or "run"
        folder = folder.strip() or os.path.join(folder_paths.get_output_directory(), "nukebridge_debug")
        out_dir = os.path.join(folder, tag)
        os.makedirs(out_dir, exist_ok=True)
        tmp = folder_paths.get_temp_directory()
        os.makedirs(tmp, exist_ok=True)
        key = uuid.uuid4().hex[:8]
        ui = []
        for i, img in enumerate(images):
            im = Image.fromarray((img[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            im.save(os.path.join(out_dir, "%s_%04d.png" % (name, i)))
            fn = "nb_%s_%s_%04d.png" % (name, key, i)
            im.save(os.path.join(tmp, fn), compress_level=1)
            ui.append({"filename": fn, "subfolder": "", "type": "temp"})
        print("[NukeBridge] PreviewAndSave: %d frame(s) -> %s" % (len(images), out_dir))
        return {"ui": {"images": ui}}


from .sapiens_crops import Sapiens2NormalCrops, Sapiens2NormalTiles
from .temporal import TemporalStabilizeNormals
from .ltx_alpha import LTXAlphaPrep, LTXAlphaFinish

NODE_CLASS_MAPPINGS = {"SaveMoGeEXR": SaveMoGeEXR, "Sapiens2NormalCrops": Sapiens2NormalCrops,
                       "TemporalStabilizeNormals": TemporalStabilizeNormals, "NormalAngleDiff": NormalAngleDiff,
                       "PreviewAndSave": PreviewAndSave,
                       "Sapiens2NormalTiles": Sapiens2NormalTiles,
                       "LTXAlphaPrep": LTXAlphaPrep, "LTXAlphaFinish": LTXAlphaFinish}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveMoGeEXR": "Save MoGe EXR (Nuke)",
                              "LTXAlphaPrep": "LTX Alpha: prep plate (fit, pad, 8n+1)",
                              "LTXAlphaFinish": "LTX Alpha: finish matte (unpad, resize)",
                              "Sapiens2NormalCrops": "Sapiens2 Normal (person crops)",
                              "TemporalStabilizeNormals": "Temporal Stabilize Normals (optical flow)",
                              "NormalAngleDiff": "Normal Angle Diff (debug)",
                              "PreviewAndSave": "Preview + Save PNG (debug)",
                              "Sapiens2NormalTiles": "Sapiens2 Normal (full frame tiles, debug)"}
