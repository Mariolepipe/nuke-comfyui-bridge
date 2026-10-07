"""
Motion-compensated temporal stabilization for per-frame passes (normals, ...).

For every frame t, the neighbours t-r..t+r are warped onto t with optical flow
(RAFT from torchvision, or OpenCV DIS as fallback), then averaged with weights:
  - how well the warped plate matches the plate at t (occlusions / bad flow -> ~0)
  - a gaussian falloff in time
Detail that is really in the image lines up after warping and survives; per-frame
noise ("crepitement") does not line up and averages out. Normals are renormalized.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

_RAFT = {}


def _raft(device):
    if "m" not in _RAFT:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        _RAFT["m"] = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).eval()
    return _RAFT["m"].to(device)


def _flow_raft(a, b, device):
    """a, b: (1,3,h,w) 0..1, h,w multiple of 8 -> flow a->b (1,2,h,w) in pixels."""
    m = _raft(device)
    with torch.inference_mode():
        return m(a * 2 - 1, b * 2 - 1, num_flow_updates=12)[-1]


def _flow_dis(a, b):
    import cv2
    g = lambda t: (t[0].mean(0).cpu().numpy() * 255).astype(np.uint8)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f = dis.calc(g(a), g(b), None)
    return torch.from_numpy(f).permute(2, 0, 1)[None].to(a.device)


def _warp(img, flow):
    """backward warp: out(x) = img(x + flow(x)). img (1,C,H,W), flow (1,2,H,W) px."""
    _, _, H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device, dtype=torch.float32),
                            torch.arange(W, device=img.device, dtype=torch.float32), indexing="ij")
    gx = (xx + flow[0, 0]) / (W - 1) * 2 - 1
    gy = (yy + flow[0, 1]) / (H - 1) * 2 - 1
    grid = torch.stack((gx, gy), -1)[None]
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
    inside = ((gx.abs() <= 1) & (gy.abs() <= 1)).float()[None, None]
    return out, inside


def stabilize(values, guide, radius=2, sigma=0.04, flow_width=1024, normals=True, log=print, max_angle=15.0):
    """
    values : (B,H,W,C) float numpy/torch, the pass to stabilize (normals in -1..1)
    guide  : (B,H,W,3) float 0..1, the plate (what the flow is computed on)
    radius : neighbours on each side (2 = 5-frame window)
    sigma  : plate mismatch tolerance; lower = stricter (keeps more flicker, fewer ghosts)
    max_angle : normals only - a warped neighbour that disagrees with the current frame by much more
                than this (degrees) is ignored there: flicker is a few degrees, a ghost (wrong arm
                position after a bad flow) is tens of degrees. 0 = off.
    """
    v = torch.as_tensor(np.asarray(values, np.float32)) if not torch.is_tensor(values) else values.float()
    gd = torch.as_tensor(np.asarray(guide, np.float32)) if not torch.is_tensor(guide) else guide.float()
    B, H, W, C = v.shape
    if radius <= 0 or B < 2:
        return v.numpy()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # flow resolution: multiple of 8, ~flow_width wide
    s = min(1.0, flow_width / float(W))
    fw, fh = max(8, int(round(W * s / 8)) * 8), max(8, int(round(H * s / 8)) * 8)
    sx, sy = W / float(fw), H / float(fh)
    small = F.interpolate(gd[..., :3].permute(0, 3, 1, 2), size=(fh, fw), mode="bilinear",
                          align_corners=False).to(dev)
    # blurred plate luminance for the match test (grain-insensitive)
    lum = gd[..., :3].mean(-1)[:, None]
    lum = F.avg_pool2d(lum, 5, 1, 2, count_include_pad=False).to(dev)

    use_raft = True
    try:
        _raft(dev)
    except Exception as e:  # no weights / no internet -> OpenCV
        log("[NukeBridge] RAFT unavailable (%s), using OpenCV DIS flow" % e)
        use_raft = False

    out = torch.empty_like(v)
    for t in range(B):
        vt = v[t].permute(2, 0, 1)[None].to(dev)
        acc = vt.clone()
        wsum = torch.ones(1, 1, H, W, device=dev)
        for k in range(-radius, radius + 1):
            j = t + k
            if k == 0 or j < 0 or j >= B:
                continue
            a, b = small[t:t + 1], small[j:j + 1]
            fl = _flow_raft(a, b, dev) if use_raft else _flow_dis(a, b)
            fl = F.interpolate(fl, size=(H, W), mode="bilinear", align_corners=False)
            fl = torch.cat((fl[:, :1] * sx, fl[:, 1:] * sy), 1)
            vj, inside = _warp(v[j].permute(2, 0, 1)[None].to(dev), fl)
            lj, _ = _warp(lum[j:j + 1], fl)
            err = (lj - lum[t:t + 1]).abs()
            w = torch.exp(-(err / sigma) ** 2) * inside * math.exp(-0.5 * (k / max(radius, 1)) ** 2)
            if normals and max_angle > 0:
                # angle between unit normals from 1 - cos (cheap, no acos): ang^2 ~= 2 (1 - cos)
                cosang = (vj * vt).sum(1, keepdim=True)
                ang2 = (2.0 * (1.0 - cosang)).clamp(min=0) * (180.0 / math.pi) ** 2
                w = w * torch.exp(-ang2 / (max_angle * max_angle))
            acc += vj * w
            wsum += w
        r = acc / wsum
        if normals:
            r = r / torch.norm(r, dim=1, keepdim=True).clamp(min=1e-6)
        out[t] = r[0].permute(1, 2, 0).cpu()
    if dev.type == "cuda" and "m" in _RAFT:
        _RAFT["m"].to("cpu")
        torch.cuda.empty_cache()
    return out.numpy()


class TemporalStabilizeNormals:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_map": ("IMAGE", {"tooltip": "0..1 encoded normals, one per frame."}),
                "plate": ("IMAGE", {"tooltip": "The source frames (optical flow is computed on them)."}),
                "radius": ("INT", {"default": 2, "min": 1, "max": 6,
                                   "tooltip": "Frames on each side. 2 = 5-frame window."}),
                "sigma": ("FLOAT", {"default": 0.04, "min": 0.005, "max": 0.5, "step": 0.005,
                                    "tooltip": "Plate mismatch tolerance. Lower = fewer ghosts, less smoothing."}),
                "max_angle": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 90.0, "step": 0.5,
                                        "tooltip": "Anti-ghost: ignore neighbours whose normal differs more than this "
                                                   "(degrees). 0 = off."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "NukeBridge"
    DESCRIPTION = "Motion-compensated temporal average (optical flow) of a per-frame normal pass."

    def run(self, normal_map, plate, radius=2, sigma=0.04, max_angle=15.0):
        n = normal_map[..., :3].float().cpu().numpy() * 2 - 1
        p = plate[..., :3].float().cpu()
        if p.shape[1:3] != n.shape[1:3]:
            p = F.interpolate(p.permute(0, 3, 1, 2), size=n.shape[1:3], mode="bilinear",
                              align_corners=False).permute(0, 2, 3, 1)
        s = stabilize(n, p, radius, sigma, max_angle=max_angle)
        return (torch.from_numpy((s + 1) * 0.5).clamp(0, 1),)
