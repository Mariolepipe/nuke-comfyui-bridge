"""
Helpers to run the LTX-2.5 Alpha Gen IC-LoRA on a plate of any size / length.

LTX wants: width and height multiples of 32, at most 1920x1088, and 8n+1 frames.
  LTXAlphaPrep   : fit the plate inside the limits (no stretch), pad to multiples of 32,
                   repeat the last frame up to 8n+1. Outputs the size / length for the latent.
  LTXAlphaFinish : remove the padding, resize the matte back to the plate size, drop the
                   extra frames. Outputs a MASK (for SaveMoGeEXR alpha_mask) and a preview.
"""

import torch
import torch.nn.functional as F


class LTXAlphaPrep:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "max_width": ("INT", {"default": 1920, "min": 256, "max": 1920, "step": 32}),
            "max_height": ("INT", {"default": 1088, "min": 256, "max": 1088, "step": 32}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "LTX_PAD_INFO")
    RETURN_NAMES = ("images", "width", "height", "length", "pad_info")
    FUNCTION = "run"
    CATEGORY = "NukeBridge"
    DESCRIPTION = "Fit + pad a plate to LTX constraints (multiple of 32, <=1920x1088, 8n+1 frames)."

    def run(self, images, max_width=1920, max_height=1088):
        B, H, W, C = images.shape
        s = min(1.0, max_width / W, max_height / H)
        w, h = max(32, int(round(W * s))), max(32, int(round(H * s)))
        tw, th = (w + 31) // 32 * 32, (h + 31) // 32 * 32
        tw, th = min(tw, max_width), min(th, max_height)
        w, h = min(w, tw), min(h, th)
        x = images[..., :3].permute(0, 3, 1, 2)
        if (h, w) != (H, W):
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        px, py = (tw - w) // 2, (th - h) // 2
        x = F.pad(x, (px, tw - w - px, py, th - h - py), mode="replicate")   # edge pixels, not black
        n = ((B - 1 + 7) // 8) * 8 + 1                                       # next 8n+1
        if n > B:
            x = torch.cat([x, x[-1:].expand(n - B, -1, -1, -1)], 0)
        info = {"orig": (B, H, W), "box": (px, py, w, h)}
        return (x.permute(0, 2, 3, 1).contiguous(), tw, th, n, info)


class LTXAlphaFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "matte": ("IMAGE", {"tooltip": "Decoded output of the Alpha Gen pass."}),
            "pad_info": ("LTX_PAD_INFO",),
        }}

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("alpha", "preview")
    FUNCTION = "run"
    CATEGORY = "NukeBridge"
    DESCRIPTION = "Crop the padding, resize the LTX matte back to the plate, trim to the plate's frame count."

    def run(self, matte, pad_info):
        B, H, W = pad_info["orig"]
        px, py, w, h = pad_info["box"]
        m = matte[..., :3].mean(-1)[:B, py:py + h, px:px + w]
        if m.shape[0] < B:                                   # should not happen, keep frame count safe
            m = torch.cat([m, m[-1:].expand(B - m.shape[0], -1, -1)], 0)
        if (h, w) != (H, W):
            m = F.interpolate(m[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        m = m.clamp(0, 1).float()
        return (m, m[..., None].expand(-1, -1, -1, 3).contiguous())
