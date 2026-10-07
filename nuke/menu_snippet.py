# Paste at the end of ~/.nuke/menu.py, and set BRIDGE_ROOT to the "nuke" folder of this repository
# (the folder that contains "comfy_bridge").
import sys
import nuke

BRIDGE_ROOT = "/path/to/nuke-comfyui-bridge/nuke"
if BRIDGE_ROOT not in sys.path:
    sys.path.append(BRIDGE_ROOT)


def _comfy_panel():
    import comfy_bridge.nuke_panel
    comfy_bridge.nuke_panel.show()


nuke.menu("Nuke").addMenu("ComfyUI").addCommand("Generate passes...", _comfy_panel, "ctrl+alt+g")
