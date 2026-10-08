# Nuke x ComfyUI bridge: paste this at the end of your menu.py
#   Windows: C:\Users\<you>\.nuke\menu.py      macOS / Linux: ~/.nuke/menu.py
# (create the file if it doesn't exist), then restart Nuke.
import os
import sys
import nuke

# 1. The "nuke" folder of this repository (the one that contains "comfy_bridge").
#    On Windows use forward slashes, or keep the r"" prefix: r"C:\tools\nuke-comfyui-bridge\nuke"
BRIDGE_ROOT = "/path/to/nuke-comfyui-bridge/nuke"

# 2. The address of your ComfyUI, as shown in the browser tab when ComfyUI is open.
#    ComfyUI started from the command line uses port 8188; the ComfyUI Desktop app often uses 8000.
os.environ.setdefault("COMFY_BRIDGE_URL", "http://127.0.0.1:8188")

if BRIDGE_ROOT not in sys.path:
    sys.path.append(BRIDGE_ROOT)


def _comfy_panel():
    import comfy_bridge.nuke_panel
    comfy_bridge.nuke_panel.show()


nuke.menu("Nuke").addMenu("ComfyUI").addCommand("Generate passes...", _comfy_panel, "ctrl+alt+g")
