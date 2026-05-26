import re
import os

# 1. Patch ControlButtons.tsx
cb_path = "figma_ui/src/app/tetris/ui/ControlButtons.tsx"
with open(cb_path, 'r') as f:
    cb = f.read()

# Replace hardcoded sound import with safe fallback
sound_import_regex = r'import clickSound from "\.\./assets/sounds/button-click\.mp3"\s*const audio = new Audio\(clickSound\)'
safe_sound = '''let audioUrl: string | null = null;
try {
  // Safe dynamic asset resolution fallback
  // audioUrl = new URL("../assets/sounds/button_click.mp3", import.meta.url).href;
  console.warn("[ASSET SYSTEM] Audio file missing. Initializing safe fallback.");
} catch (e) {
  console.warn("[ASSET SYSTEM] Failed to load audio:", e);
}
const audio = audioUrl ? new Audio(audioUrl) : null;'''

cb = re.sub(sound_import_regex, safe_sound, cb, flags=re.MULTILINE)

# Safe audio play
cb = cb.replace('audio.currentTime = 0\n        audio.play()', 'if (audio) {\n          audio.currentTime = 0\n          audio.play().catch(e => console.warn(e))\n        }')

# Replace bright red neon with dark crimson glass
cb = re.sub(r'radial-gradient\(\s*circle at 35% 30%,\s*#ffd9d9 0%,\s*#ff6666 16%,\s*#8a0000 52%,\s*#180000 100%\s*\)', 
            'radial-gradient(circle at top, #3a0c0c, #110000)', cb)

# Replace red button shadows
cb = re.sub(r'inset 0 8px 14px rgba\(255,255,255,0\.35\),\s*inset 0 -16px 28px rgba\(0,0,0,0\.85\),\s*0 0 16px rgba\(255,0,0,0\.2\),\s*0 0 40px rgba\(0,0,0,0\.9\)',
            'inset 0 8px 14px rgba(255,255,255,0.05), inset 0 -16px 28px rgba(0,0,0,0.85), 0 0 16px rgba(0,0,0,0.9), 0 0 40px rgba(0,0,0,0.9)', cb)

# Replace font ShareTech with VT323/ShareTechMono dynamic fallback
cb = cb.replace('"ShareTech"', '"VT323", "ShareTechMono", monospace')

# Replace text color #9c8d76 to #3a4a3a
cb = cb.replace('#9c8d76', '#4a5c4a')

with open(cb_path, 'w') as f:
    f.write(cb)

# 2. Patch SystemPanel.tsx
sp_path = "figma_ui/src/app/tetris/ui/SystemPanel.tsx"
with open(sp_path, 'r') as f:
    sp = f.read()

# Replace crt-noise.png with the real texture
sp = sp.replace("url('/src/app/tetris/assets/textures/crt-noise.png')", "url('/src/app/tetris/assets/textures/SmudgesLarge001/SmudgesLarge001_OVERLAY_VAR1_4K.jpg')")

# Dim green text
sp = sp.replace('#00ff9c', '#00aa55')
sp = sp.replace('rgba(0,255,120,0.22)', 'rgba(0,180,80,0.1)')
sp = sp.replace('rgba(0,255,120,0.08)', 'rgba(0,180,80,0.04)')

with open(sp_path, 'w') as f:
    f.write(sp)

# 3. Patch TetrisScreen.tsx
ts_path = "figma_ui/src/app/tetris/ui/TetrisScreen.tsx"
with open(ts_path, 'r') as f:
    ts = f.read()

# Replace crt-noise.png
ts = ts.replace("url('/src/app/tetris/assets/textures/crt-noise.png')", "url('/src/app/tetris/assets/textures/SmudgesLarge001/SmudgesLarge001_OVERLAY_VAR1_4K.jpg')")

# Reduce green gradient
ts = ts.replace('#062412', '#031209')
ts = ts.replace('#021008', '#010804')

# Dim phosphor
ts = ts.replace('rgba(0,255,120,0.16)', 'rgba(0,180,80,0.06)')
ts = ts.replace('rgba(0,255,120,0.04)', 'rgba(0,180,80,0.02)')

with open(ts_path, 'w') as f:
    f.write(ts)

print("Patching complete!")
