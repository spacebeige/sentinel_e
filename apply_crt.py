import re

filepath = 'figma_ui/src/app/tetris/ui/TerminalShell.tsx'
with open(filepath, 'r') as f:
    content = f.read()

# Split into active code to avoid messing with the commented out section
# We know the active code is at the bottom, so we'll split by the LAST occurrence
parts = content.rsplit('export default function TerminalShell', 1)

if len(parts) == 2:
    header = parts[0]
    body = 'export default function TerminalShell' + parts[1]

    # Add import to header if not present
    if "import './styles/crt.css'" not in header:
        header = "import './styles/crt.css';\n" + header

    # Global Color Replacements (Darkening)
    # Dark greens -> Charcoal / Dirty steel
    body = body.replace('bg-[#071207]', 'console-body')
    body = body.replace('bg-[#0b1f0b]', 'console-bezel')
    body = body.replace('border-[#103010]', 'border-[#2a2a2c]')
    
    # Dimming text
    body = body.replace('text-green-400', 'text-green-600')
    body = body.replace('text-green-700', 'text-[#4a5c4a]')
    body = body.replace('text-[#22ff88]', 'text-green-600')
    body = body.replace('text-[#18cc66]', 'text-green-600')

    # Dimming specific UI blocks
    body = body.replace('bg-green-500/70', 'bg-green-700/40')
    body = body.replace('border-green-400', 'border-green-700/50')
    body = body.replace('shadow-[0_0_24px_rgba(0,255,120,0.9)]', 'shadow-[0_0_24px_rgba(0,255,120,0.3)]')

    # Fix D-Pad
    # Let's use regex for D-Pad to make sure it hits
    body = re.sub(r'bg-\[\#071207\]\s+border\s+border-\[\#103010\]', 'console-dpad border-none', body)

    # Fix Action Buttons
    body = body.replace('bg-[radial-gradient(circle_at_top,#103010,#071207)]', 'console-action-btn')
    body = body.replace('border-green-900', 'border-none')
    body = body.replace('shadow-[0_0_50px_rgba(0,255,120,0.15)]', 'shadow-none')

    # Inner CRT Background
    # Actually let's just do a blanket replace if bg-black is inside Inner CRT.
    body = body.replace('rounded-[30px]\n                overflow-hidden\n              "', 
                        'rounded-[30px]\n                overflow-hidden\n                console-crt-bg\n              "')
    
    # CRT GLOW -> phosphor bloom
    body = body.replace('className="absolute inset-0 pointer-events-none"\n                style={{\n                  background: `\n                    radial-gradient(\n                      circle at center,\n                      rgba(0,255,120,0.08),\n                      transparent 60%\n                    )\n                  `\n                }}',
                        'className="console-phosphor-bloom"')

    # VIGNETTE -> console-vignette
    body = body.replace('className="absolute inset-0 pointer-events-none"\n                style={{\n                  boxShadow: `\n                    inset 0 0 120px rgba(0,0,0,1)\n                  `\n                }}',
                        'className="console-vignette"')

    # Add glass smudges right before VIGNETTE
    glass_div = '              {/* GLASS SMUDGES */}\n              <div className="console-screen-glass" />\n'
    body = body.replace('{/* VIGNETTE */}', glass_div + '\n              {/* VIGNETTE */}')

    with open(filepath, 'w') as f:
        f.write(header + body)
    print("Modifications applied.")
else:
    print("Failed to split file")
