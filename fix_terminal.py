import re

with open('figma_ui/src/app/tetris/ui/TerminalShell.tsx', 'r') as f:
    body = f.read()

# Replace grays with greens
# dark grays / blacks -> #071207
body = re.sub(r'#(050505|080808|0a0a0a|151515|1a1b18|121310|030603|020802|030b03)', '071207', body)
# mid grays -> #0b1f0b
body = re.sub(r'#(222|222222|2a2b28|2b2b2b|333|333333|1f2b1f)', '0b1f0b', body)
# light grays / borders -> #103010
body = re.sub(r'#(444|444444|555|555555|777|777777|4b4f47)', '103010', body)

# Text colors
body = re.sub(r'text-\[#(7a7a55|7d7355|8b7e5d|73684d)\]', 'text-green-400', body)
body = re.sub(r'text-\[#333\]', 'text-green-700', body)

# Buttons
body = body.replace('bg-[radial-gradient(circle_at_top,#ff4b4b,#5e0000)]', 'bg-[radial-gradient(circle_at_top,#103010,#071207)]')
body = body.replace('border-[#330000]', 'border-green-900')
body = body.replace('shadow-[0_0_50px_rgba(255,0,0,0.35)]', 'shadow-[0_0_50px_rgba(0,255,120,0.15)]')

# Purple blocks
body = body.replace('border-purple-300', 'border-green-400')
body = body.replace('bg-purple-500/70', 'bg-green-500/70')
body = body.replace('shadow-[0_0_24px_rgba(180,100,255,0.9)]', 'shadow-[0_0_24px_rgba(0,255,120,0.9)]')

with open('figma_ui/src/app/tetris/ui/TerminalShell.tsx', 'w') as f:
    f.write(body)
print("Modifications applied.")
