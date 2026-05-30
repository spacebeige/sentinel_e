import re

with open("src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

# Replace the specific definition lines 109-114
# Since we know the exact lines, we can just filter them out.
lines = content.split('\n')
new_lines = []
skip = False
for line in lines:
    if line.startswith('// ── Pro orchestration sub-modes'):
        skip = True
        continue
    if skip and line.startswith('];'):
        skip = False
        continue
    if not skip:
        new_lines.append(line)

with open("src/app/components/ChatPage.tsx", "w") as f:
    f.write('\n'.join(new_lines))
