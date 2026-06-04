import re

with open('src/legacy/services/api.js', 'r') as f:
    content = f.read()

content = re.sub(r'return adaptKernel\(raw\);', 'return raw;', content)
content = re.sub(r'return adaptSessionStats\(raw\);', 'return raw;', content)

with open('src/legacy/services/api.js', 'w') as f:
    f.write(content)
