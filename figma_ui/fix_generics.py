import re

with open('src/legacy/services/api.js', 'r') as f:
    content = f.read()

content = re.sub(r'api\.get<[^>]+>\(', 'api.get(', content)
content = re.sub(r'api\.post<[^>]+>\(', 'api.post(', content)

with open('src/legacy/services/api.js', 'w') as f:
    f.write(content)
