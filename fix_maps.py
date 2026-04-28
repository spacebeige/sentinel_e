import re
import glob

files = glob.glob("frontend/src/**/*.js", recursive=True)

# We want to find patterns like `something?.map(` or `something?.something?.map(`
# and wrap them in Array.isArray(...) ? ... : []
# But replacing safely via regex without an AST can be tricky. Let's just see how many there are.

count = 0

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split('\n')
    for i, line in enumerate(lines):
        if re.search(r'\?\.\w*\.?map\(', line):
            print(f"{fpath}:{i+1}:{line.strip()}")
            count += 1
            
print(f"Total potential crashes: {count}")
