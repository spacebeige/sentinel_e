import re
import glob

files = glob.glob("frontend/src/**/*.js", recursive=True)

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split('\n')
    for i, line in enumerate(lines):
        # find patterns like `foo.bar.map(` or `foo?.bar.map(` or `baz.map(`
        # but skip the ones we already found, skip "O.keys().map", skip "Promise.all(..).map"
        if ".map(" in line and "Object." not in line and "Array." not in line:
            # Let's just print a sample
            pass
