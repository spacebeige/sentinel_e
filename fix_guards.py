import os
import glob
import re

components_dir = "frontend/src"
files = glob.glob(components_dir + "/**/*.js", recursive=True)

fallback_regex = re.compile(r"if\s*\(!data\)\s*return\s+(?:null|);")

def replace_fallback(match):
    return "if (!data) { console.warn('Missing data in component'); return <div className='p-4 text-[#aeaeb2] text-sm'>Waiting for results...</div>; }"

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    new_content = fallback_regex.sub(replace_fallback, content)
    
    if new_content != content:
        print(f"Patched fallbacks in {fpath}")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(new_content)
