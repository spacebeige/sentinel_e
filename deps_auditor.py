import os
import re

TARGET_DIR = 'frontend/src'
DIRS_TO_CHECK = ['stores', 'hooks', 'services', 'utils']

# We want to find imports like `import X from 'package-name'`
# or `import { Y } from 'package-name'`
import_regex = re.compile(r"import\s+(?:.*?\s+from\s+)?['\"]([^'\"]+)['\"]")

imported_packages = set()

for root, dirs, files in os.walk(TARGET_DIR):
    if any(d in root for d in DIRS_TO_CHECK):
        for file in files:
            if file.endswith('.js') or file.endswith('.ts') or file.endswith('.jsx') or file.endswith('.tsx'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for match in import_regex.finditer(content):
                    import_path = match.group(1)
                    # Ignore relative or alias imports (unless the alias points to an external package)
                    if not import_path.startswith('.') and not import_path.startswith('/') and not import_path.startswith('@/'):
                        # get the base package name
                        if import_path.startswith('@'):
                            parts = import_path.split('/')
                            if len(parts) >= 2:
                                pkg_name = f"{parts[0]}/{parts[1]}"
                            else:
                                pkg_name = parts[0]
                        else:
                            pkg_name = import_path.split('/')[0]
                            
                        imported_packages.add((pkg_name, filepath))

for pkg, filepath in sorted(list(imported_packages)):
    print(f"{pkg} imported in {filepath}")

