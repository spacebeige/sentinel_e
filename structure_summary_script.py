import os
import json

def get_structure(path, max_depth=2, current_depth=0):
    structure = {"type": "directory", "name": os.path.basename(path), "children": [], "file_count": 0, "dir_count": 0}

    if current_depth >= max_depth:
        return structure

    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    child = get_structure(entry.path, max_depth, current_depth + 1)
                    structure["children"].append(child)
                    structure["dir_count"] += 1 + child["dir_count"]
                    structure["file_count"] += child["file_count"]
                elif entry.is_file(follow_symlinks=False):
                    structure["children"].append({"type": "file", "name": entry.name})
                    structure["file_count"] += 1
    except PermissionError:
        structure["children"].append({"type": "error", "name": "PermissionError"})
    return structure

if __name__ == "__main__":
    root_path = os.path.dirname(os.path.abspath(__file__))
    structure = get_structure(root_path, max_depth=2)
    with open("structure_summary.json", "w") as f:
        json.dump(structure, f, indent=2)
    print("Directory structure summary written to structure_summary.json (tree -L 2)")
