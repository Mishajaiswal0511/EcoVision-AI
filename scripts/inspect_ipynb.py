import json
import sys

nb_path = sys.argv[1]
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    lines = src.strip().split("\n")
    first_line = lines[0] if lines else ""
    print(f"Cell {i:2d} [{cell['cell_type']:8s}]: {first_line[:80]}")

