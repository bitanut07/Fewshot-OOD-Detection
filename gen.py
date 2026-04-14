# generator
import os
R = "/Users/lap14568/Fewshot-OOD-Detection"

def mk(name, content):
    path = os.path.join(R, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {name}")