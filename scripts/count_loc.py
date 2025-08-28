import os


def count_lines(root="."):
    total = 0
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        n = sum(1 for _ in f)
                except Exception:
                    n = 0
                rows.append((n, path))
                total += n
    return rows, total


if __name__ == "__main__":
    rows, total = count_lines(".")
    for n, path in rows:
        print(f"{n}\t{path}")
    print(f"Total:\t{total}")
