import os
import logging


def count_lines(root=".", exclude_dirs=("venv",)):
    total = 0
    rows = []
    exclude_dirs = set(exclude_dirs)
    for dirpath, dirnames, filenames in os.walk(root):
        # skip excluded dirs at top-level parts
        parts = os.path.normpath(dirpath).split(os.sep)
        if parts and parts[0] in exclude_dirs:
            continue
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                path = os.path.join(dirpath, fn)
                # skip files inside excluded dirs anywhere in path
                if any(
                    part in exclude_dirs
                    for part in os.path.normpath(path).split(os.sep)
                ):
                    continue
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        n = sum(1 for _ in f)
                except Exception:
                    n = 0
                rows.append((n, path))
                total += n
    return rows, total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    rows, total = count_lines(".", exclude_dirs=("venv",))
    for n, path in rows:
        logger.info("%d\t%s", n, path)
    logger.info("Total:\t%d", total)
