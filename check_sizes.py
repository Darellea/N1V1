import os

for f in os.listdir("docs"):
    if f.endswith(".md"):
        size = os.path.getsize(os.path.join("docs", f))
        print(f"{f}: {size} bytes")
