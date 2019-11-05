import hashlib
from pathlib import Path
from zlib import adler32

def hash_dir(dirpath):
    hash_list = list()
    for file in sorted(Path(dirpath).iterdir(), key=lambda x: x.name):
        with open(file, "rb") as f:
            h = hashlib.sha256()
            while True:
                chunk = f.read(h.block_size)
                if not chunk:
                    break
                h.update(chunk)
            hash_list.append(h.hexdigest())
    sequence_hash = str(adler32("".join(hash_list).encode("utf-8")))
    return sequence_hash