from typing import Dict, List, Set
from pathlib import Path
from PIL import Image

def compute_dhash(image_path: Path, hash_size: int = 8) -> int:
    """Computes difference hash (dHash) for fast near-duplicate image detection."""
    try:
        with Image.open(image_path) as img:
            resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            if hasattr(resized, "get_flattened_data"):
                pixels = list(resized.get_flattened_data())
            else:
                pixels = list(resized.getdata())

            diff = []
            for row in range(hash_size):
                for col in range(hash_size):
                    pixel_left = pixels[row * (hash_size + 1) + col]
                    pixel_right = pixels[row * (hash_size + 1) + col + 1]
                    diff.append(pixel_left > pixel_right)

            decimal_val = 0
            for index, value in enumerate(diff):
                if value:
                    decimal_val += 1 << index
            return decimal_val
    except Exception:
        return 0

def find_near_duplicates(
    image_paths: List[Path], max_hamming_distance: int = 3
) -> List[Set[Path]]:
    """Clusters near-duplicate images based on dHash Hamming distance."""
    hashes: Dict[Path, int] = {}
    for p in image_paths:
        hashes[p] = compute_dhash(p)

    clusters: List[Set[Path]] = []
    visited: Set[Path] = set()

    for p1 in image_paths:
        if p1 in visited:
            continue
        cluster = {p1}
        visited.add(p1)
        h1 = hashes[p1]

        for p2 in image_paths:
            if p2 in visited:
                continue
            h2 = hashes[p2]
            hamming = bin(h1 ^ h2).count("1")
            if hamming <= max_hamming_distance:
                cluster.add(p2)
                visited.add(p2)

        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters
