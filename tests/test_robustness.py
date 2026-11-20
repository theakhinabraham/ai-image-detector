import pickle

from PIL import Image

from src.robustness import DEGRADATIONS


def test_degradations_keep_size_and_are_picklable():
    img = Image.new("RGB", (32, 32), (120, 30, 200))
    for name, fn in DEGRADATIONS.items():
        if fn is None:
            continue
        pickle.dumps(fn)  # DataLoader workers on macOS/Windows need this
        out = fn(img)
        assert out.size == (32, 32) and out.mode == "RGB", name
