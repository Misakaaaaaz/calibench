import traceback
from imagenet_sketch_beit_base import *
from imagenet_sketch_beit_large import *
from imagenet_sketch_beitv2_base import *
from imagenet_sketch_eva02_small import *
from imagenet_sketch_eva02_base import *
from imagenet_sketch_eva02_large import *

def _safe_run(fn):
    try:
        fn()
    except Exception as e:
        print(f"[ERROR] {fn.__name__} failed: {e}")
        traceback.print_exc()
        print("-" * 80)

if __name__ == '__main__':
    for fn in [
        beit_base,
        beit_large,
        beitv2_base,
        eva02_small,
        eva02_base,
        eva02_large
    ]:
        _safe_run(fn)