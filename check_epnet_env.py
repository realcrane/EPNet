from __future__ import annotations

import importlib
import sys


GROUPS = {
    "elastic": ["tensorflow", "numpy", "scipy", "h5py", "matplotlib", "tqdm"],
    "pnet": ["tensorflow", "numpy"],
}


def check_module(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        return True, str(getattr(module, "__version__", "ok"))
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"


def main() -> int:
    print("python:", sys.executable)
    print("version:", sys.version.replace("\n", " "))
    failed = False
    for group, modules in GROUPS.items():
        print(f"\n[{group}]")
        for module in modules:
            ok, detail = check_module(module)
            print(f"{module}: {'OK' if ok else 'MISSING'} ({detail})")
            if group in {"elastic", "pnet"}:
                failed = failed or not ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
