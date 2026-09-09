"""离线评测门禁:提案应用后跑全量回归,指标不退化才放行。

门禁口径:eval/run_eval.py --gate 退出码 0(全场景通过)。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check(python: str | None = None) -> tuple[bool, str]:
    """Run evaluation with optional python executable; default to this interpreter."""
    # 沿用当前环境 / Preserve the active Conda or virtualenv interpreter.
    py = python or sys.executable
    proc = subprocess.run(
        [py, "eval/run_eval.py", "--gate"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-5:]
    return proc.returncode == 0, "\n".join(tail)


def main() -> None:
    ok, output = check()
    print(output)
    print("门禁结果:" + ("✅ 放行" if ok else "❌ 拦截(指标退化)"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
