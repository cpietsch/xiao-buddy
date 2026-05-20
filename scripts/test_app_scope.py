from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import EXAMPLES, INITIAL_ANSWER


def main() -> None:
    examples = "\n".join(str(example[1]) for example in EXAMPLES)
    _assert("Seeed sensor" in INITIAL_ANSWER, "initial answer should advertise broader Seeed sensor scope")
    _assert("Wio-SX1262" in examples, "examples should include LoRa module coverage")
    _assert("Grove Vision AI V2" in examples, "examples should include vision sensor coverage")
    _assert("Wio Terminal" in examples, "examples should include non-XIAO sensor workflows")
    _assert("RS485 Expansion Board" in examples, "examples should include expansion/robotics-adjacent wiring")
    print("PASS app scope copy regression")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
