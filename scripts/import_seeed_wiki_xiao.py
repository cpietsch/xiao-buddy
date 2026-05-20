from __future__ import annotations

import sys

from import_seeed_wiki import main


if __name__ == "__main__":
    if not any(arg == "--scope" or arg.startswith("--scope=") for arg in sys.argv[1:]):
        sys.argv.extend(["--scope", "xiao"])
    main()
