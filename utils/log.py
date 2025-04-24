from __future__ import annotations
import os
import sys
import builtins
from pathlib import Path
from typing import Any

LOG_LOCATION = "./app.log"

def change_log_location(log_location: str):
    global LOG_LOCATION
    LOG_LOCATION = log_location + "/print.log"

def get_log_location() -> str | os.PathLike[str]:
    return LOG_LOCATION

def log_print(
        *objects: Any,
        sep: str = " ",
        end: str = "\n",
        file = sys.stdout,
        flush: bool = False,
        **kwargs,
) -> None:
    builtins.print(*objects, sep=sep, end=end, file=file, flush=flush, **kwargs)
    msg: str = sep.join(map(str, objects)) + end
    log_path = Path(get_log_location()).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(msg)