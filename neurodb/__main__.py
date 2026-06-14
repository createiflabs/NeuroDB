"""``python -m neurodb`` entrypoint.

Subcommands:
  (default) / serve   run the API with uvicorn
  backup <dest>       write a consistent snapshot to <dest> (file or directory)
  restore <src>       replace the live data file with <src> (validated first)
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import get_settings
from .observability import configure_logging


def _serve() -> None:
    import uvicorn

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    # The store keeps all state in this process — multiple workers would each
    # get their own copy and corrupt the shared data file. Enforce one worker.
    workers = os.environ.get("WEB_CONCURRENCY")
    if workers and workers != "1":
        raise SystemExit(
            "NeuroDB must run with a single worker (in-process store); "
            f"WEB_CONCURRENCY={workers} is not supported."
        )
    uvicorn.run(
        "neurodb.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        workers=1,
    )


def _data_file(args: argparse.Namespace) -> str:
    return args.data_file or get_settings().data_file


def _cmd_backup(args: argparse.Namespace) -> None:
    from .backup import backup_store
    from .store import NeuroStore

    store = NeuroStore(_data_file(args))
    target = backup_store(store, args.dest)
    print(f"backup written to {target}")


def _cmd_restore(args: argparse.Namespace) -> None:
    from .backup import restore_file

    preserved = restore_file(args.src, _data_file(args))
    print(f"restored {_data_file(args)} from {args.src}")
    if preserved:
        print(f"previous data file preserved at {preserved}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="neurodb", description="NeuroDB CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve", help="run the API server (default)")

    p_backup = sub.add_parser("backup", help="write a consistent snapshot")
    p_backup.add_argument("dest", help="destination file or directory")
    p_backup.add_argument("--data-file", help="override NEURODB_DATA_FILE")

    p_restore = sub.add_parser("restore", help="restore the live data file from a backup")
    p_restore.add_argument("src", help="backup file to restore from")
    p_restore.add_argument("--data-file", help="override NEURODB_DATA_FILE")

    args = parser.parse_args(argv)

    if args.command in (None, "serve"):
        _serve()
    elif args.command == "backup":
        _cmd_backup(args)
    elif args.command == "restore":
        _cmd_restore(args)
    else:  # pragma: no cover - argparse rejects unknown commands
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
