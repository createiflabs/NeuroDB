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


def _cmd_migrate(args: argparse.Namespace) -> None:
    from .store import _MANIFEST_VERSION, NeuroStore, peek_manifest_version

    data_file = _data_file(args)
    before = peek_manifest_version(data_file)
    if before is None:
        print(f"no data file (or no manifest) at {data_file}; nothing to migrate")
        return
    if before == _MANIFEST_VERSION:
        print(f"data file already at manifest v{before}; no migration needed")
        return
    # Loading migrates in memory; saving stamps the current version on disk.
    store = NeuroStore(data_file, fail_on_corrupt_load=True)
    store.save_all()
    after = peek_manifest_version(data_file)
    print(f"migrated {data_file}: manifest v{before} -> v{after}")


def _cmd_collection(args: argparse.Namespace) -> None:
    import json

    from .collections import bundle as bundle_mod

    if args.collection_command == "info":
        print(json.dumps(bundle_mod.info(args.bundle), indent=2, default=str))
    elif args.collection_command == "verify":
        result = bundle_mod.verify_bundle(args.bundle)
        print(json.dumps(result.to_dict(), indent=2))
    elif args.collection_command == "load":
        from .store import NeuroStore

        store = NeuroStore(_data_file(args))
        mem = store.load_collection(args.bundle, args.name)
        print(f"loaded collection {mem.name!r} ({mem.count} patterns) into {_data_file(args)}")


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

    p_migrate = sub.add_parser("migrate", help="upgrade the data file to the current format")
    p_migrate.add_argument("--data-file", help="override NEURODB_DATA_FILE")

    p_coll = sub.add_parser("collection", help="inspect/verify/load collection bundles")
    coll_sub = p_coll.add_subparsers(dest="collection_command", required=True)
    p_ci = coll_sub.add_parser("info", help="print a bundle's metadata (no patterns)")
    p_ci.add_argument("bundle")
    p_cv = coll_sub.add_parser("verify", help="verify a bundle's signature")
    p_cv.add_argument("bundle")
    p_cl = coll_sub.add_parser("load", help="load a bundle into the data file")
    p_cl.add_argument("bundle")
    p_cl.add_argument("--name", help="override the collection name")
    p_cl.add_argument("--data-file", help="override NEURODB_DATA_FILE")

    args = parser.parse_args(argv)

    if args.command in (None, "serve"):
        _serve()
    elif args.command == "backup":
        _cmd_backup(args)
    elif args.command == "restore":
        _cmd_restore(args)
    elif args.command == "migrate":
        _cmd_migrate(args)
    elif args.command == "collection":
        _cmd_collection(args)
    else:  # pragma: no cover - argparse rejects unknown commands
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
