#!/usr/bin/env python3
"""Assign ownership of legacy (pre-scoping) documents.

Documents ingested before privacy scoping have no entry in data/ownership.json
and are hidden from everyone by default (unless VAAANI_LEGACY_DOCS_SHARED=1).
Run this on the server to hand them to the right accounts.

Usage:
    python scripts/assign_owner.py --list
    python scripts/assign_owner.py --match "morphology" --user-id 14
    python scripts/assign_owner.py --all --user-id 14 [--school-id 3]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import json
from config import METADATA_PATH
import scope


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="list files and their owners")
    ap.add_argument("--match", help="substring of the file name/key to assign")
    ap.add_argument("--all", action="store_true", help="assign every unowned file")
    ap.add_argument("--user-id", type=int, help="owner user id")
    ap.add_argument("--school-id", type=int, action="append", default=[],
                    help="also share with this school id (repeatable)")
    args = ap.parse_args()

    meta = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {"files": {}}
    ownership = scope._load()

    if args.list or not (args.match or args.all):
        for key, info in meta.get("files", {}).items():
            rec = ownership.get(key)
            owner = f"owners={rec['owners']} schools={rec['school_ids']}" if rec else "UNOWNED (hidden)"
            print(f"{info.get('name', key)}\n    {key}\n    {owner}")
        return

    if args.user_id is None:
        ap.error("--user-id is required with --match/--all")

    n = 0
    for key, info in meta.get("files", {}).items():
        if args.all and key in ownership:
            continue
        if args.match and args.match.lower() not in key.lower() \
                and args.match.lower() not in info.get("name", "").lower():
            continue
        scope.record_ownership(key, args.user_id, args.school_id)
        print(f"assigned: {info.get('name', key)} -> user {args.user_id}"
              + (f" schools {args.school_id}" if args.school_id else ""))
        n += 1
    print(f"{n} file(s) assigned.")


if __name__ == "__main__":
    main()
