#!/usr/bin/env python
"""Finish "v4 replaces v2" on the model repo: remove the v2 weights from the repo ROOT.

Layout after this: root = README.md only; v1/, v2/, v4/ hold the versions.
(v4 stays in v4/ -- weights.py already maps it there, so nothing is re-uploaded.)

  python scripts/cleanup_root_v2.py --dry-run
  python scripts/cleanup_root_v2.py --delete

Refuses to delete unless BOTH hold, so it cannot break a running consumer:
  1. every root v2 file already exists under v2/ with the same LFS sha256 (or size);
  2. the Space's live app.py mentions classify_v4, i.e. the Space no longer reads v2
     from the root.
"""
import argparse, sys
from huggingface_hub import HfApi, hf_hub_download

REPO, SPACE = "ct-tranchau/Rootscope", "ct-tranchau/Rootscope"
KEEP = {".gitattributes", "README.md"}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true"); g.add_argument("--delete", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    info = api.model_info(REPO, files_metadata=True)
    sib = {s.rfilename: s for s in info.siblings}
    root = sorted(n for n in sib if "/" not in n and n not in KEEP)
    print(f"root v2 files ({len(root)}):", root)

    # guard 1: v2/ holds identical copies
    problems = []
    for n in root:
        m = sib.get(f"v2/{n}")
        if m is None: problems.append(f"missing v2/{n}"); continue
        a_, b_ = sib[n], m
        same = (a_.lfs.sha256 == b_.lfs.sha256) if (a_.lfs and b_.lfs) else (a_.size == b_.size)
        if not same: problems.append(f"v2/{n} differs from root")
    print("guard 1 (v2/ identical):", "OK" if not problems else problems)

    # guard 2: the live Space is already on v4
    try:
        p = hf_hub_download(SPACE, "app.py", repo_type="space")
        on_v4 = "classify_v4" in open(p).read() or "is_v4" in open(p).read()
    except Exception as e:
        on_v4 = False; print("   (could not read Space app.py:", type(e).__name__, ")")
    print("guard 2 (Space on v4):", "OK" if on_v4 else "NOT YET -- the live Space still reads root v2")

    if a.dry_run:
        print("\n--dry-run: would delete", root if (not problems and on_v4) else "NOTHING (guards not satisfied)")
        return
    if problems or not on_v4:
        sys.exit("refusing to delete: guards not satisfied")
    api.delete_files(repo_id=REPO, repo_type="model", delete_patterns=root,
                     commit_message="v4 replaces v2: root weights removed (v2 kept under v2/)")
    print("deleted", len(root), "root files; versions now live in v1/ v2/ v4/")


if __name__ == "__main__":
    main()
