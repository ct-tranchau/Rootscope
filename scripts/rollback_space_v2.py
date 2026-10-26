#!/usr/bin/env python
"""Put the v2 app back on the Space, from the local Rootscope_V2 checkout.

  python scripts/rollback_space_v2.py            # do it
  python scripts/rollback_space_v2.py --dry-run

Uploads V2's app.py (weights.py patched to read v2/ from the model repo) / classify_worker.py / requirements.txt / README.md, its rootscope/*.py and
the two example TIFFs, and DELETES the files that exist only in v4 (classify_v4.py, context.py,
radial_order.py, _eval_honest.py) so a v2 app never imports v4 code by accident.
"""
import argparse, shutil, sys, tempfile
from pathlib import Path
from huggingface_hub import HfApi
V2 = Path(__file__).resolve().parents[2] / "Rootscope_V2"
SPACE = "ct-tranchau/Rootscope"
V4_ONLY = ["rootscope/classify_v4.py", "rootscope/context.py", "rootscope/radial_order.py", "rootscope/_eval_honest.py"]
FILES = {"webapp/app.py": "app.py", "webapp/classify_worker.py": "classify_worker.py",
         "webapp/requirements.txt": "requirements.txt", "webapp/README.md": "README.md"}
def main():
    a = argparse.ArgumentParser(); a.add_argument("--dry-run", action="store_true"); a = a.parse_args()
    plan = [(V2 / s, d) for s, d in FILES.items()]
    plan += [(p, f"rootscope/{p.name}") for p in sorted((V2 / "rootscope").glob("*.py")) if p.name not in ("finetune.py", "train.py")]
    plan += [(p, f"examples/{p.name}") for p in sorted((V2 / "examples").glob("*.tif"))]
    missing = [str(s) for s, _ in plan if not s.exists()]
    if missing: sys.exit(f"missing locally: {missing}")
    print(f"ROLLBACK -> {SPACE}: {len(plan)} v2 files, delete {V4_ONLY}")
    for s, d in plan: print(f"   {d}")
    if a.dry_run: print("--dry-run: nothing changed"); return
    api = HfApi(); api.whoami()
    with tempfile.TemporaryDirectory() as td:
        for s, d in plan:
            (Path(td) / d).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(s, Path(td) / d)
        # Since the 2026-09-21 root cleanup the v2 weights live under v2/ in the model repo,
        # not at the root, so the restored v2 loader must ask for the v2/ prefix.
        wp = Path(td) / "rootscope/weights.py"; w = wp.read_text()
        old_pref = 'return "" if model_version() == "v2" else f"{model_version()}/"'
        if old_pref not in w: sys.exit("weights.py: _hf_prefix() line not found; refusing to roll back blind")
        wp.write_text(w.replace(old_pref, 'return f"{model_version()}/"'))
        api.upload_folder(folder_path=td, repo_id=SPACE, repo_type="space",
                          ignore_patterns=["__pycache__/*", "*.pyc", "*.bak_*"],
                          delete_patterns=V4_ONLY + ["rootscope/__pycache__/*"],
                          commit_message="rollback: RootScope v2 app restored")
    print(f"done -> https://huggingface.co/spaces/{SPACE}")
if __name__ == "__main__": main()
