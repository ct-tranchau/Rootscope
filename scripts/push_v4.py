#!/usr/bin/env python
"""Push RootScope v4 to Hugging Face.

  python scripts/push_v4.py --dry-run          # list what would be uploaded, touch nothing
  python scripts/push_v4.py --weights          # model repo: add v4/ folder (additive, safe)
  python scripts/push_v4.py --space            # Space: replace the live app with v4
  python scripts/push_v4.py --weights --space  # both

Needs a WRITE token: `hf auth login` (or HF_TOKEN in the environment).
Nothing here deletes anything on the Hub. The Space upload is a commit on the
Space's git history, so it can be reverted from the Hub UI.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "ct-tranchau/Rootscope"
SPACE_REPO = "ct-tranchau/Rootscope"
MODEL_PREFIX = "v4/"

WEIGHT_FILES = ["backbone.pt", "meta.json", "lgbm_s42.joblib", "lgbm_s1.joblib", "lgbm_s7.joblib",
                "lgbm_geom_s42.joblib", "lgbm_geom_s1.joblib", "lgbm_geom_s7.joblib",
                "label_changes.csv"]     # the 80 corrected labels the v4.1 bundles were trained on
SPACE_FILES = {            # local -> path in Space
    "webapp/app.py": "app.py",
    "webapp/classify_worker.py": "classify_worker.py",
    "webapp/requirements.txt": "requirements.txt",
    "webapp/README.md": "README.md",
}
SPACE_EXAMPLES = ["examples/Acorulea_RootTip_Maturation.tif",
                  "examples/Spennellii_RootTip_EarlyMaturation.tif"]
VENDOR_PKG = "rootscope"    # the Space vendors its own copy of the package
IGNORE = ["__pycache__/*", "*.pyc", "*.bak_*", "*.log", "*.egg-info/*"]


def _api():
    from huggingface_hub import HfApi
    api = HfApi()
    me = api.whoami()        # raises on an invalid token: fail before touching anything
    role = (me.get("auth") or {}).get("accessToken", {}).get("role")
    print(f"logged in as {me.get('name')}  (token role: {role})")
    if role not in (None, "write", "fineGrained"):
        sys.exit(f"token role '{role}' cannot write. Create a WRITE token at "
                 f"https://huggingface.co/settings/tokens and run `hf auth login`.")
    return api


def plan_weights():
    missing = [f for f in WEIGHT_FILES if not (ROOT / "models" / f).exists()]
    if missing:
        sys.exit(f"missing weights in models/: {missing}")
    out = [(ROOT / "models" / f, MODEL_PREFIX + f) for f in WEIGHT_FILES]
    out.append((ROOT / "MODEL_CARD.md", "README.md"))       # model-repo README = model card
    return out


def plan_space():
    out = [(ROOT / src, dst) for src, dst in SPACE_FILES.items()]
    out += [(ROOT / e, e) for e in SPACE_EXAMPLES if (ROOT / e).exists()]
    # species.table() opens this CSV when a species name is typed; without it the Space crashes.
    for p in sorted((ROOT / VENDOR_PKG / "data").glob("*.csv")):
        out.append((p, f"{VENDOR_PKG}/data/{p.name}"))
    # Inference modules only. The training files (finetune.py, train.py and the
    # research training module) stay out of the Space: nothing there imports
    # them, and they pull in libraries the Space's requirements no longer list.
    SKIP = {"finetune.py", "train.py"}
    for p in sorted((ROOT / VENDOR_PKG).glob("*.py")):
        if ".bak_" in p.name or p.name in SKIP or p.name.startswith("_train"):
            continue
        out.append((p, f"{VENDOR_PKG}/{p.name}"))
    return out


def show(plan, title):
    tot = sum(p.stat().st_size for p, _ in plan)
    print(f"\n{title}: {len(plan)} files, {tot/1e6:.0f} MB")
    for src, dst in plan:
        print(f"   {src.stat().st_size/1e6:7.1f} MB  {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--weights", action="store_true")
    ap.add_argument("--space", action="store_true")
    ap.add_argument("--space-repo", default=SPACE_REPO,
                    help="push the app somewhere else first, e.g. ct-tranchau/Rootscope-v4-test")
    a = ap.parse_args()
    if not (a.weights or a.space or a.dry_run):
        ap.error("pick --weights and/or --space (or --dry-run)")

    pw, ps = plan_weights(), plan_space()
    if a.dry_run or a.weights:
        show(pw, f"MODEL REPO {MODEL_REPO}  (new folder {MODEL_PREFIX}, additive)")
    if a.dry_run or a.space:
        show(ps, f"SPACE {a.space_repo}  (replaces the live app)")
    if a.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    api = _api()
    if a.weights:
        print(f"\n==> model repo {MODEL_REPO}")
        for src, dst in pw:
            print(f"   uploading {dst} ...", flush=True)
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                            repo_id=MODEL_REPO, repo_type="model",
                            commit_message=f"v4: add {dst}")
        print("   done")
    if a.space:
        print(f"\n==> space {a.space_repo}")
        api.create_repo(repo_id=a.space_repo, repo_type="space", space_sdk="gradio",
                        exist_ok=True)
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for src, dst in ps:
                (td / dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, td / dst)
            # delete_patterns: the live Space carries rootscope/__pycache__/*.pyc from an
            # earlier push. Stale .pyc next to new .py are ignored by Python (source
            # mtime/size mismatch) but they are clutter; drop them in the same commit.
            api.upload_folder(folder_path=str(td), repo_id=a.space_repo, repo_type="space",
                              ignore_patterns=IGNORE,
                              delete_patterns=["rootscope/__pycache__/*", "**/*.pyc"],
                              commit_message="RootScope v4: manuscript model (LightGBM per round, "
                                             "3 seeds, layer + soft-neighbor context, radial prior)")
        print(f"   done -> https://huggingface.co/spaces/{a.space_repo}")


if __name__ == "__main__":
    main()
