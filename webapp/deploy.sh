#!/usr/bin/env bash
#
# Deploy the RootScope Gradio app to a Hugging Face Space.
#
#   bash webapp/deploy.sh                      # private Space (default)
#   bash webapp/deploy.sh --public             # public Space
#   bash webapp/deploy.sh --space me/OtherName # different owner/name
#
# Prerequisites, all checked below:
#   - logged in:  huggingface-cli login
#   - ZeroGPU eligibility: account >= 30 days old with a verified email, or
#     PRO, or an approved community grant. Without it the create call returns
#     HTTP 402 and nothing is created.
#
set -euo pipefail

SPACE="ct-tranchau/Rootscope"
PRIVATE="true"
HARDWARE="zero-a10g"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --public)   PRIVATE="false"; shift ;;
    --space)    SPACE="$2"; shift 2 ;;
    --hardware) HARDWARE="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEBAPP="$REPO_ROOT/webapp"

echo "==> Space:     https://huggingface.co/spaces/$SPACE"
echo "==> Private:   $PRIVATE"
echo "==> Hardware:  $HARDWARE"
echo

echo "==> Checking authentication and eligibility"
python - "$SPACE" <<'PY'
import sys, datetime, requests, huggingface_hub
space = sys.argv[1]
tok = huggingface_hub.get_token()
if not tok:
    sys.exit("Not logged in. Run: huggingface-cli login")
who = requests.get("https://huggingface.co/api/whoami-v2",
                   headers={"Authorization": f"Bearer {tok}"}).json()
user = who.get("name")
print(f"    logged in as : {user}")
if not user:
    sys.exit("Token rejected. Run: huggingface-cli login")

owner = space.split("/")[0]
if owner != user and owner not in [o.get("name") for o in who.get("orgs", [])]:
    sys.exit(f"You cannot push to '{owner}' as '{user}'.")

ov = requests.get(f"https://huggingface.co/api/users/{user}/overview").json()
created = ov.get("createdAt")
if created:
    age = (datetime.datetime.now(datetime.timezone.utc)
           - datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))).days
    print(f"    account age  : {age} days")
    if age < 30 and not who.get("isPro"):
        print(f"    WARNING: ZeroGPU hosting needs a 30-day-old account or PRO.")
        print(f"             Eligible in {30 - age} more day(s), or subscribe,")
        print(f"             or request a community grant.")
print(f"    PRO          : {who.get('isPro')}")
PY

echo
echo "==> Creating the Space (idempotent)"
python - "$SPACE" "$PRIVATE" "$HARDWARE" <<'PY'
import sys
from huggingface_hub import HfApi
space, private, hardware = sys.argv[1], sys.argv[2] == "true", sys.argv[3]
try:
    url = HfApi().create_repo(repo_id=space, repo_type="space",
                              space_sdk="gradio", space_hardware=hardware,
                              private=private, exist_ok=True)
    print(f"    ok: {url}")
except Exception as e:
    msg = str(e)
    if "402" in msg:
        sys.exit("    HTTP 402 - not eligible yet. Nothing was created.\n"
                 "    Wait for the 30-day mark, subscribe to PRO, or request a\n"
                 "    community grant, then re-run this script.")
    raise
PY

echo
echo "==> Uploading app files"
python - "$SPACE" "$WEBAPP" "$REPO_ROOT" <<'PY'
import sys, pathlib
from huggingface_hub import HfApi
space, webapp, root = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
api = HfApi()
for src, dest in [(webapp / "app.py", "app.py"),
                  (webapp / "requirements.txt", "requirements.txt"),
                  (webapp / "README.md", "README.md"),
                  (root / "examples" / "Acorulea_RootTip_Maturation.tif",
                   "examples/Acorulea_RootTip_Maturation.tif"),
                  (root / "examples" / "Spennellii_RootTip_EarlyMaturation.tif",
                   "examples/Spennellii_RootTip_EarlyMaturation.tif")]:
    api.upload_file(path_or_fileobj=str(src), path_in_repo=dest,
                    repo_id=space, repo_type="space")
    print(f"    uploaded {dest}")
PY

echo
echo "==> Done. Watch the build at:"
echo "    https://huggingface.co/spaces/$SPACE"
echo
echo "First boot downloads ~460 MB of classifiers, the DINOv2 backbone, and the"
echo "Cellpose-SAM checkpoint, so expect a slow first start."
echo
echo "BEFORE MAKING IT PUBLIC: run both example images on the Space and diff the"
echo "CSVs against a local run. The Space resolves torch>=2.8 (ZeroGPU's floor)"
echo "while RootScope pins torch==2.4.0 locally, so that stack is unvalidated."
