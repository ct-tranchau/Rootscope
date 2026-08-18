"""Run RootScope's CPU classification stage in a throwaway subprocess.

Why this file exists
--------------------
Unpickling the XGBoost classifier builds a Booster, and building a Booster
probes for CUDA devices. On ZeroGPU that registers device state the
snapshot/restore cycle cannot reproduce, so *every* subsequent ``@spaces.GPU``
task in that process aborts, including a bare
``torch.zeros(2048, 2048, device="cuda")``. Confirmed on the live Space: the
diagnostics' trivial-allocation step returned ``GPU task aborted`` while the
environment itself was healthy (torch 2.11.0+cu130, ``cuda.is_available True``).

Merely having the model resident is enough to break the GPU stages, even though
classification happens long after them. So the main process must never load a
classifier at all. This worker loads all three from disk, does the whole
iterative classify + post-process + overlay pass, writes its results, and
exits, taking the poisoned CUDA state with it. The parent is left holding
torch state only, the same shape as the Cellpose-SAM Space, which snapshots
cleanly.

The cost is reloading ~460 MB of classifiers per run instead of once at
startup: about 2 s against a 2-4 minute pipeline.

Invoked by ``app.py`` as::

    python classify_worker.py <job.pkl> <result.pkl>

Both paths live in the run's own temp directory. The result pickle always gets
written, success or failure, so the parent can tell a crash apart from a
process that died without saying why.
"""

import pickle
import sys
import traceback


def run(job):
    """Load the classifiers and run the CPU stage. Returns the per-cell table."""
    # Imported here, not at module scope, so that even this process does no
    # model work until it has a job to do.
    from rootscope import predict as rs

    models, scalers, feature_cols, le = rs.load_models(job["model_dir"])
    if not models:
        raise RuntimeError(f"No usable models found in {job['model_dir']}")

    df = rs.stage_classify(
        job["df_base"], job["masks"], job["img_rgb"],
        job["layer_lookup"], job["adjacency"],
        models, scalers, feature_cols, le,
        out_dir=job["out_dir"],
        stem=job["stem"],
        source_name=job["source_name"],
        um_per_px=job["um_per_px"],
        max_rounds=job["max_rounds"],
        label_cells=job["label_cells"],
    )
    return {"ok": True, "df": df, "classes": [str(c) for c in le.classes_],
            "models": list(models.keys())}


def main():
    job_path, result_path = sys.argv[1], sys.argv[2]
    try:
        with open(job_path, "rb") as f:
            job = pickle.load(f)
        result = run(job)
    except Exception:
        # Never let the parent guess. A traceback here is far more use than
        # "the subprocess exited 1".
        result = {"ok": False, "traceback": traceback.format_exc()}
        print("[classify-worker] FAILED\n" + result["traceback"], flush=True)

    with open(result_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
