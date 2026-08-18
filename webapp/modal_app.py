"""
Deploy the RootScope web app on Modal — a permanent public URL, free tier.

This serves the very same Gradio interface as webapp/app.py (the one written
for Hugging Face Spaces). Users visit a URL, upload a TIFF, and download the
overlay and per-cell CSV. Nothing about the UI changes; only where it runs.

Why this exists: hosting a ZeroGPU Space needs an account at least 30 days old,
and Modal has no such gate. Modal's free Starter plan includes $30/month of
compute credits, bills per second, and scales to zero when idle — at roughly a
minute of GPU-container time per image that is on the order of a thousand
images a month at no cost.

    pip install modal
    modal setup                                   # one-time browser login
    modal serve webapp/modal_app.py               # temporary URL, live reload
    modal deploy webapp/modal_app.py              # permanent URL

`modal deploy` prints the public URL. It stays up with no session to keep
alive, and costs nothing while nobody is using it.
"""

import modal

APP_NAME = "rootscope"

# Pin RootScope to a commit or tag before sharing the URL, so a later push to
# main cannot silently change what users get.
ROOTSCOPE_SPEC = "git+https://github.com/ct-tranchau/Rootscope.git@main"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        # OpenCV and scikit-image pull these in; debian_slim ships without them
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        "gradio>=5.0",
        "fastapi[standard]",
        ROOTSCOPE_SPEC,
        # Same hard pin as the package: the released .joblib models were
        # pickled with 1.7.2 and other versions can fail to unpickle.
        "scikit-learn==1.7.2",
    )
    .env({
        # Keep every downloaded weight on the Volume below, so a cold start
        # does not re-fetch ~460 MB of classifiers + the DINOv2 backbone +
        # the Cellpose-SAM checkpoint.
        "HF_HOME": "/cache/huggingface",
        "TORCH_HOME": "/cache/torch",
        "CELLPOSE_LOCAL_MODELS_PATH": "/cache/cellpose",
        "GRADIO_ANALYTICS_ENABLED": "0",
    })
    # app.py is the Space's file, reused verbatim. Baked in at build time so
    # the container does not depend on the local filesystem at run time.
    # classify_worker.py has to come along: app.py shells out to it for the
    # classification stage, and without it every run dies at that step.
    .add_local_file("app.py", "/root/app.py", copy=True)
    .add_local_file("classify_worker.py", "/root/classify_worker.py", copy=True)
)

# Weights survive between containers and deploys.
cache = modal.Volume.from_name("rootscope-cache", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu="T4",
    volumes={"/cache": cache},
    # Model load is ~12 s and a large image can take a few minutes.
    timeout=1800,
    # Keep a warm container for 5 min after the last request, so a second
    # upload does not pay the model-loading cost again.
    scaledown_window=300,
    max_containers=2,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def ui():
    """Serve webapp/app.py's Gradio Blocks over ASGI."""
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    # Importing app.py loads the classifiers, Cellpose-SAM and DINOv2 onto the
    # GPU. That happens once per container, not once per request.
    import sys

    sys.path.insert(0, "/root")
    import app as rootscope_app  # noqa: E402

    rootscope_app.demo.queue(max_size=20)
    # On Gradio 6 the theme is applied at serve time rather than on Blocks(),
    # so hand it to mount_gradio_app to match how the Space looks.
    kwargs = {}
    if getattr(rootscope_app, "_THEME_ON_LAUNCH", False):
        kwargs["theme"] = rootscope_app._THEME
    return mount_gradio_app(
        app=FastAPI(),
        blocks=rootscope_app.demo,
        path="/",
        allowed_paths=[str(rootscope_app.TMP_ROOT)],
        **kwargs,
    )


@app.function(volumes={"/cache": cache}, timeout=1800)
def warm_cache():
    """Pre-download every weight into the Volume.

    Run once after deploying so the first real visitor does not wait for
    ~550 MB of downloads:

        modal run webapp/modal_app.py::warm_cache
    """
    from rootscope.cnn_embeddings import load_dinov2
    from rootscope.extract_features import load_cellpose_model
    from rootscope.weights import resolve_cnn_weights, resolve_model_dir

    print("classifiers ->", resolve_model_dir())
    w = resolve_cnn_weights()
    print("backbone    ->", w)
    load_dinov2(weights_path=str(w) if w else None, use_gpu=False)
    load_cellpose_model(use_gpu=False)
    cache.commit()
    print("cache warmed")
