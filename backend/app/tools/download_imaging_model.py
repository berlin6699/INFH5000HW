"""Download the optional local imaging weights once, then exit."""

from app.services.imaging_model import ImagingModelError, ensure_model_ready, model_status


def main() -> None:
    status = model_status()
    if status["weights_downloaded"]:
        print(f"Local Chest X-ray weights ready ({status['weights_bytes']} bytes).")
        return
    print("Downloading local Chest X-ray weights...")
    try:
        ensure_model_ready()
    except ImagingModelError as exc:
        # Keep the Mock dashboard deployable when the first run has no network.
        print(f"Warning: {exc} The dashboard will fall back to Mock imaging.")
        return
    print(f"Local Chest X-ray weights ready ({model_status()['weights_bytes']} bytes).")


if __name__ == "__main__":
    main()
