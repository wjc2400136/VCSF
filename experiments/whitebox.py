from _common import run_protocol


if __name__ == "__main__":
    raise SystemExit(
        run_protocol(
            "whitebox",
            "COCO-default current-method source-matched white-box diagonal with twelve AP/AR metrics.",
            default_datasets=["coco"],
            default_stop_on_error=True,
        )
    )
