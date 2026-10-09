from _common import run_protocol


if __name__ == "__main__":
    raise SystemExit(
        run_protocol(
            "main_transfer",
            "COCO-default six-source, sixteen-target main transfer experiment with TeX and figures.",
            default_datasets=["coco"],
            default_stop_on_error=True,
        )
    )
