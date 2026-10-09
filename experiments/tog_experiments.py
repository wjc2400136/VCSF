from _common import run_protocol


if __name__ == "__main__":
    raise SystemExit(
        run_protocol(
            "method_ablation",
            "TOG objective and budget studies.",
            fixed_method="tog",
            default_sources=["yolov3_d53"],
        )
    )
