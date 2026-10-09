from _common import run_protocol


if __name__ == "__main__":
    raise SystemExit(
        run_protocol(
            "method_ablation",
            "NAA keeping-rate, target-layer, epsilon and budget studies.",
            fixed_method="naa",
        )
    )
