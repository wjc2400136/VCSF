from _common import run_protocol


if __name__ == "__main__":
    raise SystemExit(
        run_protocol(
            "method_ablation",
            "LGP loss, target, candidate-count, foreground-scale and budget studies.",
            fixed_method="lgp",
        )
    )
