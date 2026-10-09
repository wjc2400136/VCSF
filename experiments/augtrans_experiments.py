from _common import run_protocol


if __name__ == "__main__":
    raise SystemExit(
        run_protocol(
            "method_ablation",
            "AugTrans cumulative/individual augmentation, loss and sensitivity studies.",
            fixed_method="augtrans",
        )
    )
