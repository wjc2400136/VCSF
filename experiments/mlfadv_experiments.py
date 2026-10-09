from _common import run_protocol


if __name__ == "__main__":
    raise SystemExit(
        run_protocol(
            "method_ablation",
            "MLFAdv module, radius 50--500 and budget studies.",
            fixed_method="mlfadv",
        )
    )
