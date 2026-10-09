from _common import run_protocol


if __name__ == "__main__":
    raise SystemExit(
        run_protocol(
            "method_ablation",
            "NumbOD spatial/frequency component and budget studies.",
            fixed_method="numbod",
        )
    )
