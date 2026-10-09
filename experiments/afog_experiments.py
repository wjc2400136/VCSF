from _common import run_protocol


if __name__ == "__main__":
    raise SystemExit(
        run_protocol(
            "method_ablation",
            "AFOG attention, objective, budget, quality and efficiency studies.",
            fixed_method="afog",
            default_sources=["deformable_detr_r50"],
        )
    )
