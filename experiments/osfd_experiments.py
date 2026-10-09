from _common import run_protocol


if __name__ == "__main__":
    raise SystemExit(
        run_protocol(
            "method_ablation",
            "OSFD independent RRB component, amplification and budget studies.",
            fixed_method="osfd",
        )
    )
