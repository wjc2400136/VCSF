from _common import run_protocol


if __name__ == "__main__":
    raise SystemExit(
        run_protocol(
            "method_ablation",
            "SFIM-B SIM/FIM/PSIM/PFIM, masking, epsilon and budget studies.",
            fixed_method="sfim_b",
        )
    )
