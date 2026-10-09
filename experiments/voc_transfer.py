"""Run the formal unified eleven-method Pascal VOC transfer benchmark.

Run this file from the repository root. Formal execution covers all 4,952
VOC2007 test images registered as ``val``, all six canonical sources and all
sixteen targets. ``--devices cuda:0,cuda:1`` assigns complete source-method
groups to two independent workers; it never shards an attack or its images.
The frozen matrix contains 1,056 logical cells: 848 real evaluations from 53
payloads and 208 audited structural skips. Each payload retains 500 evidence
PNGs only after its complete sixteen-target evaluation passes.
"""

from _common import run_protocol


if __name__ == "__main__":
    raise SystemExit(
        run_protocol(
            "voc_all_methods",
            "Formal VOC all-method generation, transfer evaluation and reproducible reporting.",
            default_datasets=["voc"],
            default_stop_on_error=True,
            default_payload_retention="fixed_count_after_group_validation",
            default_retained_image_count=500,
            default_prediction_archive="gzip",
        )
    )
