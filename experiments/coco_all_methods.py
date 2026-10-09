"""Run the formal unified eleven-method COCO transfer benchmark.

Run this file from the repository root. Its registered protocol is
fail-closed: formal execution always covers all 5,000 COCO val2017 images,
all six sources and all sixteen targets. A generated source-method payload is
pruned to the pre-frozen 500-image evidence subset only after all sixteen
target evaluations pass their integrity gates.

The default remains a beginner-friendly serial run on ``cuda:0``. Use
``--devices cuda:0,cuda:1`` for deterministic throughput parallelism: each
worker owns a complete source-method group, while the coordinator alone writes
global records. This is not distributed-gradient or per-image sharding.
"""

from _common import run_protocol


if __name__ == "__main__":
    raise SystemExit(
        run_protocol(
            "coco_all_methods",
            "Formal COCO all-method generation, transfer evaluation and reproducible reporting.",
            default_datasets=["coco"],
            default_stop_on_error=True,
            default_payload_retention="fixed_count_after_group_validation",
            default_retained_image_count=500,
            default_prediction_archive="gzip",
        )
    )
