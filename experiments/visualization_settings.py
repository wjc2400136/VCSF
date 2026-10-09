"""Shared, edit-in-place settings for prediction visualization.

Set ``SAVE_PREDICTION_VISUALIZATIONS`` to ``True`` to make clean, transfer,
ablation and input-robustness evaluations save post-NMS prediction overlays.
This is disabled by default because a full transfer matrix would otherwise
create many images.  These settings affect drawing only, never AP evaluation.
"""


SAVE_PREDICTION_VISUALIZATIONS = False
PREDICTION_VISUALIZATION_SCORE_THRESHOLD = 0.30
PREDICTION_VISUALIZATION_MAX_IMAGES = 20
PREDICTION_VISUALIZATION_MAX_DETECTIONS = 100
