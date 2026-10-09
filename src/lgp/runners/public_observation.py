"""Exact read-only observer relocated from the original qualification tool."""
from __future__ import annotations

import weakref
import torch

from .vcsf_research_plan import require

class ObservedAdapter:
    """Observe existing adapter calls and input-gradient events without replacing outputs."""

    OBSERVED = {"extract_features", "prediction_scores", "pre_nms_prediction_scores"}

    def __init__(self, adapter):
        self.delegate = adapter
        self.calls = []
        self.backward_events = []
        self._handles = []
        self._seen = []

    def __getattr__(self, name):
        value = getattr(self.delegate, name)
        if name not in self.OBSERVED or not callable(value):
            return value

        def observed(raw, *args, **kwargs):
            kind = "feature" if name == "extract_features" else "detector_initialization_attempt"
            self.calls.append(dict(api=name, kind=kind, batch_rows=int(raw.shape[0]),
                                   shape=list(raw.shape), requires_grad=bool(raw.requires_grad)))
            if raw.requires_grad and not any(reference() is raw for reference in self._seen):
                self._seen.append(weakref.ref(raw))

                def gradient_seen(gradient):
                    self.backward_events.append(dict(kind=kind, shape=list(gradient.shape),
                        batch_rows=int(gradient.shape[0]), finite=bool(torch.isfinite(gradient).all().item())))
                    return None

                self._handles.append(raw.register_hook(gradient_seen))
            return value(raw, *args, **kwargs)

        return observed

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._seen.clear()

    def summary(self, trajectory):
        feature = [row for row in self.calls if row["kind"] == "feature"]
        risk = [row for row in self.calls if row["kind"] != "feature"]
        feature_backward = [row for row in self.backward_events if row["kind"] == "feature"]
        risk_backward = [row for row in self.backward_events if row["kind"] != "feature"]
        require(len(feature) == len(feature_backward) == 19
                and all(row["batch_rows"] == 2 and row["requires_grad"] for row in feature)
                and len(risk) in (1, 2) and all(row["batch_rows"] == 1 for row in risk)
                and len(risk_backward) == 1
                and all(row["finite"] for row in self.backward_events),
                "Observed adapter invocation or input-gradient event schedule differs")
        terms = sum(row["feature_surface_level_count"] for row in trajectory[1:])
        require(terms == 114, "Observed trajectory feature term count differs")
        return dict(observed_adapter_calls=self.calls,
                    observed_adapter_input_gradient_events=self.backward_events,
                    observed_feature_batch_forward_calls=len(feature),
                    observed_feature_batch_rows=sum(row["batch_rows"] for row in feature),
                    observed_detector_initialization_forward_attempts=len(risk),
                    observed_feature_input_backward_events=len(feature_backward),
                    observed_detector_input_backward_events=len(risk_backward),
                    observed_feature_terms_from_trajectory=terms,
                    derived_clean_reference_forward_image_rows=len(feature),
                    derived_adversarial_differentiable_feature_views=len(feature),
                    derived_feature_partial_backward_depth="backbone_plus_neck_not_detection_head",
                    row_role_assignment="frozen_code_cat_clean_detached_then_adversarial",
                    declared_auxiliary_forward_passes=0,
                    auxiliary_zero_excludes_regular_batched_clean_reference_rows=True,
                    calibrated_whole_detector_BE=None, FLOPs=None,
                    observation_scope="adapter_API_calls_and_input_gradient_hooks_not_physical_kernel_accounting")
