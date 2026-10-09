"""Exact nonnumeric helpers relocated from the original author tooling."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

from .vcsf_research_plan import canonical_hash, require

def check(condition, message):
    if not condition:
        raise RuntimeError(message)

def _normal(value):
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), "Model config has a non-string key")
        return {key: _normal(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normal(item) for item in value]
    require(value is None or type(value) in (str, bool, int, float),
        "Model config contains a non-declarative value")
    require(type(value) is not float or math.isfinite(value), "Model config contains a non-finite value")
    return value

def normalized_model_config(value):
    if isinstance(value, dict):
        return {key: normalized_model_config(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized_model_config(item) for item in value]
    if isinstance(value, str) and value.startswith('/'):
        return str(Path(value).resolve())
    return value

def verify_loaded_config(config, prepared):
    expected = deepcopy(prepared.to_dict())
    observed = deepcopy(config.to_dict())
    # MMDetection 3.0.0 init_detector clears only this initializer before loading.
    if 'init_cfg' in expected['model']['backbone']:
        expected['model']['backbone']['init_cfg'] = None
    model = expected['model']
    if model.get('type') in ('DeformableDETR', 'DINO'):
        for section, attention in (('encoder', 'self_attn_cfg'),
                ('decoder', 'self_attn_cfg'), ('decoder', 'cross_attn_cfg')):
            attention_config = model.get(section, {}).get('layer_cfg', {}).get(attention)
            if attention_config is not None:
                check(attention_config.get('batch_first', True) is True,
                    'Deformable attention constructor requires batch_first=True')
                attention_config['batch_first'] = True
        head = model['bbox_head']
        derived = ('share_pred_layer', 'num_pred_layer', 'as_two_stage')
        check(not any(key in head for key in derived),
            'Deformable DETR head contains constructor-owned fields before loading')
        refine = model.get('with_box_refine', False)
        two_stage = model.get('as_two_stage', False)
        head.update(share_pred_layer=not refine,
            num_pred_layer=model['decoder']['num_layers'] + int(two_stage),
            as_two_stage=two_stage)
        if model['type'] == 'DINO':
            check(refine and two_stage, 'DINO requires refinement and two-stage construction')
            denoising = model.get('dn_cfg')
            if denoising is not None:
                check(not any(key in denoising for key in ('num_classes', 'num_queries', 'hidden_dim')),
                    'DINO denoising configuration violates constructor preconditions')
                denoising.update(num_classes=head['num_classes'],
                    embed_dims=2 * model['positional_encoding']['num_feats'],
                    num_matching_queries=model['num_queries'])
    if model.get('roi_head') is not None:
        model['roi_head']['train_cfg'] = deepcopy(model['train_cfg']['rcnn']) if model.get('train_cfg') is not None else None
        model['roi_head']['test_cfg'] = deepcopy(model['test_cfg']['rcnn'])
    if model.get('bbox_head') is not None:
        model['bbox_head']['train_cfg'] = deepcopy(model.get('train_cfg'))
        model['bbox_head']['test_cfg'] = deepcopy(model.get('test_cfg'))
        head = model['bbox_head']
        loss = head.get('loss_cls', {})
        if head.get('type') == 'DETRHead' and loss.get('class_weight') is not None:
            import torch
            weight = loss['class_weight']
            background = loss.get('bg_cls_weight', weight)
            check(type(weight) is float and type(background) is float,
                'DETR scalar class weights differ from constructor contract')
            check(torch.get_default_dtype() == torch.float32,
                'DETR constructor default dtype changed')
            reference = torch.ones(head['num_classes'] + 1, dtype=torch.float32) * weight
            reference[head['num_classes']] = background
            actual = observed['model']['bbox_head']['loss_cls'].get('class_weight')
            check(isinstance(actual, torch.Tensor) and actual.device.type == 'cpu'
                and actual.dtype == reference.dtype and actual.shape == reference.shape
                and not actual.requires_grad and torch.equal(actual, reference),
                'DETR loaded class-weight tensor differs from exact constructor transform')
            # Serialize only this verified constructor tensor, not arbitrary config values.
            descriptor = dict(constructor_tensor_dtype=str(reference.dtype),
                shape=list(reference.shape), values=reference.tolist())
            loss['class_weight'] = descriptor
            loss.pop('bg_cls_weight', None)
            observed['model']['bbox_head']['loss_cls']['class_weight'] = deepcopy(descriptor)
    expected.pop('work_dir', None)
    observed.pop('work_dir', None)
    expected_hash = canonical_hash(normalized_model_config(_normal(expected)))
    actual_hash = canonical_hash(normalized_model_config(_normal(observed)))
    check(actual_hash == expected_hash, 'Loaded model effective configuration differs from pinned inputs')
    return actual_hash
