"""Prepare full dataset input views or re-read them in a separate invocation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from audit_vcsf_cpu_analysis import Evidence, check, child, plain
from run_vcsf_core_points import bound_registry, source_binding
from lgp.io import atomic_json, file_digest
from lgp.runners.vcsf_final_followup_plan import FREEZE_SHA256, PROTOCOL, prepare_final_followup_plan
from lgp.runners.vcsf_final_followup_inputs import project_group_inputs
from lgp.runners.vcsf_research_plan import canonical_hash
from tools.vcsf_final_followup_asset_inputs import collect_dataset_inputs, verify_saved_dataset_inputs

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'experiments/prepare_vcsf_final_followup_inputs.py'
FREEZE = 'docs/research/vcsf-final-method-freeze-20260917.json'
NAMESPACE = 'vcsf_final_followup_inputs'


def _publication(read, reference, namespace):
    path = plain(Path(reference['file']))
    check(path.name == 'completion.json' and path.parents[1].name == namespace and
        path.parents[2].name == 'plans' and path.parents[3].name == 'outputs',
        'Preparation completion namespace differs')
    value = read.json(path, reference['sha256'])
    check(isinstance(value['source_sha256'], dict) and value['source_sha256'],
        'Missing producer source identity')
    for name, digest in value['source_sha256'].items():
        read.bytes(child(path.parents[4], name), digest)
    check(not (path.parent / 'failure.json').exists(), 'Failed preparation cannot be consumed')
    return path.parent, value


def _catalogue(read, registry, preparation, input_report):
    root, completion = _publication(read, preparation, 'vcsf_final_followup')
    check(completion['status'] == 'prepared_final_followup_pending_qualification_and_admission' and
        completion['freeze_sha256'] == FREEZE_SHA256 and
        all(completion[k] is False for k in ('runner_armed', 'formal_execution_admission', 'reuse_accepted')),
        'Preparation has incompatible identity or acceptance scope')
    artifacts = completion['artifacts_sha256']
    check(set(artifacts) in ({'catalogue.json'}, {'catalogue.json', 'seed42_observation.json'}) and
        {p.name for p in root.iterdir()} == set(artifacts) | {'completion.json'},
        'Preparation artifact set differs')
    for name, digest in artifacts.items():
        read.bytes(child(root, name), digest)
    saved = read.json(root / 'catalogue.json', artifacts['catalogue.json'])
    frozen = read.bytes(ROOT / FREEZE, FREEZE_SHA256, 1 << 20)
    freeze = json.loads(frozen)
    for name, digest in freeze['review_records_sha256'].items():
        read.bytes(ROOT / name, digest)
    check(completion['input_report_sha256'] == freeze['authenticated_input_report']['sha256'],
        'Preparation used another authenticated input report')
    report = read.bytes(plain(Path(input_report)), completion['input_report_sha256'], 128 << 20)
    current = prepare_final_followup_plan(registry, frozen, report, saved['devices'])
    check(canonical_hash(current) == canonical_hash(saved), 'Prepared catalogue differs from reconstruction')
    return saved


def main(argv=None, *, entry_sha256=None, bootstrap_sha256=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['coco', 'voc'])
    parser.add_argument('--preparation', type=Path)
    parser.add_argument('--preparation-sha256')
    parser.add_argument('--input-report', type=Path)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--publication', type=Path)
    parser.add_argument('--publication-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--max-images', type=int,
        help='Rejected: input binding requires the registered full image set.')
    args = parser.parse_args(argv)
    output, owned = None, False
    try:
        check(sys.platform == 'linux' and Path(sys.prefix).name == 'oda' and sys.executable == str(Path(sys.prefix) / "bin" / "python"),
            'Require authorized server ODA')
        check(args.max_images is None, 'Limited-image input bindings are not admitted')
        read = Evidence()
        source = source_binding(ENTRY, entry_sha256, bootstrap_sha256)
        for name, digest in source.items():
            read.bytes(ROOT / name, digest)
        def refreshed_source():
            current = source_binding(ENTRY, entry_sha256, bootstrap_sha256)
            check(all(current.get(name) == digest for name, digest in source.items()),
                'Consumer source changed during asset reconstruction')
            for name, digest in current.items():
                read.bytes(ROOT / name, digest)
            read.unchanged()
            return current
        registry = bound_registry(read)
        supplied = (args.preparation, args.preparation_sha256, args.input_report,
            args.publication, args.publication_sha256, args.output)
        if args.plan_only:
            check(not any(supplied), 'Plan-only takes no evidence or output paths')
            read.unchanged()
            print(json.dumps(dict(status='final_followup_input_scope_only', audit=args.audit,
                datasets=registry.protocols[PROTOCOL]['datasets'], dataset=args.dataset,
                shared_model_roles=22, group_model_roles=17, reads_actual_assets=False,
                model_calls=0, runner_armed=False, formal_execution_admission=False)))
            return 0
        check(args.output is not None, 'Supply a fresh output directory')
        output = plain(args.output.absolute())
        category = 'audits' if args.audit else 'plans'
        check(output.parent == ROOT / 'outputs' / category / NAMESPACE and not output.exists(),
            'Use a fresh direct output leaf in the input preparation/audit namespace')
        if args.audit:
            check(args.publication is not None and args.publication_sha256 and
                not any((args.dataset, args.preparation, args.preparation_sha256, args.input_report)),
                'Audit takes only an exact publication reference and output')
            publication = dict(file=str(plain(args.publication.absolute())), sha256=args.publication_sha256)
            root, completion = _publication(read, publication, NAMESPACE)
            check({p.name for p in root.iterdir()} ==
                {'assets', 'shared_inputs.json', 'group_views.json', 'completion.json'},
                'Input publication contains unexpected or incomplete artifacts')
            check(completion['status'] == 'final_followup_dataset_inputs_pending_independent_readback',
                'Publication is not a prepared dataset input binding')
            check(all(completion[k] is False for k in ('reuse_accepted', 'formal_execution_admission',
                'checkpoint_qualification_accepted', 'implementation_bridge_accepted')),
                'Publication changed its acceptance scope')
            request = completion['request']
            plan = _catalogue(read, registry, request['preparation'], request['input_report'])
            artifacts = completion['artifacts_sha256']
            check(set(artifacts) == {'shared_inputs.json', 'group_views.json'}, 'Input publication files differ')
            shared = read.json(root / 'shared_inputs.json', artifacts['shared_inputs.json'])
            check(plain(Path(shared['dataset_output_directory'])) == root / 'assets' and
                shared['dataset'] == request['dataset'], 'Asset directory or dataset escaped publication')
            audit, rows = verify_saved_dataset_inputs(read, registry, shared)
            groups = [g for g in plan['groups'] if g['dataset'] == request['dataset']]
            views = [project_group_inputs(registry, g, shared, rows) for g in groups]
            check(type(completion['groups']) is int and completion['groups'] == len(views) and
                read.json(root / 'group_views.json', artifacts['group_views.json']) == views,
                'Saved group input views differ from actual reconstruction')
            complete_source = refreshed_source()
            output.mkdir(parents=True, exist_ok=False)
            owned = True
            result = dict(audit, publication_reference=publication, source_sha256=complete_source,
                group_views_content_sha256=canonical_hash(views), groups=len(views),
                actual_asset_reconstruction_verified=True, created_at=datetime.now(timezone.utc).isoformat())
            atomic_json(output / 'receipt.json', result)
        else:
            check(args.dataset is not None and args.preparation is not None and args.preparation_sha256 and
                args.input_report is not None and not any((args.publication, args.publication_sha256)),
                'Supply dataset, preparation reference, original input report and output')
            request = dict(dataset=args.dataset,
                preparation=dict(file=str(plain(args.preparation.absolute())), sha256=args.preparation_sha256),
                input_report=str(plain(args.input_report.absolute())))
            plan = _catalogue(read, registry, request['preparation'], request['input_report'])
            output.mkdir(parents=True, exist_ok=False)
            owned = True
            assets = output / 'assets'
            assets.mkdir()
            shared, rows = collect_dataset_inputs(read, registry, args.dataset, assets)
            views = [project_group_inputs(registry, g, shared, rows)
                for g in plan['groups'] if g['dataset'] == args.dataset]
            atomic_json(output / 'shared_inputs.json', shared)
            atomic_json(output / 'group_views.json', views)
            complete_source = refreshed_source()
            result = dict(status='final_followup_dataset_inputs_pending_independent_readback',
                request=request, source_sha256=complete_source,
                artifacts_sha256={name: file_digest(output / name)
                    for name in ('shared_inputs.json', 'group_views.json')},
                groups=len(views), created_at=datetime.now(timezone.utc).isoformat(),
                checkpoint_qualification_accepted=False, implementation_bridge_accepted=False,
                reuse_accepted=False, formal_execution_admission=False, model_calls=0, AP_replays=0)
            atomic_json(output / 'completion.json', result)
        print(json.dumps(dict(status=result['status'], groups=result['groups'], output=str(output))))
        return 0
    except Exception as exc:
        failure = dict(status='final_followup_input_preparation_failed', error=str(exc))
        if owned:
            atomic_json(output / 'failure.json', failure)
        print(json.dumps(failure), file=sys.stderr)
        return 1
