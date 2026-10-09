"""Render existing prediction archives offline; never run a detector."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cell-width', type=int, default=512)
    parser.add_argument('--cell-height', type=int, default=384)
    parser.add_argument('--font', type=Path)
    parser.add_argument('--no-pdf', action='store_true')
    args = parser.parse_args()
    from lgp.reporting.qualitative_inputs import load_panel_request
    plan = load_panel_request(args.request.absolute(), args.request_sha256, max_images=args.max_images)
    if args.plan_only:
        print(json.dumps(dict(status='offline_panel_inputs_verified_no_output_written',
            image_ids=plan['image_ids'], rows=[r['label'] for r in plan['rows']],
            evidence_files=len(plan['evidence_inputs']), display_smoke=plan['display_smoke'],
            model_calls=0, AP_replays=0), indent=2))
        return
    from lgp.reporting.qualitative_panel import render_panel
    output = args.output or PROJECT_ROOT / 'outputs/qualitative_panels' / datetime.now(
        timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    result = render_panel(plan, output, cell_width=args.cell_width, cell_height=args.cell_height,
        font_path=args.font, pdf=not args.no_pdf)
    print(json.dumps(dict(status=result['status'], output=str(output.absolute()),
        image_ids=result['image_ids'], method_order=result['method_order'], model_calls=0, AP_replays=0), indent=2))


if __name__ == '__main__':
    main()
