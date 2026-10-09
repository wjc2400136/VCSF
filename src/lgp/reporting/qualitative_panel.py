"""Render offline, provenance-bound qualitative panels without model calls."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..visualization import render_detection_overlay
from .qualitative_inputs import verify_panel_inputs


def _font(size, font_path=None):
    if font_path is not None:
        return ImageFont.truetype(str(font_path), size=size)
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size=size)
    except OSError:
        return ImageFont.load_default()


def _lines(draw, text, width, font):
    lines = []
    for paragraph in text.split('\n'):
        line = ''
        for word in paragraph.split():
            candidate = (line + ' ' + word) if line else word
            if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
                line = candidate
                continue
            if line:
                lines.append(line)
                line = ''
            for character in word:
                if line and draw.textbbox((0, 0), line + character, font=font)[2] > width:
                    lines.append(line)
                    line = ''
                line += character
        lines.append(line)
    line_height = draw.textbbox((0, 0), 'Ag', font=font)[3] + 6
    return lines, line_height


def _wrapped(draw, text, box, font):
    x, y, width, height = box
    lines, line_height = _lines(draw, text, width, font)
    if line_height * len(lines) > height:
        raise ValueError('Label does not fit the panel; shorten it or increase cell dimensions')
    for index, value in enumerate(lines):
        draw.text((x, y + index * line_height), value, fill='#171717', font=font)


def _digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def render_panel(plan, output, *, cell_width=512, cell_height=384, font_path=None, pdf=True):
    """Keep original pixels untouched; output copies/overlays are derived assets."""
    for value in (cell_width, cell_height):
        if type(value) is not int or not 128 <= value <= 1536:
            raise ValueError('Cell dimensions must be integers between 128 and 1536')
    if type(pdf) is not bool:
        raise ValueError('pdf must be boolean')
    output = Path(output).absolute()
    if '..' in output.parts or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('Output must not contain symlinks or parent traversal')
    for reference in plan['evidence_inputs']:
        path = Path(reference['file'])
        if path.name in ('metrics.json', 'run.json') and path.parent in (output, *output.parents):
            raise ValueError('Do not write derived figures into an existing evaluation or generation')
    verify_panel_inputs(plan)
    row_budgets = [
        None if row['attack'] == 'clean' else (
            row.get('budget_profile', plan['budget_profile'])
            if plan['schema_version'] == 1 else row['budget_profile'])
        for row in plan['rows']]
    budget_binding = plan.get('budget_binding', 'shared_request')
    font_reference = None
    if font_path is not None:
        font_path = Path(font_path).absolute()
        if '..' in font_path.parts or any(p.is_symlink() for p in (font_path, *font_path.parents)):
            raise ValueError('Font path must not contain symlinks or parent traversal')
        font_reference = dict(file=str(font_path), sha256=_digest(font_path))
        _font(20, font_path)
    multiplier = 2 if plan['originals'] else 1
    columns = len(plan['image_ids']) * multiplier
    gap, margin, labels = 12, 16, 196
    width = margin * 2 + labels + columns * cell_width + max(0, columns - 1) * gap
    font, small = _font(20, font_path), _font(16, font_path)
    measure = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    title = '{} {} | {} -> {}'.format(plan['dataset'], plan['split'], plan['source'], plan['target'])
    title_lines, title_line_height = _lines(measure, title, width - 2 * margin, font)
    title_height = len(title_lines) * title_line_height
    column_labels = ['Image {}\n{}'.format(image_id,
        'Original' if plan['originals'] and offset == 0 else 'Predictions')
        for image_id in plan['image_ids'] for offset in range(multiplier)]
    column_height = max(len(_lines(measure, value, cell_width, small)[0])
                        * _lines(measure, value, cell_width, small)[1] for value in column_labels)
    column_y = margin + title_height + gap
    heading = max(122, column_y + column_height + gap)
    footer = 'Display threshold: {:.4f} | Maximum detections: {} | Scores are unchanged predictions'.format(
        plan['score_threshold'], plan['max_detections'])
    footer_lines, footer_line_height = _lines(measure, footer, width - 2 * margin, small)
    footer_height = max(52, len(footer_lines) * footer_line_height + margin * 2)
    height = heading + len(plan['rows']) * (cell_height + gap) + footer_height
    if width * height > 64 * 1024 * 1024:
        raise ValueError('Panel exceeds the 64-megapixel canvas limit')
    output.mkdir(parents=True, exist_ok=False)
    try:
        canvas = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(canvas)
        _wrapped(draw, title, (margin, margin, width - 2 * margin, title_height), font)
        for index, label in enumerate(column_labels):
            x = margin + labels + index * (cell_width + gap)
            _wrapped(draw, label, (x, column_y, cell_width, column_height), small)
        cells = []
        for row_index, (row, row_budget) in enumerate(zip(plan['rows'], row_budgets)):
            y = heading + row_index * (cell_height + gap)
            label = row['label']
            if plan['schema_version'] == 2 and row_budget is not None:
                label += '\nBudget: ' + row_budget.replace('_', ' ')
            _wrapped(draw, label, (margin, y + 8, labels - 18, cell_height - 16), font)
            if [c['image_id'] for c in row['cells']] != plan['image_ids']:
                raise ValueError('Panel row image order differs')
            for column, cell in enumerate(row['cells']):
                prefix = 'r{:02d}_i{:012d}'.format(row_index, cell['image_id'])
                original = output / 'cells' / (prefix + '_original.png')
                overlay = output / 'cells' / (prefix + '_predictions.png')
                original.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(cell['image']['file']) as source:
                    image = ImageOps.exif_transpose(source).convert('RGB')
                if image.size != (cell['width'], cell['height']):
                    raise ValueError('Decoded image dimensions changed')
                image.save(original, format='PNG')
                display_scale = min(cell_width / image.width, cell_height / image.height)
                overlay_font_size = max(1, int(round(16 / display_scale)))
                result = render_detection_overlay(original, overlay, cell['boxes'], cell['scores'],
                    cell['labels'], plan['class_names'], score_threshold=plan['score_threshold'],
                    max_detections=plan['max_detections'], font_size=overlay_font_size)
                paths = [original, overlay] if plan['originals'] else [overlay]
                for offset, path in enumerate(paths):
                    index = column * multiplier + offset
                    x = margin + labels + index * (cell_width + gap)
                    with Image.open(path) as source:
                        contained = ImageOps.contain(source.convert('RGB'), (cell_width, cell_height), Image.LANCZOS)
                    canvas.paste(contained, (x + (cell_width - contained.width) // 2,
                        y + (cell_height - contained.height) // 2))
                cells.append(dict(row=row_index, label=row['label'], attack=row['attack'],
                    budget_profile=row_budget,
                    image_id=cell['image_id'], input_image=cell['image'],
                    original_file=str(original.relative_to(output)), original_sha256=_digest(original),
                    overlay_file=str(overlay.relative_to(output)), overlay_sha256=_digest(overlay),
                    detections_available=cell['detections_available'],
                    detections_selected=cell['detections_selected'], detections_drawn=result['detections_drawn'],
                    requested_overlay_font_size=overlay_font_size,
                    caption_layout_limitations=result['caption_layout_limitations']))
        _wrapped(draw, footer, (margin, height - footer_height + margin,
            width - 2 * margin, footer_height - 2 * margin), small)
        canvas.save(output / 'panel.png', format='PNG')
        files = {'panel.png': _digest(output / 'panel.png')}
        if pdf:
            canvas.save(output / 'panel.pdf', format='PDF', resolution=300.0)
            files['panel.pdf'] = _digest(output / 'panel.pdf')
        verify_panel_inputs(plan)
        if font_reference is not None and _digest(font_path) != font_reference['sha256']:
            raise ValueError('Font changed while rendering')
        manifest = dict(status='offline_qualitative_panel_rendered_not_scientifically_accepted',
            dataset=plan['dataset'], split=plan['split'], source=plan['source'], target=plan['target'],
            image_ids=plan['image_ids'], method_order=[row['attack'] for row in plan['rows']],
            selection_note=plan['selection_note'], score_threshold=plan['score_threshold'],
            max_detections=plan['max_detections'], originals=plan['originals'],
            dimensions=[width, height], cell_dimensions=[cell_width, cell_height],
            evidence_sha256=plan['evidence_sha256'], evidence_inputs=plan['evidence_inputs'],
            request_sha256=plan['request_sha256'], display_smoke=plan['display_smoke'],
            requested_image_ids=plan['requested_image_ids'], evaluation_images=plan['evaluation_images'],
            checkpoint_sha256=plan['checkpoint_sha256'], budget_profile=plan['budget_profile'],
            budget_binding=budget_binding,
            row_budget_profiles=[dict(attack=row['attack'], budget_profile=budget)
                                for row, budget in zip(plan['rows'], row_budgets)],
            image_coordinate_contract=plan['image_coordinate_contract'],
            files_sha256=files, cells=cells,
            model_calls=0, AP_replays=0, scientific_acceptance=False,
            formal_metrics_verified=False)
        if font_reference is not None:
            manifest['font_reference'] = font_reference
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
        return manifest
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps(dict(error=repr(error), scientific_acceptance=False)),
            encoding='utf-8')
        raise
