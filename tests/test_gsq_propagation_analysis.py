import json

import pytest

from scripts.analyze_gsq_fp8_propagation import analyze


def test_packed_comparison_uses_matched_baseline_and_teacher(tmp_path):
    def row(value, teacher):
        return dict(tokens=2, teacher_sha256=teacher, kl_teacher_candidate=value, mse=value,
                    top1_agreement=1-value, top5_agreement=1-value, top10_agreement=1-value)
    report = dict(state='complete', arms={
        'baseline': {'rows': [row(.1, 'a'), row(.1, 'b')]},
        'baseline_packed': {'rows': [row(.5, 'a'), row(.5, 'b')]},
        'staged_packed': {'rows': [row(.3, 'a'), row(.3, 'b')]}})
    path = tmp_path/'report.json'
    path.write_text(json.dumps(report))
    result = analyze(path, arms=['staged_packed'], baseline_arm='baseline_packed')
    assert result['baseline'] == 'baseline_packed'
    assert all(metric['classification'] == 'clear positive'
               for metric in result['arms']['staged_packed'].values())
    assert result['arms']['staged_packed']['mse']['candidate_minus_baseline'] == pytest.approx(-.2)
    report['arms']['staged_packed']['rows'][1]['teacher_sha256'] = 'different-document'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Mismatched paired documents'):
        analyze(path, arms=['staged_packed'], baseline_arm='baseline_packed')
