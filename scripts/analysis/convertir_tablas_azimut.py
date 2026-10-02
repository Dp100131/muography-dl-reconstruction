"""Derive versioned azimuth tables from the exact audited historical tables.

Only the constant azimuth-width factor changes (255/256). This does not
validate physical exposure or convert any transported histogram/ROI.
Usage: python convertir_tablas_azimut.py SOURCE_TABLES NEW_DIRECTORY
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

EXPECTED = {
    'row_weights_30d.txt': '93ce67abe51c59ea3ac5c07dc99fd1615ae147df82c16d1bad64f5c0712d2c0c',
    'p_cdf.txt': '7161c8adf11711139840007dff26178221c4cfbff8fc3253cabdda50c533f6ae',
    'meta.json': '7ac4b6a9353af701b56e5a97ee783ed59072d58df52d80ddbdc0f5f70147d35d',
}


def convert(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError('destination must be new; no in-place conversions')
    for name, expected in EXPECTED.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'unrecognized or already converted source: {name}')
    rows = [tuple(map(float, line.split())) for line in
            (source/'row_weights_30d.txt').read_text().splitlines()
            if line.strip() and not line.startswith('#')]
    if len(rows) != 256:
        raise ValueError('expected 256 elevation rows')
    factor = 255/256
    text = '# elev_center_deg lambda_open_counts_30days_per_row; az-edges-v1\n'
    text += ''.join(f'{e:.6f} {w*factor:.17g}\n' for e, w in rows)
    meta = json.loads((source/'meta.json').read_text())
    meta.update(azimuth_convention='az-edges-v1', n_azimuth_columns=256,
                azimuth_min_deg=-30.0, azimuth_max_deg=30.0,
                total_open_sky_counts_30days_per_view=sum(w*factor for _, w in rows),
                derivation={'source_sha256': EXPECTED, 'row_scale': factor,
                            'reason': 'azimuth width changes from 60/255 to 60/256',
                            'scope': 'numerical convention only; physical exposure remains unvalidated'})
    destination.mkdir(parents=True)
    (destination/'row_weights_30d.txt').write_text(text, encoding='ascii', newline='\n')
    shutil.copyfile(source/'p_cdf.txt', destination/'p_cdf.txt')
    (destination/'meta.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8', newline='\n')
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    a = p.parse_args()
    print(json.dumps(convert(a.source, a.destination), indent=2))
