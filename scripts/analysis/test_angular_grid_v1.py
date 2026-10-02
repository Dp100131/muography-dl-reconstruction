"""Run from the patched repository root using unittest discovery."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/simulation'))
from surrogate import imaging


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


converter = module('converter', ROOT/'scripts/analysis/convertir_tablas_azimut.py')
builder = module('builder', ROOT/'scripts/analysis/build_generator_tables.py')


class AngularGridTests(unittest.TestCase):
    def test_centers_and_unmodified_elevation(self):
        az, el = imaging.angular_grid()
        np.testing.assert_allclose(np.rad2deg(az), -30+(np.arange(256)+0.5)*60/256, atol=1e-12)
        np.testing.assert_allclose(np.rad2deg(el), np.linspace(2, 47, 256), atol=1e-12)
        self.assertAlmostEqual(np.rad2deg(az[0]), -29.8828125)
        self.assertAlmostEqual(np.rad2deg(az[-1]), 29.8828125)

    def test_uniform_source_one_million_directions(self):
        az, _ = imaging.angular_grid()
        centers = np.rad2deg(az)
        edges = np.r_[centers[0]-60/512, (centers[:-1]+centers[1:])/2, centers[-1]+60/512]
        samples = -30+60*(np.arange(1048576)+0.5)/1048576
        counts, _ = np.histogram(samples, edges)
        np.testing.assert_array_equal(counts, np.full(256, 4096))

    def test_expected_counts_solid_angle_contract(self):
        # Exercise the ACTUAL surrogate count function with a constant flux.
        shape = (256, 256)
        with patch.object(imaging, 'opacity_map', return_value=np.zeros(shape)), \
             patch.object(imaging, 'air_baseline_opacity', return_value=np.zeros(shape)), \
             patch.object(imaging, 'flux_table', return_value=lambda e,c: np.ones(shape)):
            lam, transmission, _ = imaging.expected_counts(None, None, 30)
        el = np.deg2rad(np.linspace(2,47,256))
        expected = 1.44e4*30*86400*np.deg2rad(60)*np.deg2rad(45/255)*np.cos(el).sum()
        self.assertAlmostEqual(lam.sum()/expected, 1, places=12)
        np.testing.assert_array_equal(lam[:,0], lam[:,-1])
        np.testing.assert_array_equal(transmission, np.ones(shape))

    def test_table_conversion_and_no_double_scaling(self):
        src = ROOT/'scripts/simulation/geant4_app/tables'
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'new'
            meta = converter.convert(src, dest)
            old, new = np.loadtxt(src/'row_weights_30d.txt'), np.loadtxt(dest/'row_weights_30d.txt')
            np.testing.assert_array_equal(old[:,0], new[:,0])
            np.testing.assert_allclose(new[:,1]/old[:,1], 255/256, rtol=1e-15)
            self.assertEqual((src/'p_cdf.txt').read_bytes(), (dest/'p_cdf.txt').read_bytes())
            self.assertAlmostEqual(new[:,1].sum()/meta['total_open_sky_counts_30days_per_view'], 1, places=13)
            with self.assertRaises(FileExistsError): converter.convert(src, dest)
            with self.assertRaises(ValueError): converter.convert(dest, Path(tmp)/'double')
            self.assertFalse((Path(tmp)/'double').exists())

    def test_builder_rejects_historical_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'view.npz'
            path.with_suffix('.meta.json').write_text('{}')
            with patch.object(builder, 'CAMPAIGN_VIEW', path):
                with self.assertRaises(ValueError): builder.build_row_weights()

    def test_builder_accepts_new_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'view.npz'
            np.savez(path, lam=np.full((256,256),2.0), transmission=np.ones((256,256)))
            path.with_suffix('.meta.json').write_text(json.dumps({'exposure_days':30,'angular_grid':{
                'azimuth_convention':'az-edges-v1','n_az':256,'n_el':256,
                'az_edges_deg':[-30.0,30.0],'az_step_deg':60/256}}))
            with patch.object(builder, 'CAMPAIGN_VIEW', path):
                np.testing.assert_array_equal(builder.build_row_weights(), np.full(256,512))


if __name__ == '__main__':
    unittest.main()
