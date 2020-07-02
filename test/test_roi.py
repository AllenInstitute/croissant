import croissant.roi as ROI
import pytest


@pytest.fixture(scope='module')
def roi_dict():
    d = {
            'roi_id': 'abc',
            'coo_rows': [2, 4, 6],
            'coo_cols': [12, 14, 16],
            'coo_data': [1.2, 2.3, 3.4],
            'image_shape': (512, 512),
            'depth': 125,
            'full_genotype': 'wild',
            'targeted_structure': 'the_brain',
            'rig': 'The_Scope',
            'trace': [0.1, 0.1, 0.1, 1.0, 0.9, 0.7, 0.5, 0.2, 0.1],
            'label': True
            }
    yield d


def test_RoiWithMetadata(roi_dict):
    roi = ROI.RoiWithMetadata.from_dict(roi_dict)
    for k in ['roi_id', 'coo_rows', 'coo_cols', 'coo_data', 'image_shape']:
        assert roi_dict[k] == roi.roi[k]
    for k in ['depth', 'full_genotype', 'targeted_structure', 'rig']:
        assert roi_dict[k] == roi.roi_meta[k]
    assert roi_dict['trace'] == roi.trace
    assert roi_dict['label'] == roi.label
