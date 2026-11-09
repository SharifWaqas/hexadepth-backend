import pytest
from hexadepth.comparison import compare, with_metadata, REFERENCES

def test_first_view_vs_full_uses_same_rate_and_frame_size():
    r=compare(64*10,64*100,64*1000,1200)
    assert r['full_file_seconds'] > r['detail_seconds'] > r['first_view_seconds']
    assert r['same_64_byte_framing'] and not r['quality_equivalence']
    assert not r['hardware_tested']
    assert r['first_view_advantage'] > 50

def test_exact_nominal_frames_calculation():
    r=compare(640,640,6400,2400)
    assert r['first_view_seconds']==pytest.approx(.48+640*8/2400,abs=.001)
    assert r['full_file_seconds']==pytest.approx(.48+6400*8/2400,abs=.001)

def test_reject_invalid():
    for args in ((0,2,3,1000),(20,10,30,1000),(20,30,40,0)):
        with pytest.raises(ValueError):compare(*args)

def test_references_unique_and_grounded():
    assert len({r.id for r in REFERENCES})==len(REFERENCES)
    assert all(r.bps>0 and r.reference.startswith('https://') for r in REFERENCES)
    assert with_metadata(compare(64,128,640,1200))['modem_reference']['id']=='benthos_mfsk_1200'
