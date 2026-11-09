"""Priority-first transmission: exact framed bytes, degraded preview and safe unknown fallback."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from hexadepth import priority as pr
from hexadepth import file_transport as ft
from server import app


@pytest.fixture
def unseen_image():
    # No training images / labels: a synthetic sonar-like scene with texture.
    rng = np.random.default_rng(619)
    img = rng.integers(0, 50, size=(240, 320), dtype=np.uint8)
    cv2.ellipse(img, (210, 100), (30, 18), 22, 0, 360, 190, -1)
    ok, encoded = cv2.imencode('.png', img)
    assert ok
    return encoded.tobytes()


def test_metadata_roundtrip():
    regions=[pr.Region(10,20,50,80,210),pr.Region(90,100,200,220,180)]
    packed=pr.metadata_pack(320,240,regions,False)
    assert len(packed) == 30
    assert len(ft.frame_file(packed)) == 1
    w,h,trained,restored=pr.metadata_unpack(ft.reassemble_file(ft.frame_file(packed)))
    assert (w,h,trained)==(320,240,False)
    assert [r.as_dict() for r in restored]==[r.as_dict() for r in regions]


def test_bad_metadata_is_rejected():
    with pytest.raises(pr.PriorityError):pr.metadata_unpack(b'HDP1')
    with pytest.raises(pr.PriorityError):pr.metadata_unpack(pr.metadata_pack(320,240,[pr.Region(1,1,20,20,200)],False)+b'0')


def test_unseen_scene_reconstructs_without_model(unseen_image):
    r=pr.simulate_progressive(unseen_image,rate_bps=2400)
    assert r['ok']
    assert not r['model_used']
    assert r['original_bytes']==len(unseen_image)
    assert [s['id'] for s in r['stages']]==['metadata','overview','detail']
    assert all(s['integrity_ok'] and s['wire_bytes']%64==0 for s in r['stages'])
    assert 0 < r['overview_seconds'] < r['detail_seconds'] < r['full_file_estimate_seconds']
    assert len(r['regions']) > 0
    for s in r['stages'][1:]:
        assert s['image'].startswith('data:image/png;base64,')


@pytest.mark.parametrize('stage', ['metadata','overview','detail'])
def test_crc_rejects_bad_stage(unseen_image,stage):
    r=pr.simulate_progressive(unseen_image,corrupt_stage=stage)
    assert r['ok'] is False
    assert r['failed_stage']==stage
    assert 'CRC mismatch' in r['error']


def test_blank_image_no_invented_detection():
    blank=np.full((200,280),90,np.uint8)
    ok,buf=cv2.imencode('.png',blank)
    assert ok
    r=pr.simulate_progressive(buf.tobytes())
    assert r['ok'] and r['regions']==[] and not r['model_used']
    assert r['stages'][2]['packet_count']==0


def test_rate_changes_time_not_packet_size(unseen_image):
    slow=pr.simulate_progressive(unseen_image,rate_bps=1200)
    fast=pr.simulate_progressive(unseen_image,rate_bps=9600)
    assert slow['preview_wire_bytes']==fast['preview_wire_bytes']
    assert slow['overview_seconds']>fast['overview_seconds']
    assert slow['full_file_packets']==fast['full_file_packets']


def test_trained_box_path_does_not_turn_saliency_into_labels(unseen_image):
    shape=cv2.imdecode(np.frombuffer(unseen_image,np.uint8),cv2.IMREAD_GRAYSCALE)
    found=[{'box_px':[173,78,249,128],'confidence':75,'class_id':10}]
    regions,has_model=pr.select_regions(shape,found)
    assert has_model is True
    assert regions[0].class_id==10
    assert any(r.class_id==255 for r in regions[1:])


def test_api_with_real_image():
    c=TestClient(app)
    names=c.get('/api/samples').json()
    chosen=next(s['name'] for s in names if s['group']=='real')
    r=c.post('/api/priority',data={'sample':chosen,'rate_bps':'2400'})
    assert r.status_code==200, r.text
    d=r.json()
    assert d['ok'] and d['stages'][0]['packet_count']==1
    assert d['preview_wire_bytes']>0 and d['overview_seconds']<d['full_file_estimate_seconds']
    assert 'detector_note' in d


def test_invalid_input_rejected():
    c=TestClient(app)
    assert c.post('/api/priority',data={'rate_bps':2400}).status_code==400
    assert c.post('/api/priority',data={'sample':'../bad.png'}).status_code==404
    assert c.post('/api/priority', files={'file':('bad.png',b'not an image','image/png')}).status_code==400


def test_priority_page_is_served():
    c=TestClient(app)
    r=c.get('/priority.html')
    assert r.status_code==200
    assert 'Request exact image' in r.text
    assert c.get('/priority.js').status_code==200
