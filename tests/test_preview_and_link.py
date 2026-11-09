"""Model-independent compression and packetized acoustic-link simulator."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from hexadepth import link, preview, synth
from server import app

client = TestClient(app)


def test_64bit_preview_is_exactly_eight_bytes_and_deterministic():
    image = synth.render(synth.default_scenes()[0], seed=42)
    one = preview.encode_preview(image)
    two = preview.encode_preview(image)
    assert one == two and len(one) == 8
    small = preview.decode_preview(one)
    assert small.shape == (8, 8)
    assert set(np.unique(small)).issubset({0, 255})


def test_preview_handles_unfamiliar_and_blank_images_without_ai():
    rng = np.random.default_rng(191)
    unrelated = rng.integers(0, 256, (321, 183), dtype=np.uint8)
    assert len(preview.encode_preview(unrelated)) == 8
    flat = np.full((60, 120), 105, np.uint8)
    assert np.all(preview.decode_preview(preview.encode_preview(flat)) == 0)


def test_preview_rejects_invalid_inputs():
    for bad in [np.zeros((10, 10, 3), np.uint8), np.zeros((0, 3), np.uint8), np.zeros((10, 10), np.float32)]:
        with pytest.raises(ValueError):
            preview.encode_preview(bad)
    with pytest.raises(ValueError):
        preview.decode_preview(b"too short")


@pytest.mark.parametrize('payload,lanes', [
    (bytes(range(8)), 1), (bytes(range(8)), 2), (bytes(range(8)), 4),
    (bytes(range(64)), 1), (bytes(range(64)), 4), (bytes(range(9)), 4),
])
def test_fragment_round_trip(payload, lanes):
    frames = link.frame_fragments(payload, lanes)
    assert len(frames) == lanes
    assert all(len(frame) >= link.FRAGMENT_OVERHEAD_BYTES + 1 for frame in frames)
    assert link.reassemble_fragments(list(reversed(frames))) == payload


def test_missing_or_corrupted_packet_is_rejected():
    frames = link.frame_fragments(bytes(range(64)), 4)
    with pytest.raises(link.LinkError, match="Missing"):
        link.reassemble_fragments(frames[:-1])
    broken = bytearray(frames[2]); broken[4] ^= 1
    with pytest.raises(link.LinkError, match="CRC"):
        link.reassemble_fragments(frames[:2] + [bytes(broken)] + frames[3:])
    with pytest.raises(link.LinkError, match="duplicate"):
        link.reassemble_fragments(frames + [frames[0]])


def test_parallel_link_simulation_is_conditional_and_accounted():
    payload = bytes(range(64))
    single = link.simulate_link(payload, channels=1, rate_bps=100)
    parallel = link.simulate_link(payload, channels=4, rate_bps=100)
    assert single['ok'] and parallel['ok']
    assert single['channels_used'] == 1 and parallel['channels_used'] == 4
    assert single['wire_bytes'] == 64 + 5 and parallel['wire_bytes'] == 64 + 4 * 5
    assert parallel['parallel_seconds'] < single['parallel_seconds']
    assert parallel['estimated_speedup'] < 4  # overhead, propagation and setup
    assert not link.simulate_link(payload, channels=4, corrupt=True)['ok']


def test_preview_api_accepts_bundled_real_sonar_and_upload():
    samples = client.get('/api/samples').json()
    real = next(s for s in samples if s['group'] == 'real')
    r = client.post('/api/preview', data={'sample': real['name']})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data['payload_bits'] == 64 and data['payload_bytes'] == 8
    assert len(bytes.fromhex(data['payload_hex'])) == 8
    assert data['ai_required'] is False and data['class_prediction'] is None
    assert len(data['preview_grid']) == 8 and len(data['preview_grid'][0]) == 8
    assert data['images']['reconstructed'].startswith('data:image/png;base64,')
    assert data['raw_bytes'] == data['width'] * data['height']
    rng = np.random.default_rng(110)
    img = rng.integers(0, 256, (80, 50), dtype=np.uint8)
    ok, buf = cv2.imencode('.png', img)
    assert ok
    r2 = client.post('/api/preview', files={'file': ('random.png',buf.tobytes(),'image/png')})
    assert r2.status_code == 200 and r2.json()['payload_bytes'] == 8


def test_preview_api_bad_requests():
    assert client.post('/api/preview').status_code == 400
    assert client.post('/api/preview', data={'sample': '../file'}).status_code == 404
    assert client.post('/api/preview', files={'file':('bad.jpg',b'bad','image/jpeg')}).status_code == 400


def test_link_api_end_to_end_and_corruption_detection():
    data = client.post('/api/preview', data={'sample':client.get('/api/samples').json()[0]['name']}).json()
    body = {'payload_hex':data['payload_hex'], 'channels':4, 'rate_bps':100, 'raw_bytes':data['raw_bytes']}
    r = client.post('/api/link/simulate', json=body)
    assert r.status_code == 200
    result = r.json()
    assert result['ok'] and result['payload_hex'] == data['payload_hex']
    assert result['wire_bytes'] == 8 + 4 * 5
    body['corrupt'] = True
    bad = client.post('/api/link/simulate', json=body).json()
    assert not bad['ok'] and 'CRC' in bad['error']
    assert client.post('/api/link/simulate', json={'payload_hex':'?' }).status_code == 400
    assert client.post('/api/link/simulate', json={'payload_hex':'00','channels':7}).status_code == 422
