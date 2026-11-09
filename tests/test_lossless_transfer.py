"""Lossless original-file-byte transfer, 64-byte framing, and API contract."""
import base64
import hashlib

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from hexadepth import file_transport as ft
from server import app

client = TestClient(app)


@pytest.mark.parametrize("source", [
    b"z" * 90_000,
    bytes(range(256)) * 30,
    b"hello", 
    bytes(np.random.default_rng(42).integers(0, 256, size=10_000, dtype=np.uint8)),
])
@pytest.mark.parametrize("channels", [1, 2, 4])
def test_bit_exact_roundtrip_for_any_bytes(source, channels):
    result = ft.simulate_file_transfer(source, channels=channels, rate_bps=9600)
    assert result['ok'] and result['restored'] == source
    assert result['sha256_source'] == result['sha256_received']
    assert result['packet_count'] * 64 == result['wire_bytes']
    assert result['payload_bytes_per_packet'] == 46
    assert result['packet_header_and_crc_bytes'] == 18
    assert all(n['wire_bytes'] == n['packet_count'] * 64 for n in result['fragments'])
    assert len(bytes.fromhex(result['first_packet_hex'])) == 64
    assert result['parallel_seconds'] <= result['single_link_seconds']


def test_original_png_file_bytes_not_just_reencoded_pixels():
    img = np.arange(301 * 701, dtype=np.uint32).reshape((301, 701)).astype(np.uint8)
    ok, buf = cv2.imencode('.png', img)
    assert ok
    source = buf.tobytes()
    result = ft.simulate_file_transfer(source, channels=4, rate_bps=1200)
    assert result['restored'] == source
    assert result['sha256_source'] == hashlib.sha256(source).hexdigest()


def test_frames_exactly_64_and_reassembled_out_of_order():
    envelope, _ = ft.encode_file(b"reference scan" * 100)
    frames = ft.frame_file(envelope)
    assert all(len(f) == 64 for f in frames)
    assert ft.reassemble_file(frames[::-1]) == envelope


def test_missing_duplicate_and_corruption_rejected():
    payload, _ = ft.encode_file(b"frames" * 500)
    frames = ft.frame_file(payload)
    assert len(frames) > 1
    with pytest.raises(ft.TransferError, match="Missing packets"):
        ft.reassemble_file(frames[:-1])
    with pytest.raises(ft.TransferError, match="duplicate"):
        ft.reassemble_file(frames + frames[:1])
    corrupted = frames[:]
    flipped = bytearray(corrupted[0]); flipped[20] ^= 1
    corrupted[0] = bytes(flipped)
    with pytest.raises(ft.TransferError, match="CRC"):
        ft.reassemble_file(corrupted)
    bad = ft.simulate_file_transfer(b"frames" * 500, channels=4, corrupt=True)
    assert not bad['ok'] and bad['restored'] is None
    assert bad['sha256_received'] is None


def test_receiver_checks_file_sha256_even_with_valid_packet_crc():
    envelope, _ = ft.encode_file(b"hello" * 200)
    modified = bytearray(envelope); modified[9] ^= 1
    frames = ft.frame_file(bytes(modified))
    with pytest.raises(ft.TransferError, match="SHA-256"):
        ft.decode_file(ft.reassemble_file(frames))


def test_lossless_api_with_real_sonar_and_corruption():
    real = next(s for s in client.get('/api/samples').json() if s['group'] == 'real')
    response = client.post('/api/lossless', data={'sample':real['name'], 'channels':'4', 'rate_bps':'9600'})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['ok'] is True and result['ai_required'] is False
    assert result['sha256_source'] == result['sha256_received']
    assert result['images']['reconstructed'].startswith('data:image/png;base64,')
    assert len(base64.b64decode(result['reconstructed_file_base64'])) == result['source_bytes']
    assert result['wire_bytes'] == result['packet_count'] * 64
    assert result['channels_used'] == 4
    broken = client.post('/api/lossless', data={'sample':real['name'], 'channels':'4', 'corrupt':'true'})
    assert broken.status_code == 200
    assert broken.json()['ok'] is False and broken.json()['images']['reconstructed'] is None


def test_lossless_api_with_unknown_uploaded_image_and_bad_requests():
    rng = np.random.default_rng(19)
    arr = rng.integers(0, 256, (67, 91), dtype=np.uint8)
    ok, buf = cv2.imencode('.png', arr)
    assert ok
    source = buf.tobytes()
    result = client.post('/api/lossless', files={'file':('unknown.png', source, 'image/png')}).json()
    assert result['ok'] and result['sha256_received'] == hashlib.sha256(source).hexdigest()
    assert base64.b64decode(result['reconstructed_file_base64']) == source
    assert client.post('/api/lossless').status_code == 400
    assert client.post('/api/lossless', data={'sample':'../../secret'}).status_code == 404
    assert client.post('/api/lossless', files={'file':('bad.png', b'bad', 'image/png')}).status_code == 400
    assert client.post('/api/lossless', data={'sample': 'sample'}, files={'file': ('big.png', b'x' * (25*1024*1024 + 1))}).status_code == 413
    assert client.post('/api/lossless', files={'file':('unknown.png',source,'image/png')}, data={'channels':'3'}).status_code == 400
