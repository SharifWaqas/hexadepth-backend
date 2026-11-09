import random

import pytest

from hexadepth import packet as pk


def make_det(n=15, seed=0):
    rng = random.Random(seed)
    return pk.Detection(
        uuv_id=rng.randint(0, 65535),
        timestamp=rng.randint(0, 2**32 - 1),
        drone_x_m=rng.randint(0, 65535),
        drone_y_m=rng.randint(0, 65535),
        drone_depth_m=round(rng.uniform(0, 6000), 1),
        obj_dx_m=round(rng.uniform(-3000, 3000), 1),
        obj_dy_m=round(rng.uniform(-3000, 3000), 1),
        extent_m=round(rng.uniform(0, 600), 2),
        class_id=rng.randint(0, 255),
        confidence=rng.randint(0, 100),
        vertices=[(rng.randint(0, 255), rng.randint(0, 255)) for _ in range(n)],
    )


def test_packet_is_exactly_64_bytes():
    assert pk.PACKET_SIZE == 64
    assert len(pk.pack(make_det())) == 64


@pytest.mark.parametrize("seed", range(200))
def test_round_trip_is_lossless(seed):
    n = random.Random(seed).randint(3, pk.MAX_VERTICES)
    det = make_det(n, seed)
    out = pk.unpack(pk.from_hex(pk.to_hex(pk.pack(det))))
    assert out == det


def test_every_single_bit_flip_is_detected():
    raw = pk.pack(make_det())
    for bit in range(pk.PACKET_SIZE * 8):
        with pytest.raises(pk.PacketError):
            pk.unpack(pk.flip_bit(raw, bit))


def test_rejects_wrong_length_and_bad_hex():
    with pytest.raises(pk.PacketError):
        pk.unpack(b"\x00" * 63)
    with pytest.raises(pk.PacketError):
        pk.from_hex("ZZ")


@pytest.mark.parametrize("n", [0, 2, 21])
def test_rejects_bad_vertex_counts(n):
    with pytest.raises(pk.PacketError):
        pk.pack(make_det(n))


def test_rejects_out_of_range_fields():
    det = make_det()
    det.confidence = 101
    with pytest.raises(pk.PacketError):
        pk.pack(det)
