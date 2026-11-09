"""Public-modem reference comparisons for an honest *same-link* transfer model.

These are derived from measured 64-byte application frames and published nominal
bit rates. They are not measurements of underwater equipment or a claim about
an actual governmental image-transfer protocol.
"""
from __future__ import annotations

from dataclasses import dataclass

SETUP_SECONDS = 0.48

@dataclass(frozen=True)
class ModemReference:
    id: str
    name: str
    bps: int
    notes: str
    reference: str

REFERENCES = (
    ModemReference('benthos_mfsk_1200','Teledyne Benthos ATM-900 · MFSK',1200,
                   'Nominal MFSK data rate; Gavia AUV option. Military-relevant system, not a measured government mission.',
                   'https://www.teledynemarine.com/en-us/products/Pages/acoustic-modem-option-control-module.aspx'),
    ModemReference('benthos_mfsk_2400','Teledyne Benthos ATM-900 · high-rate MFSK',2400,
                   'Public ATM-900 product datasheet; quoted modem mode rate, not a guaranteed payload rate.',
                   'https://www.oceanografialitoral.com/wp-content/uploads/benthos_atm-900_brochure_2.pdf'),
    ModemReference('whoi_5400','WHOI Micromodem · upper mode',5400,
                   'Published maximum supported data rate; real throughput depends on configuration and conditions.',
                   'https://acomms.whoi.edu/micro-modem'),
    ModemReference('benthos_psk_15360','Teledyne Benthos ATM-900 · optional PSK',15360,
                   'Optional coprocessor / high-rate PSK mode; range and SNR may reduce achieved throughput.',
                   'https://www.teledynemarine.com/en-us/products/Pages/acoustic-modem-option-control-module.aspx'),
)


def compare(wire_bytes_first: int, wire_bytes_detail: int, wire_bytes_exact: int,
            rate_bps: int, setup_seconds: float = SETUP_SECONDS) -> dict:
    """Compare staged delivery and a *standalone* exact file on identical link rates.

    The exact file is sent separately on demand; it is not added to the staged
    transmission total. All stages include CRC/headers and 64-byte padding.
    """
    if min(wire_bytes_first, wire_bytes_detail, wire_bytes_exact) <= 0:
        raise ValueError('Wire bytes must be positive')
    if wire_bytes_detail < wire_bytes_first:
        raise ValueError('Detail bytes must include first-view bytes')
    if rate_bps <= 0 or setup_seconds < 0:
        raise ValueError('Invalid acoustic link model')
    overview = setup_seconds + wire_bytes_first * 8.0 / rate_bps
    detail = setup_seconds + wire_bytes_detail * 8.0 / rate_bps
    full = setup_seconds + wire_bytes_exact * 8.0 / rate_bps
    return {
        'link_bps': rate_bps,
        'same_64_byte_framing': True,
        'first_view_wire_bytes': wire_bytes_first,
        'detail_wire_bytes': wire_bytes_detail,
        'full_wire_bytes': wire_bytes_exact,
        'first_view_seconds': round(overview, 3),
        'detail_seconds': round(detail, 3),
        'full_file_seconds': round(full, 3),
        'first_view_advantage': round(full / overview, 2),
        'quality_equivalence': False,
        'hardware_tested': False,
        'notes': 'The first view is lossy. The conventional baseline is the full original file in the same 64-byte frames. Both times assume ideal link rate and one 0.48s setup delay. No modem or military system was operated.',
    }


def with_metadata(comp: dict) -> dict:
    r = next((r for r in REFERENCES if r.bps == comp['link_bps']), None)
    comp = dict(comp)
    comp['modem_reference'] = None if r is None else {
        'id': r.id, 'name': r.name, 'bps': r.bps,
        'notes': r.notes, 'url': r.reference
    }
    return comp
