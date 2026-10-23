import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cli():
    spec = importlib.util.spec_from_file_location('aura_cxr_cli', ROOT / 'aura_cxr.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_records_lock_identities_and_chronology():
    detail = _cli().check_run_records()
    assert 'lock identities recomputed' in detail and 'validation' in detail
