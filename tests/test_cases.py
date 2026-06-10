from sionna_device_migration_repro.cases import get_case, iter_cases


def test_case_registry_contains_core_cases():
    names = {case.name for case in iter_cases()}

    assert "awgn" in names
    assert "wrapped-awgn-channel" in names
    assert "apply-ofdm" in names
    assert "apply-time" in names
    assert "binary-memoryless" in names
    assert "edfa" in names
    assert "flat-fading" in names
    assert "kronecker-flat-fading" in names


def test_get_case_rejects_unknown_name():
    try:
        get_case("does-not-exist")
    except KeyError as exc:
        assert "Unknown case" in str(exc)
    else:
        raise AssertionError("get_case should reject unknown case names")
