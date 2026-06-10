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
    assert "mapper-qam" in names
    assert "llrs2symbol-logits" in names
    assert "custom-window" in names
    assert "root-raised-cosine-filter" in names
    assert "resource-grid-mapper" in names
    assert "ls-channel-estimator" in names
    assert "lmmse-equalizer" in names
    assert "mimo-list2llr-simple" in names
    assert "mimo-k-best-detector" in names
    assert "fec-crc-encoder" in names
    assert "fec-ldpc-5g-encoder" in names
    assert "fec-polar-5g-decoder" in names


def test_case_registry_contains_phy_mapping_signal_fec_mimo_and_ofdm_categories():
    cases = tuple(iter_cases())

    assert any("phy" in case.categories for case in cases)
    assert any("mapping" in case.categories for case in cases)
    assert any("signal" in case.categories for case in cases)
    assert any("fec" in case.categories for case in cases)
    assert any("mimo" in case.categories for case in cases)
    assert any("ofdm" in case.categories for case in cases)


def test_get_case_rejects_unknown_name():
    try:
        get_case("does-not-exist")
    except KeyError as exc:
        assert "Unknown case" in str(exc)
    else:
        raise AssertionError("get_case should reject unknown case names")
