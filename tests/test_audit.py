import pytest

torch = pytest.importorskip("torch")

from sionna_device_migration_repro.audit import audit_device_tree


class DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._device_str = "cpu"
        self.register_buffer("registered", torch.ones(1))
        self.unregistered = torch.ones(1)

    @property
    def device(self):
        return self._device_str


def test_audit_detects_logical_and_tensor_mismatches():
    block = DummyBlock()

    issues = audit_device_tree(block, "cuda:0")
    paths = {issue.path for issue in issues}

    assert "root._device_str" in paths
    assert "root.registered" in paths
    assert "root.unregistered" in paths


def test_audit_has_no_false_positive_for_matching_cpu():
    block = DummyBlock()

    issues = audit_device_tree(block, "cpu")

    assert issues == []
