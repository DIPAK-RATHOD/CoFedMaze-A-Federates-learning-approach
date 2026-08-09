import pytest

from evaluation import resource_monitor


@pytest.mark.skipif(resource_monitor.psutil is None, reason="psutil is not installed")
def test_resource_monitor_captures_process_memory():
    monitor = resource_monitor.ResourceMonitor()
    sample = monitor.sample()

    assert sample.memory_bytes > 0
    assert len(monitor.samples) == 1
