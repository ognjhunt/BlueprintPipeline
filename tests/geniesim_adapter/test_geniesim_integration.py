import os

import pytest

from tools.geniesim_adapter.local_framework import CommandType, GenieSimGRPCClient


@pytest.mark.integration
def test_geniesim_checker_status_contract() -> None:
    if os.getenv("GENIESIM_INTEGRATION_TEST") != "1":
        pytest.skip("GENIESIM_INTEGRATION_TEST not enabled")

    client = GenieSimGRPCClient()
    assert client.connect(), "Failed to connect to Genie Sim server"

    try:
        result = client.send_command(CommandType.GET_CHECKER_STATUS, {"checker": "status"})
        assert result.available, result.error
        assert result.success, result.error
        payload = result.payload or {}
        assert payload.get("msg"), "Expected checker status message"
    finally:
        client.disconnect()
