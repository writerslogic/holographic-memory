# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest

from holographic_sdk.importers import HmsSink, ImporterError
from holographic_sdk.importers.sink import _delete_owned, _get


@pytest.mark.parametrize("failure", ["committed", "server_error", "transport"])
@pytest.mark.parametrize("operation", ["write", "delete"])
@pytest.mark.parametrize("envelope", ["problem", "legacy"])
def test_mutation_failures_are_not_retried(failure, operation, envelope):
    requests = []

    def handle(request):
        requests.append(request)
        if failure == "transport":
            raise httpx.ReadError("connection lost", request=request)
        code = "mutation_committed" if failure == "committed" else "internal"
        body = {"code": code, "detail": "failure"} if envelope == "problem" else {"error": {"code": code, "message": "failure"}}
        return httpx.Response(500, json=body)

    sink = HmsSink("http://fixture.invalid", transport=httpx.MockTransport(handle), retries=2, backoff=0)
    try:
        client = sink.client(None)
        with pytest.raises(ImporterError, match="committed" if failure == "committed" else "uncertain") as error:
            if operation == "write":
                sink._call(client.add_documents, [{"id": "x", "text": "text"}], retry_safe=False)
            else:
                sink._call(_delete_owned, client, "x", "owner", retry_safe=False)
        assert len(requests) == 1
        assert error.value.code == ("mutation_committed" if failure == "committed" else "mutation_uncertain")
        assert error.value.committed == (True if failure == "committed" else None)
    finally:
        sink.close()


def test_read_failure_remains_retryable():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(503 if len(requests) == 1 else 200, json={})

    sink = HmsSink("http://fixture.invalid", transport=httpx.MockTransport(handle), retries=1, backoff=0)
    try:
        assert sink._call(_get, sink.client(None), "/read").status_code == 200
        assert len(requests) == 2
    finally:
        sink.close()
