# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading

import pytest

from budget_ledger import BudgetError, BudgetLedger, UNITS_PER_USD, usd_units


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def ledger(tmp_path, ceiling="2", allocations=None, name="a"):
    path = tmp_path / "private.sqlite3"
    BudgetLedger.initialize(path, ceiling, allocations or {name: ceiling})
    value = BudgetLedger(path, name)
    value.start()
    return value


@pytest.mark.parametrize("delta,admitted", [(-1, False), (0, True), (1, True)])
def test_reservation_boundary_is_exact_to_one_currency_unit(tmp_path, delta, admitted):
    amount = 123_456
    limit = str((amount + delta) / UNITS_PER_USD)
    value = ledger(tmp_path, limit)
    try:
        if admitted:
            identifier = value.reserve(digest("input"), amount)
            value.settle(identifier, amount)
            assert usd_units(value.snapshot()["spend_usd"]) == amount
        else:
            with pytest.raises(BudgetError, match="insufficient"):
                value.reserve(digest("input"), amount)
            assert value.snapshot()["settled_calls"] == 0
    finally:
        value.close()


def test_simultaneous_calls_cannot_reserve_the_same_unspent_money(tmp_path):
    value = ledger(tmp_path, "1")
    barrier = threading.Barrier(2)

    def reserve(number):
        barrier.wait()
        try:
            return value.reserve(digest(str(number)), 600_000_000)
        except BudgetError:
            return None

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(reserve, [0, 1]))
    assert sum(result is not None for result in results) == 1
    assert value.snapshot()["reserved_usd"] == .6
    value.settle(next(result for result in results if result), 1)
    value.close()


def test_restart_and_separate_allocations_preserve_cumulative_charges(tmp_path):
    value = ledger(tmp_path, "2", {"a": "1", "b": "1"})
    identifier = value.reserve(digest("first"), UNITS_PER_USD)
    value.settle(identifier, UNITS_PER_USD)
    value.settle(identifier, UNITS_PER_USD)
    value.close()
    reopened = BudgetLedger(value.path, "a", "1")
    reopened.start()
    with pytest.raises(BudgetError, match="insufficient"):
        reopened.reserve(digest("second"), 1)
    assert reopened.snapshot()["settled_calls"] == 1
    reopened.close()
    other = BudgetLedger(value.path, "b", "1")
    other.start()
    identifier = other.reserve(digest("other board"), UNITS_PER_USD)
    other.settle(identifier, UNITS_PER_USD)
    other.close()
    with pytest.raises(BudgetError, match="immutable"):
        BudgetLedger(value.path, "b", "1.01")
    with pytest.raises(FileExistsError):
        BudgetLedger.initialize(value.path, "3", {"a": "3"})


def test_identical_prompts_are_distinct_calls_but_settlement_is_idempotent(tmp_path):
    value = ledger(tmp_path)
    identifier = value.reserve(digest("one physical call"), 100)
    value.settle(identifier, 99)
    second = value.reserve(digest("one physical call"), 100)
    assert second != identifier
    value.settle(second, 99)
    value.settle(second, 99)
    with pytest.raises(BudgetError, match="cannot be settled"):
        value.settle(identifier, 98)
    assert value.snapshot()["spend_usd"] == 198 / UNITS_PER_USD
    value.close()


def test_crash_or_unresolved_request_cannot_reset_the_ledger(tmp_path):
    value = ledger(tmp_path)
    value.reserve(digest("unknown charge"), 500)
    restarted = BudgetLedger(value.path, "a")
    with pytest.raises(BudgetError, match="reconciliation"):
        restarted.start()
    value.halt("transport_uncertain")
    value.close()
    with pytest.raises(BudgetError, match="reconciliation"):
        restarted.start()
    assert restarted.snapshot()["unresolved_calls"] == 1
    assert restarted.snapshot()["reserved_usd"] == 500 / UNITS_PER_USD


@pytest.mark.parametrize("amount", [True, -1, float("nan"), float("inf"), "0.0000000001", "1e100000",
                                    "0.99999999999999999999999999999999", "1e-100000"])
def test_invalid_policy_amounts_are_rejected_without_creating_a_ledger(tmp_path, amount):
    path = tmp_path / "invalid.sqlite3"
    with pytest.raises(BudgetError):
        BudgetLedger.initialize(path, amount, {"a": "0"})
    assert not path.exists()


def test_allocations_cannot_exceed_provider_ceiling_or_be_added_on_reopen(tmp_path):
    path = tmp_path / "invalid.sqlite3"
    with pytest.raises(BudgetError, match="exceed"):
        BudgetLedger.initialize(path, "1", {"a": "1", "b": "0.000000001"})
    assert not path.exists()
    value = ledger(tmp_path, "1")
    with pytest.raises(BudgetError, match="unknown allocation"):
        BudgetLedger(value.path, "new-board")
    value.close()


def test_symlinks_and_public_budget_files_are_rejected(tmp_path):
    value = ledger(tmp_path, "1")
    value.close()
    link = tmp_path / "link.sqlite3"
    link.symlink_to(value.path)
    with pytest.raises(BudgetError, match="symlink"):
        BudgetLedger(link, "a")
    value.path.chmod(0o644)
    with pytest.raises(BudgetError, match="private regular"):
        BudgetLedger(value.path, "a")


def test_private_ledger_cannot_be_created_in_another_git_worktree(tmp_path):
    other = tmp_path / "other-checkout"
    other.mkdir(mode=0o700)
    (other / ".git").write_text("gitdir: synthetic")
    with pytest.raises(BudgetError, match="outside the repository"):
        BudgetLedger.initialize(other / "private.sqlite3", "1", {"final": "1"})
    assert not (other / "private.sqlite3").exists()
