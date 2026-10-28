# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Private, fixed-allocation OpenAI reservations; no provider calls or retries."""

from __future__ import annotations

import argparse
from contextlib import closing
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid

UNITS_PER_USD = 1_000_000_000
MAX_UNITS = 1_000_000 * UNITS_PER_USD
REPOSITORY = Path(__file__).resolve().parents[3]


class BudgetError(ValueError):
    pass


def usd_units(value) -> int:
    if type(value) not in (str, int, float) or len(str(value)) > 128:
        raise BudgetError("amount must be a finite nonnegative USD amount")
    try:
        amount = Decimal(str(value))
        if not amount.is_finite() or amount < 0 or amount > 1_000_000:
            raise InvalidOperation
        if amount == 0:
            return 0
        parts = amount.as_tuple()
        coefficient = int("".join(str(digit) for digit in parts.digits))
        exponent = parts.exponent + 9
        if exponent >= 0:
            return coefficient * 10 ** exponent
        places = -exponent
        if places >= len(parts.digits):
            raise InvalidOperation
        units, remainder = divmod(coefficient, 10 ** places)
        if remainder:
            raise InvalidOperation
        return units
    except (InvalidOperation, ValueError, OverflowError):
        raise BudgetError("amount must be finite nonnegative USD with at most nine decimal places") from None


def _identifier(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
        raise BudgetError("invalid final-run allocation identifier")


def _private_path(path, *, existing: bool) -> Path:
    path = Path(path).absolute()
    if path.is_symlink():
        raise BudgetError("budget files must not be symlinks")
    path = path.parent.resolve(strict=True) / path.name
    if (path == REPOSITORY or REPOSITORY in path.parents
            or any((parent / ".git").exists() for parent in path.parents)):
        raise BudgetError("budget files must be outside the repository")
    if path.parent.stat().st_mode & 0o077:
        raise BudgetError("budget directory must be private to its owner")
    if existing and (not path.is_file() or path.stat().st_mode & 0o077):
        raise BudgetError("budget file must be a private regular file")
    return path


class BudgetLedger:
    @classmethod
    def initialize(cls, path, ceiling_usd, allocations: dict) -> None:
        ceiling = usd_units(ceiling_usd)
        if not isinstance(allocations, dict) or not 1 <= len(allocations) <= 128:
            raise BudgetError("provide between one and 128 fixed final-run allocations")
        rows = []
        for name, amount in allocations.items():
            _identifier(name)
            rows.append((name, usd_units(amount)))
        if sum(amount for _, amount in rows) > ceiling:
            raise BudgetError("allocations exceed the fixed provider ceiling")
        path = _private_path(path, existing=False)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        with closing(sqlite3.connect(path)) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE provider (id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL, "
                       "ceiling INTEGER NOT NULL CHECK(ceiling>=0), halted INTEGER NOT NULL DEFAULT 0, "
                       "active_session TEXT)")
            db.execute("CREATE TABLE allocations (name TEXT PRIMARY KEY, ceiling INTEGER NOT NULL CHECK(ceiling>=0))")
            db.execute("CREATE TABLE calls (id TEXT PRIMARY KEY, allocation TEXT NOT NULL REFERENCES allocations(name), "
                       "request_sha256 TEXT NOT NULL, reserved INTEGER NOT NULL CHECK(reserved>0), "
                       "actual INTEGER CHECK(actual>=0 AND actual<=reserved), state TEXT NOT NULL "
                       "CHECK(state IN ('reserved','settled','uncertain')), reason TEXT, "
                       "CHECK(state!='settled' OR actual IS NOT NULL))")
            db.execute("INSERT INTO provider(id,version,ceiling) VALUES(1,1,?)", (ceiling,))
            db.executemany("INSERT INTO allocations(name,ceiling) VALUES(?,?)", rows)
            db.commit()

    def __init__(self, path, allocation: str, expected_usd=None):
        _identifier(allocation)
        self.path = _private_path(path, existing=True)
        self.allocation = allocation
        self.session = None
        with self._transaction() as db:
            provider = db.execute("SELECT version,ceiling FROM provider WHERE id=1").fetchone()
            row = db.execute("SELECT ceiling FROM allocations WHERE name=?", (allocation,)).fetchone()
            if (provider is None or provider[0] != 1 or type(provider[1]) is not int
                    or not 0 <= provider[1] <= MAX_UNITS or row is None
                    or type(row[0]) is not int or not 0 <= row[0] <= provider[1]):
                raise BudgetError("invalid ledger policy or unknown allocation")
            if expected_usd is not None and usd_units(expected_usd) != row[0]:
                raise BudgetError("requested allocation differs from its immutable ledger ceiling")
            self.limit_units = row[0]

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=10)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def start(self) -> None:
        if self.session is not None:
            raise BudgetError("budget session already started")
        session = uuid.uuid4().hex
        with self._transaction() as db:
            halted, active = db.execute("SELECT halted,active_session FROM provider WHERE id=1").fetchone()
            unresolved = db.execute("SELECT COUNT(*) FROM calls WHERE state!='settled'").fetchone()[0]
            if halted or active is not None or unresolved:
                raise BudgetError("provider ledger is halted, already owned, or requires reconciliation")
            db.execute("UPDATE provider SET active_session=? WHERE id=1", (session,))
        self.session = session

    def reserve(self, request_sha256: str, amount: int) -> str:
        if (not isinstance(request_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", request_sha256)
                or type(amount) is not int or not 0 < amount <= MAX_UNITS):
            raise BudgetError("invalid request reservation")
        identifier = uuid.uuid4().hex
        with self._transaction() as db:
            ceiling, halted, active = db.execute(
                "SELECT ceiling,halted,active_session FROM provider WHERE id=1").fetchone()
            if halted or self.session is None or active != self.session:
                raise BudgetError("provider ledger is halted or this proxy does not own it")
            used = db.execute("SELECT COALESCE(SUM(CASE WHEN state='settled' THEN actual ELSE reserved END),0) "
                              "FROM calls").fetchone()[0]
            allocated, = db.execute("SELECT ceiling FROM allocations WHERE name=?", (self.allocation,)).fetchone()
            run_used = db.execute("SELECT COALESCE(SUM(CASE WHEN state='settled' THEN actual ELSE reserved END),0) "
                                  "FROM calls WHERE allocation=?", (self.allocation,)).fetchone()[0]
            if used + amount > ceiling or run_used + amount > allocated:
                raise BudgetError("insufficient unreserved final-run or provider allocation")
            db.execute("INSERT INTO calls(id,allocation,request_sha256,reserved,state) VALUES(?,?,?,?,'reserved')",
                       (identifier, self.allocation, request_sha256, amount))
        return identifier

    def settle(self, identifier: str, actual: int) -> None:
        if type(actual) is not int or actual < 0:
            raise BudgetError("invalid actual charge")
        with self._transaction() as db:
            row = db.execute("SELECT reserved,actual,state FROM calls WHERE id=? AND allocation=?",
                             (identifier, self.allocation)).fetchone()
            if row is None or actual > row[0] or (row[2] != "reserved" and row != (row[0], actual, "settled")):
                raise BudgetError("reservation cannot be settled with this charge or state")
            db.execute("UPDATE calls SET actual=?,state='settled' WHERE id=?", (actual, identifier))

    def halt(self, reason: str, identifier=None) -> None:
        if not isinstance(reason, str) or not re.fullmatch(r"[a-z_]{1,64}", reason):
            raise BudgetError("invalid reconciliation reason")
        with self._transaction() as db:
            db.execute("UPDATE provider SET halted=1 WHERE id=1")
            if identifier is not None:
                db.execute("UPDATE calls SET state='uncertain',reason=? WHERE id=? AND state='reserved'",
                           (reason, identifier))

    def close(self) -> None:
        if self.session is None:
            return
        with self._transaction() as db:
            active, = db.execute("SELECT active_session FROM provider WHERE id=1").fetchone()
            if active != self.session:
                raise BudgetError("budget session ownership changed")
            unresolved = db.execute("SELECT COUNT(*) FROM calls WHERE state!='settled'").fetchone()[0]
            db.execute("UPDATE provider SET active_session=NULL,halted=CASE WHEN ? THEN 1 ELSE halted END WHERE id=1",
                       (unresolved,))
        self.session = None

    def snapshot(self) -> dict:
        with self._transaction() as db:
            halted, = db.execute("SELECT halted FROM provider WHERE id=1").fetchone()
            spent, held, settled, unresolved = db.execute(
                "SELECT COALESCE(SUM(actual),0),COALESCE(SUM(CASE WHEN state!='settled' THEN reserved ELSE 0 END),0),"
                "COALESCE(SUM(state='settled'),0),COALESCE(SUM(state!='settled'),0) FROM calls WHERE allocation=?",
                (self.allocation,)).fetchone()
        return {"provider": "openai", "allocation": self.allocation, "spend_usd": spent / UNITS_PER_USD,
                "reserved_usd": held / UNITS_PER_USD, "settled_calls": settled,
                "unresolved_calls": unresolved, "halted": bool(halted)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("init", choices=["init"])
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    policy_path = _private_path(args.policy, existing=True)
    if policy_path.stat().st_size > 65536:
        parser.error("private policy exceeds its size limit")

    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise BudgetError("duplicate private policy field")
            result[key] = value
        return result

    try:
        policy = json.loads(policy_path.read_text(), object_pairs_hook=unique)
        if (not isinstance(policy, dict) or set(policy) != {"provider", "ceiling_usd", "allocations"}
                or policy["provider"] != "openai"):
            raise BudgetError("unsupported private budget policy")
        BudgetLedger.initialize(args.ledger, policy["ceiling_usd"], policy["allocations"])
    except (OSError, ValueError, sqlite3.Error):
        parser.error("private ledger initialization failed; inspect policy and existing files locally")


if __name__ == "__main__":
    main()
