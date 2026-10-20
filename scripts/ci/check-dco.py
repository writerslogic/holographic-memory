# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Developer Certificate of Origin check for a pull request (https://developercertificate.org).

Every commit in the pull request must carry a `Signed-off-by: Name <email>` trailer whose
email is the commit's author or committer email, which is what `git commit -s` writes.
Merge commits and bot commits (dependabot, github-actions) are exempt. Commits by people
with write access to the repository are held to the same rule but reported as warnings, so
maintainers are reminded without the check blocking them; everyone else fails the check.

Environment: GH_TOKEN, REPO, PR_NUMBER, PR_AUTHOR, PR_AUTHOR_ASSOCIATION. Run `git commit -s` for new commits, or
`git rebase --signoff origin/main && git push --force-with-lease` to fix an existing branch.
"""
import json
import os
import re
import subprocess
import sys

TRAILER = re.compile(r"^Signed-off-by:\s*(?P<name>.+?)\s*<(?P<email>[^<>\s]+)>\s*$", re.MULTILINE | re.IGNORECASE)
WRITE_ROLES = {"admin", "maintain", "write"}


def gh(path: str, paginate: bool = False):
    cmd = ["gh", "api", path]
    if paginate:
        cmd += ["--paginate", "--slurp"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    data = json.loads(proc.stdout)
    if paginate:
        return [item for page in data for item in page]
    return data


def has_write_access(repo: str, login: str, cache: dict) -> bool:
    if login not in cache:
        data = gh(f"repos/{repo}/collaborators/{login}/permission")
        if data is None and login == os.environ.get("PR_AUTHOR"):
            # The workflow token may not be allowed to read collaborator permissions; the
            # event payload still says how the pull request author relates to the repository.
            cache[login] = os.environ.get("PR_AUTHOR_ASSOCIATION") in {"OWNER", "MEMBER", "COLLABORATOR"}
        else:
            cache[login] = bool(data) and data.get("permission") in WRITE_ROLES
    return cache[login]


def main() -> int:
    repo, number = os.environ["REPO"], os.environ["PR_NUMBER"]
    commits = gh(f"repos/{repo}/pulls/{number}/commits", paginate=True)
    if commits is None:
        print("could not list the pull request commits", file=sys.stderr)
        return 2
    failures, warnings, checked = [], [], 0
    access: dict = {}
    for commit in commits:
        sha = commit["sha"][:10]
        body = commit["commit"]["message"]
        author = commit.get("author") or {}
        login = author.get("login", "")
        if len(commit["parents"]) > 1 or author.get("type") == "Bot" or login.endswith("[bot]"):
            continue
        checked += 1
        emails = {
            commit["commit"]["author"]["email"].lower(),
            commit["commit"]["committer"]["email"].lower(),
        }
        signed = {m["email"].lower() for m in TRAILER.finditer(body)}
        if signed & emails:
            continue
        subject = body.splitlines()[0]
        reason = "no Signed-off-by trailer" if not signed else f"Signed-off-by email does not match {sorted(emails)}"
        line = f"{sha} {subject!r}: {reason}"
        (warnings if login and has_write_access(repo, login, access) else failures).append(line)
    print(json.dumps({"commits": len(commits), "checked": checked, "failures": len(failures), "warnings": len(warnings)}))
    for line in warnings:
        print(f"::warning::DCO (maintainer, not blocking) {line}")
    for line in failures:
        print(f"::error::DCO {line}")
    if failures:
        print(
            "\nSign off every commit with `git commit -s`, or fix this branch with\n"
            "  git rebase --signoff origin/main && git push --force-with-lease\n"
            "The sign-off certifies the Developer Certificate of Origin 1.1 "
            "(https://developercertificate.org); no CLA is required.",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
