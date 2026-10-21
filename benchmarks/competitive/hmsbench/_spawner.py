# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Small helper process that starts measured commands and reports each child's own peak RSS.

On Linux, execve carries the high-water mark of the pre-exec address space into the child's
`ru_maxrss`, so a child forked from the large orchestrator reports at least the orchestrator's RSS.
This process stays small, forks every measured command from itself, and returns that child's
`wait4` rusage. Protocol, one JSON object per line on stdin: {argv, env, cwd, stdout, stderr}.
Replies, one line each: "pid <n>" when the command is running, then
"done <exit status> <peak RSS bytes>". The helper exits when stdin closes.
"""

import json
import os
import sys


def reply(text):
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def serve():
    for line in sys.stdin:
        job = json.loads(line)
        pid = os.fork()
        if pid == 0:
            try:
                for fd, path, flags in ((0, os.devnull, os.O_RDONLY), (1, job["stdout"], os.O_WRONLY | os.O_APPEND),
                                        (2, job["stderr"], os.O_WRONLY | os.O_APPEND)):
                    source = os.open(path, flags)
                    try:
                        os.dup2(source, fd)
                    finally:
                        if source != fd:
                            os.close(source)
                if job["cwd"]:
                    os.chdir(job["cwd"])
                os.execvpe(job["argv"][0], job["argv"], job["env"])
            except OSError as error:
                os.write(2, f"{job['argv'][0]}: {error}\n".encode())
            finally:
                os._exit(127)
        reply(f"pid {pid}")
        _, status, usage = os.wait4(pid, 0)
        reply(f"done {os.waitstatus_to_exitcode(status)} {usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)}")


if __name__ == "__main__":
    serve()
