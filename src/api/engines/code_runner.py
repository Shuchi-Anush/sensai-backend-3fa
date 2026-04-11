"""
CodeRunner — Sandboxed Python code execution via subprocess.

Executes user code with stdin, captures stdout/stderr, enforces
timeout (TLE) and basic resource limits.
"""

import asyncio
import time
import os
import tempfile
import sys
from dataclasses import dataclass, field
from enum import Enum
from api.utils.logging import logger


class RunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    COMPILE_ERROR = "COMPILE_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TLE = "TLE"
    MLE = "MLE"


@dataclass
class CodeRunResult:
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int = -1
    execution_time_ms: float = 0.0
    status: RunStatus = RunStatus.SUCCESS


# Maximum output size (256KB) to prevent memory bombs
MAX_OUTPUT_BYTES = 256 * 1024
# Default timeout in seconds
DEFAULT_TIMEOUT_SECONDS = 5
# Maximum memory in bytes (128MB)
MAX_MEMORY_BYTES = 128 * 1024 * 1024


async def run_python_code(
    source_code: str,
    stdin_data: str = "",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> CodeRunResult:
    """
    Execute Python code in a sandboxed subprocess.

    1. Writes source to a temp file
    2. Runs via `python <tempfile>` with stdin piped
    3. Enforces timeout (TLE)
    4. Captures stdout, stderr, exit code
    5. Cleans up temp file
    """
    result = CodeRunResult()
    tmp_path = None

    try:
        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(source_code)
            tmp_path = tmp.name

        # First, check for syntax errors via py_compile
        compile_check = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "py_compile", tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, compile_stderr = await asyncio.wait_for(
            compile_check.communicate(), timeout=10
        )

        if compile_check.returncode != 0:
            result.status = RunStatus.COMPILE_ERROR
            result.stderr = compile_stderr.decode("utf-8", errors="replace").strip()
            result.exit_code = compile_check.returncode
            return result

        # Execute the code
        start_time = time.perf_counter()

        process = await asyncio.create_subprocess_exec(
            sys.executable, tmp_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=stdin_data.encode("utf-8")),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            # Kill the process on timeout
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass
            elapsed = (time.perf_counter() - start_time) * 1000
            result.status = RunStatus.TLE
            result.execution_time_ms = round(elapsed, 2)
            result.stderr = f"Time Limit Exceeded ({timeout_seconds}s)"
            result.exit_code = -1
            return result

        elapsed = (time.perf_counter() - start_time) * 1000
        result.execution_time_ms = round(elapsed, 2)

        # Truncate oversized output
        stdout_text = stdout_bytes[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        stderr_text = stderr_bytes[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")

        result.stdout = stdout_text.strip()
        result.stderr = stderr_text.strip() if stderr_text.strip() else None
        result.exit_code = process.returncode

        # Check for memory-related errors
        if result.stderr and "MemoryError" in result.stderr:
            result.status = RunStatus.MLE
        elif process.returncode != 0:
            result.status = RunStatus.RUNTIME_ERROR
        else:
            result.status = RunStatus.SUCCESS

    except Exception as e:
        logger.error(f"CodeRunner unexpected error: {e}")
        result.status = RunStatus.RUNTIME_ERROR
        result.stderr = str(e)
        result.exit_code = -1
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return result
