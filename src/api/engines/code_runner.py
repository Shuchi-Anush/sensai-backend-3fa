"""
CodeRunner — Sandboxed code execution via subprocess.

Supports: Python, JavaScript (Node.js), Java, C++.

Uses run_in_executor + blocking subprocess.run so it works on all
platforms including Windows (SelectorEventLoop does not support
asyncio.create_subprocess_exec).
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum

from api.utils.logging import logger


class RunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    COMPILE_ERROR = "COMPILE_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TLE = "TLE"
    MLE = "MLE"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class CodeRunResult:
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int = -1
    execution_time_ms: float = 0.0
    status: RunStatus = RunStatus.SUCCESS


# Maximum output size (256 KB)
MAX_OUTPUT_BYTES = 256 * 1024
DEFAULT_TIMEOUT_SECONDS = 5


# ---------- shared subprocess helper ----------

def _exec(cmd: list[str], stdin_data: str, timeout: float, cwd: str | None = None) -> tuple[str, str, int, float]:
    """Run a command, return (stdout, stderr, returncode, elapsed_ms)."""
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        input=stdin_data.encode("utf-8"),
        capture_output=True,
        timeout=timeout,
        cwd=cwd,
    )
    elapsed = (time.perf_counter() - start) * 1000
    stdout = proc.stdout[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace").strip()
    stderr = proc.stderr[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace").strip()
    return stdout, stderr, proc.returncode, elapsed


# ---------- language runners (all synchronous) ----------

def _run_python(source_code: str, stdin_data: str, timeout: float) -> CodeRunResult:
    result = CodeRunResult()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(source_code)
            tmp_path = f.name

        # Syntax check
        cc = subprocess.run(
            [sys.executable, "-m", "py_compile", tmp_path],
            capture_output=True, timeout=10,
        )
        if cc.returncode != 0:
            result.status = RunStatus.COMPILE_ERROR
            result.stderr = cc.stderr.decode("utf-8", errors="replace").strip()
            return result

        try:
            stdout, stderr, rc, elapsed = _exec([sys.executable, tmp_path], stdin_data, timeout)
        except subprocess.TimeoutExpired:
            result.status = RunStatus.TLE
            result.stderr = f"Time Limit Exceeded ({timeout}s)"
            return result

        result.stdout = stdout
        result.stderr = stderr or None
        result.exit_code = rc
        result.execution_time_ms = round(elapsed, 2)
        if "MemoryError" in (stderr or ""):
            result.status = RunStatus.MLE
        elif rc != 0:
            result.status = RunStatus.RUNTIME_ERROR
        else:
            result.status = RunStatus.SUCCESS
    except Exception as e:
        logger.error(f"CodeRunner[python] error: {e}")
        result.status = RunStatus.RUNTIME_ERROR
        result.stderr = str(e)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return result


def _run_javascript(source_code: str, stdin_data: str, timeout: float) -> CodeRunResult:
    result = CodeRunResult()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(source_code)
            tmp_path = f.name

        try:
            stdout, stderr, rc, elapsed = _exec(["node", tmp_path], stdin_data, timeout)
        except subprocess.TimeoutExpired:
            result.status = RunStatus.TLE
            result.stderr = f"Time Limit Exceeded ({timeout}s)"
            return result
        except FileNotFoundError:
            result.status = RunStatus.RUNTIME_ERROR
            result.stderr = "Node.js not found. Please install Node.js to run JavaScript."
            return result

        result.stdout = stdout
        result.stderr = stderr or None
        result.exit_code = rc
        result.execution_time_ms = round(elapsed, 2)
        result.status = RunStatus.RUNTIME_ERROR if rc != 0 else RunStatus.SUCCESS
    except Exception as e:
        logger.error(f"CodeRunner[javascript] error: {e}")
        result.status = RunStatus.RUNTIME_ERROR
        result.stderr = str(e)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return result


def _run_java(source_code: str, stdin_data: str, timeout: float) -> CodeRunResult:
    """Java: saves as Solution.java, compiles with javac, runs with java."""
    result = CodeRunResult()
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp()
        java_file = os.path.join(tmp_dir, "Solution.java")
        with open(java_file, "w", encoding="utf-8") as f:
            f.write(source_code)

        # Compile
        try:
            cc = subprocess.run(
                ["javac", java_file],
                capture_output=True, timeout=30, cwd=tmp_dir,
            )
        except FileNotFoundError:
            result.status = RunStatus.RUNTIME_ERROR
            result.stderr = "javac not found. Please install JDK to run Java."
            return result

        if cc.returncode != 0:
            result.status = RunStatus.COMPILE_ERROR
            result.stderr = cc.stderr.decode("utf-8", errors="replace").strip()
            return result

        # Run
        try:
            stdout, stderr, rc, elapsed = _exec(
                ["java", "-cp", tmp_dir, "Solution"], stdin_data, timeout
            )
        except subprocess.TimeoutExpired:
            result.status = RunStatus.TLE
            result.stderr = f"Time Limit Exceeded ({timeout}s)"
            return result
        except FileNotFoundError:
            result.status = RunStatus.RUNTIME_ERROR
            result.stderr = "java not found. Please install JRE/JDK to run Java."
            return result

        result.stdout = stdout
        result.stderr = stderr or None
        result.exit_code = rc
        result.execution_time_ms = round(elapsed, 2)
        result.status = RunStatus.RUNTIME_ERROR if rc != 0 else RunStatus.SUCCESS
    except Exception as e:
        logger.error(f"CodeRunner[java] error: {e}")
        result.status = RunStatus.RUNTIME_ERROR
        result.stderr = str(e)
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except OSError:
                pass
    return result


def _run_cpp(source_code: str, stdin_data: str, timeout: float) -> CodeRunResult:
    result = CodeRunResult()
    tmp_src = None
    tmp_exe = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False, encoding="utf-8") as f:
            f.write(source_code)
            tmp_src = f.name

        tmp_exe = tmp_src.replace(".cpp", (".exe" if os.name == "nt" else ".out"))

        # Compile
        try:
            cc = subprocess.run(
                ["g++", "-o", tmp_exe, tmp_src, "-std=c++17"],
                capture_output=True, timeout=30,
            )
        except FileNotFoundError:
            result.status = RunStatus.RUNTIME_ERROR
            result.stderr = "g++ not found. Please install GCC/MinGW to run C++."
            return result

        if cc.returncode != 0:
            result.status = RunStatus.COMPILE_ERROR
            result.stderr = cc.stderr.decode("utf-8", errors="replace").strip()
            return result

        # Run
        try:
            stdout, stderr, rc, elapsed = _exec([tmp_exe], stdin_data, timeout)
        except subprocess.TimeoutExpired:
            result.status = RunStatus.TLE
            result.stderr = f"Time Limit Exceeded ({timeout}s)"
            return result

        result.stdout = stdout
        result.stderr = stderr or None
        result.exit_code = rc
        result.execution_time_ms = round(elapsed, 2)
        result.status = RunStatus.RUNTIME_ERROR if rc != 0 else RunStatus.SUCCESS
    except Exception as e:
        logger.error(f"CodeRunner[cpp] error: {e}")
        result.status = RunStatus.RUNTIME_ERROR
        result.stderr = str(e)
    finally:
        for p in [tmp_src, tmp_exe]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
    return result


# ---------- dispatcher ----------

_RUNNERS = {
    "python": _run_python,
    "javascript": _run_javascript,
    "java": _run_java,
    "cpp": _run_cpp,
}


def _run_code_sync(
    source_code: str,
    language: str,
    stdin_data: str,
    timeout: float,
) -> CodeRunResult:
    runner = _RUNNERS.get(language.lower())
    if runner is None:
        r = CodeRunResult()
        r.status = RunStatus.UNSUPPORTED
        r.stderr = f"Language '{language}' is not supported. Supported: python, javascript, java, cpp."
        return r
    return runner(source_code, stdin_data, timeout)


# ---------- public async API ----------

async def run_code(
    source_code: str,
    language: str = "python",
    stdin_data: str = "",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> CodeRunResult:
    """Execute code in a thread-pool (non-blocking). Supports python, javascript, java, cpp."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _run_code_sync, source_code, language, stdin_data, timeout_seconds
    )


# Backward-compat alias used by older call sites
async def run_python_code(
    source_code: str,
    stdin_data: str = "",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> CodeRunResult:
    return await run_code(source_code, "python", stdin_data, timeout_seconds)
