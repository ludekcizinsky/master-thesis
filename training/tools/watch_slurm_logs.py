#!/usr/bin/env python3
from __future__ import annotations

import argparse
import curses
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple


DEFAULT_SLURM_DIR = Path("/scratch/izar/cizinsky/thesis/slurm")

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Prompt = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]


def _rich_console() -> Optional["Console"]:
    if Console is None:
        return None
    return Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live view Slurm .out/.err logs in split columns (left=out, right=err)."
    )
    parser.add_argument("--user", default=os.getenv("USER", ""), help="User for squeue query.")
    parser.add_argument(
        "--jobid",
        default=None,
        help=(
            "Job ID to monitor (e.g. 2788165_2). If omitted, script lists running jobs "
            "from squeue and prompts for selection."
        ),
    )
    parser.add_argument(
        "--slurm-dir",
        type=Path,
        default=DEFAULT_SLURM_DIR,
        help="Directory containing slurm log files.",
    )
    parser.add_argument("--lines", type=int, default=60, help="Tail lines per pane.")
    parser.add_argument("--refresh", type=float, default=0.5, help="Refresh interval in seconds.")
    return parser.parse_args()


def run_squeue(user: str) -> List[Tuple[str, str, str]]:
    cmd = ["squeue", "-u", user]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to run {' '.join(cmd)}: {proc.stderr.strip()}")

    lines = [line.rstrip("\n") for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []

    jobs: List[Tuple[str, str, str]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        jobid = parts[0]
        partition = parts[1]
        name = parts[2]
        jobs.append((jobid, name, partition))
    return jobs


def choose_jobid(jobs: List[Tuple[str, str, str]]) -> str:
    console = _rich_console()
    if console is not None and Table is not None and Prompt is not None:
        table = Table(title="Running Slurm Jobs", show_lines=False)
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("JOBID", style="magenta")
        table.add_column("NAME", style="green")
        table.add_column("PARTITION", style="yellow")
        for idx, (jobid, name, partition) in enumerate(jobs, start=1):
            table.add_row(str(idx), jobid, name, partition)
        console.print(table)
        raw = Prompt.ask("Choose job by index or type JOBID").strip()
    else:
        print("Running jobs:")
        for idx, (jobid, name, partition) in enumerate(jobs, start=1):
            print(f"  {idx:2d}. {jobid:>12}  {name:<10}  partition={partition}")
        raw = input("Choose job by index or type JOBID: ").strip()

    if not raw:
        raise RuntimeError("No selection provided.")
    if raw.isdigit():
        index = int(raw)
        if not 1 <= index <= len(jobs):
            raise RuntimeError(f"Index out of range: {index}")
        return jobs[index - 1][0]
    return raw


def _extract_array_task_ids(slurm_dir: Path, base_jobid: str) -> List[str]:
    pattern = re.compile(rf"\.{re.escape(base_jobid)}_(\d+)\.out$")
    task_ids = set()
    for out_file in slurm_dir.glob(f"*.{base_jobid}_*.out"):
        match = pattern.search(out_file.name)
        if match:
            task_ids.add(f"{base_jobid}_{match.group(1)}")
    return sorted(task_ids)


def resolve_log_files(slurm_dir: Path, jobid: str) -> Tuple[Path, Path, str]:
    out_exact = sorted(slurm_dir.glob(f"*.{jobid}.out"))
    err_exact = sorted(slurm_dir.glob(f"*.{jobid}.err"))
    if out_exact and err_exact:
        return out_exact[-1], err_exact[-1], jobid

    if "_" not in jobid:
        task_ids = _extract_array_task_ids(slurm_dir, jobid)
        if task_ids:
            console = _rich_console()
            if console is not None and Table is not None and Prompt is not None:
                table = Table(title=f"Array Tasks for {jobid}")
                table.add_column("#", justify="right", style="cyan", no_wrap=True)
                table.add_column("JOBID_TASK", style="magenta")
                for idx, task_id in enumerate(task_ids, start=1):
                    table.add_row(str(idx), task_id)
                console.print(table)
                raw = Prompt.ask("Choose task by index or type full JOBID_TASK").strip()
            else:
                print(f"Found array tasks for {jobid}:")
                for idx, task_id in enumerate(task_ids, start=1):
                    print(f"  {idx:2d}. {task_id}")
                raw = input("Choose task by index or type full JOBID_TASK: ").strip()

            if raw.isdigit():
                index = int(raw)
                if not 1 <= index <= len(task_ids):
                    raise RuntimeError(f"Index out of range: {index}")
                resolved_jobid = task_ids[index - 1]
            elif raw:
                resolved_jobid = raw
            else:
                raise RuntimeError("No task selection provided.")

            out_task = sorted(slurm_dir.glob(f"*.{resolved_jobid}.out"))
            err_task = sorted(slurm_dir.glob(f"*.{resolved_jobid}.err"))
            if out_task and err_task:
                return out_task[-1], err_task[-1], resolved_jobid

    raise FileNotFoundError(
        f"Could not find matching .out/.err files for job '{jobid}' in {slurm_dir}"
    )


def tail_lines(path: Path, n_lines: int) -> List[str]:
    if not path.exists():
        return [f"[waiting for file] {path}"]
    try:
        with path.open("r", encoding="utf-8", errors="replace") as file_handle:
            return list(deque(file_handle, maxlen=max(1, n_lines)))
    except Exception as exc:  # pragma: no cover
        return [f"[error reading {path.name}] {exc}"]


def _draw_pane(
    stdscr: curses.window,
    y_start: int,
    x_start: int,
    width: int,
    height: int,
    title: str,
    lines: List[str],
) -> None:
    title_text = f" {title} "
    stdscr.addnstr(y_start, x_start, title_text, max(0, width - 1), curses.A_BOLD)

    visible_height = max(0, height - 1)
    trimmed = lines[-visible_height:]
    for idx, line in enumerate(trimmed, start=1):
        safe = line.rstrip("\n")
        stdscr.addnstr(y_start + idx, x_start, safe, max(0, width - 1))


def run_live_view(out_path: Path, err_path: Path, jobid: str, lines: int, refresh_s: float) -> None:
    def _loop(stdscr: curses.window) -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)

        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            if width < 40 or height < 8:
                stdscr.addstr(0, 0, "Terminal too small. Resize to continue.")
                stdscr.refresh()
                time.sleep(refresh_s)
                continue

            separator_x = width // 2
            left_width = separator_x - 1
            right_width = width - separator_x - 1

            header = (
                f"JOB {jobid} | q=quit | refresh={refresh_s:.1f}s | "
                f"left={out_path.name} right={err_path.name}"
            )
            stdscr.addnstr(0, 0, header, width - 1, curses.A_REVERSE)

            for row in range(1, height):
                stdscr.addch(row, separator_x, ord("|"))

            pane_height = height - 1
            visible_lines = min(max(1, lines), max(1, pane_height - 1))
            out_lines = tail_lines(out_path, visible_lines)
            err_lines = tail_lines(err_path, visible_lines)

            _draw_pane(stdscr, 1, 0, left_width, pane_height, "STDOUT", out_lines)
            _draw_pane(stdscr, 1, separator_x + 1, right_width, pane_height, "STDERR", err_lines)

            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                break
            time.sleep(refresh_s)

    curses.wrapper(_loop)


def main() -> None:
    args = parse_args()

    if args.jobid is None:
        if not args.user:
            raise RuntimeError("No --user provided and USER env var is empty.")
        jobs = run_squeue(args.user)
        if not jobs:
            raise RuntimeError(f"No running jobs found for user '{args.user}'.")
        jobid = choose_jobid(jobs)
    else:
        jobid = args.jobid

    out_path, err_path, resolved_jobid = resolve_log_files(args.slurm_dir, jobid)

    console = _rich_console()
    if console is not None and Panel is not None:
        info = (
            f"[bold]Monitoring[/bold] {resolved_jobid}\n"
            f"[green]OUT[/green]  {out_path}\n"
            f"[red]ERR[/red]  {err_path}\n\n"
            f"[bold]Control[/bold]\n"
            f"q: quit"
        )
        console.print(Panel(info, title="Slurm Log Watcher", border_style="cyan"))
    else:
        print(f"Monitoring {resolved_jobid}")
        print(f"  OUT: {out_path}")
        print(f"  ERR: {err_path}")
        print("Control: q=quit")

    time.sleep(0.5)
    run_live_view(out_path, err_path, resolved_jobid, args.lines, args.refresh)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
