#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from consolidator.app import (
    ERROR_LOG_NAME,
    MACOS_TRASH,
    ErrorLogger,
    compute_hash,
    ensure_unique_name,
    process_files,
    prune_empty_dirs,
    validate_relationships,
)


@dataclass
class Scenario:
    name: str
    inject_error: bool
    expect_error_log_entry: bool


SCENARIOS = [
    Scenario(name="clean_run", inject_error=False, expect_error_log_entry=False),
    Scenario(name="error_run", inject_error=True, expect_error_log_entry=True),
]


def main() -> None:
    console = Console()
    results = []

    for scenario in SCENARIOS:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            src_dirs = [base / "source_a", base / "source_b", base / "source_c"]
            dest = base / "result"

            for path in src_dirs:
                path.mkdir(parents=True, exist_ok=True)
            dest.mkdir(parents=True, exist_ok=True)

            protected_path, protected_hash = populate_fixtures(
                src_dirs[0], src_dirs[1], scenario.inject_error
            )
            populate_extra_fixture(src_dirs[2])

            validate_relationships(console, src_dirs, dest)
            log_path = dest / ERROR_LOG_NAME
            with ErrorLogger(log_path) as error_logger:
                process_files(src_dirs, dest, console, error_logger)
                inject_macos_artifacts(dest)
                prune_empty_dirs(dest)
                ensure_no_trash(dest)
                verify_longest_name(dest)
                verify_timestamp_suffix_logic(dest)

            if protected_path is not None:
                protected_path.chmod(0o755)

            source_hashes = gather_hashes(src_dirs)
            if protected_hash:
                source_hashes.discard(protected_hash)
            dest_hashes = gather_hashes([dest])

            if source_hashes != dest_hashes:
                console.print(
                    "[red]Множество хэшей итоговой папки отличается от объединения источников.[/red]"
                )
                sys.exit(1)

            if has_empty_directories(dest):
                console.print(
                    "[red]После консолидации остались пустые папки в результате.[/red]"
                )
                sys.exit(1)

            verify_error_log(log_path, scenario.expect_error_log_entry)
            total_source_files = count_files(src_dirs)
            results.append(
                (
                    scenario.name,
                    len(dest_hashes),
                    total_source_files,
                    total_source_files - len(dest_hashes),
                )
            )

    table = Table(title="Smoke summary")
    table.add_column("Scenario")
    table.add_column("Unique files")
    table.add_column("Sources")
    table.add_column("Duplicates")

    for row in results:
        table.add_row(row[0], str(row[1]), str(row[2]), str(row[3]))

    console.print(table)


def populate_fixtures(src1: Path, src2: Path, inject_error: bool) -> tuple[Path | None, str | None]:
    """Создаёт тестовые файлы с дубликатами и конфликтами имён."""
    write_text(src1 / "alpha.txt", "alpha")
    write_text(src1 / "nested" / "shared.bin", "same-payload")
    write_text(src1 / "conflict" / "data.txt", "original")
    write_text(src1 / "reports" / "report copy.txt", "monthly report")

    write_text(src2 / "beta.txt", "beta")
    write_text(src2 / "nested" / "shared.bin", "same-payload")  # дубликат по содержимому
    write_text(src2 / "conflict" / "data.txt", "modified")  # конфликт имён
    write_text(src2 / "unique" / "gamma.txt", "gamma")
    write_text(src2 / "reports" / "report.txt", "monthly report")  # тот же хэш, другое имя

    if inject_error:
        # создаём папку, к которой нельзя получить доступ, чтобы спровоцировать запись в лог
        protected_dir = src2 / "protected"
        write_text(protected_dir / "secret.txt", "hidden")
        protected_hash = compute_hash(protected_dir / "secret.txt")
        os.chmod(protected_dir, 0o000)
        return protected_dir, protected_hash
    return None, None


def populate_extra_fixture(src: Path) -> None:
    write_text(src / "delta" / "notes.txt", "delta")
    write_text(src / "reports" / "annual report extended 2024.txt", "monthly report")


def inject_macos_artifacts(dest: Path) -> None:
    """Добавляет .DS_Store в несколько каталогов, чтобы убедиться, что очистка сработает."""
    for directory in [dest, *(p for p in dest.rglob("*") if p.is_dir())]:
        marker = directory / ".DS_Store"
        marker.write_text("macOS metadata test")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def gather_hashes(roots: Iterable[Path]) -> set[str]:
    hashes: set[str] = set()
    for root in roots:
        for file_path in root.rglob("*"):
            if file_path.is_file() and file_path.name not in MACOS_TRASH and file_path.name != ERROR_LOG_NAME:
                hashes.add(compute_hash(file_path))
    return hashes


def has_empty_directories(root: Path) -> bool:
    for directory in sorted({p for p in root.rglob("*") if p.is_dir()}, key=lambda p: len(p.parts)):
        entries = [p for p in directory.iterdir() if p.name not in MACOS_TRASH]
        if not entries:
            return True
    return False


def count_files(roots: Sequence[Path]) -> int:
    count = 0
    for root in roots:
        for file_path in root.rglob("*"):
            if (
                file_path.is_file()
                and file_path.name not in MACOS_TRASH
                and file_path.name != ERROR_LOG_NAME
            ):
                count += 1
    return count


def ensure_no_trash(dest: Path) -> None:
    leftovers = [p for p in dest.rglob("*") if p.name in MACOS_TRASH]
    if leftovers:
        raise AssertionError(f"После очистки остались системные файлы: {leftovers}")


def verify_timestamp_suffix_logic(dest: Path) -> None:
    target_dir = dest / "ts_conflict"
    target_dir.mkdir(parents=True, exist_ok=True)
    base_target = target_dir / "report.txt"
    base_target.write_text("original")

    stub_dir = dest.parent / "ts_sources"
    stub_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        (datetime(2024, 11, 8, 10, 0, 0, tzinfo=timezone.utc), "report.txt__241108"),
        (datetime(2024, 11, 8, 10, 0, 0, tzinfo=timezone.utc), "report.txt__241108-10"),
        (datetime(2024, 11, 8, 10, 5, 0, tzinfo=timezone.utc), "report.txt__241108-10-05"),
        (
            datetime(2024, 11, 8, 10, 5, 30, tzinfo=timezone.utc),
            "report.txt__241108-10-05-30",
        ),
        # конфликт столетий: 2050 (-> 50) и 2150 (-> 50) должны разрешиться расширением до YYYY
        (datetime(2050, 1, 1, 0, 0, 0, tzinfo=timezone.utc), "report.txt__500101"),
        (datetime(2150, 1, 1, 0, 0, 0, tzinfo=timezone.utc), "report.txt__21500101"),
    ]

    for idx, (dt_value, expected_name) in enumerate(scenarios):
        source_file = stub_dir / f"variant_{idx}.txt"
        source_file.write_text(f"variant {idx}")
        timestamp = dt_value.timestamp()
        os.utime(source_file, (timestamp, timestamp))
        candidate = ensure_unique_name(base_target, source_file)
        if candidate.name != expected_name:
            raise AssertionError(
                f"Ожидал имя {expected_name}, но получил {candidate.name} для сценария {idx}"
            )
        candidate.write_text(f"materialized {idx}")

    shutil.rmtree(target_dir, ignore_errors=True)
    shutil.rmtree(stub_dir, ignore_errors=True)

    copy_like = [p for p in dest.rglob("*") if p.is_file() and "copy" in p.name.lower()]
    if copy_like:
        raise AssertionError(f"В итоговых именах остались 'copy': {copy_like}")


def verify_longest_name(dest: Path) -> None:
    expected = dest / "reports" / "annual report extended 2024.txt"
    if not expected.exists():
        raise AssertionError(
            "Файл с самым длинным именем не был выбран в качестве основного дубликата."
        )


def verify_error_log(log_path: Path, expect_entry: bool) -> None:
    if not log_path.exists():
        raise AssertionError("consolidator_errors.log отсутствует в результате.")
    lines = log_path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("timestamp"):
        raise AssertionError("consolidator_errors.log имеет неверный формат заголовка.")
    payload = lines[1:]
    if expect_entry:
        if not payload or payload[0].startswith("Ошибок не обнаружено"):
            raise AssertionError("Ожидалась запись об ошибке в логе, но её нет.")
    else:
        if payload != ["Ошибок не обнаружено"]:
            raise AssertionError("При чистом прогоне лог должен содержать строку 'Ошибок не обнаружено'.")


if __name__ == "__main__":
    main()
