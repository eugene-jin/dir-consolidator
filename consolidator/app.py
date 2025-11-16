from __future__ import annotations

import hashlib
import shutil
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.validation import ValidationError, Validator
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


CHUNK_SIZE = 1024 * 1024  # 1 MiB
YES_ANSWERS = {"y", "yes", "д", "да"}
NO_ANSWERS = {"n", "no", "н", "нет"}
MAX_SOURCES = 100_000
RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 0.2
ERROR_LOG_NAME = "consolidator_errors.log"
T = TypeVar("T")


@dataclass(frozen=True)
class FileEntry:
    root: Path
    path: Path

    @property
    def relative_path(self) -> Path:
        return self.path.relative_to(self.root)


@dataclass
class NameInfo:
    rel_path: Path
    display_stem: str
    rank_key: str
    rank_length: int


@dataclass
class BestRecord:
    path: Path
    info: NameInfo


class ErrorLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.count = 0
        self._logged: Set[Tuple[str, str]] = set()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.log_path.open("w", encoding="utf-8")
        self._fh.write("timestamp\taction\tpath\terror\n")

    def log(self, action: str, path: Path, exc: Exception) -> None:
        key = (action, str(path))
        if key in self._logged:
            return
        self._logged.add(key)
        self.count += 1
        timestamp = datetime.now(timezone.utc).isoformat()
        self._fh.write(f"{timestamp}\t{action}\t{path}\t{exc}\n")
        self._fh.flush()

    def close(self) -> None:
        if self.count == 0:
            self._fh.write("Ошибок не обнаружено\n")
            self._fh.flush()
        self._fh.close()

    def __enter__(self) -> "ErrorLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def run() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Directory Consolidator[/bold cyan]\n"
            "Собирает уникальные файлы из выбранных папок в одну целевую.",
            border_style="cyan",
        )
    )

    session = PromptSession()
    sources = prompt_source_directories(session, console)
    dest = prompt_directory(
        session,
        console,
        "Укажи путь к папке результата (будет создана при необходимости)",
        must_exist=False,
        allow_create=True,
        require_empty=True,
    )

    validate_relationships(console, sources, dest)

    error_log_path = dest / ERROR_LOG_NAME

    with ErrorLogger(error_log_path) as error_logger:
        unique_copied, duplicates_skipped, renamed, discovered_total = process_files(
            sources, dest, console, error_logger
        )
        removed_dirs = prune_empty_dirs(dest)
        errors_logged = error_logger.count

    if discovered_total == 0:
        if errors_logged:
            console.print(
                f"[red]Не удалось обработать ни одного файла. Подробности: {error_log_path}[/red]"
            )
        else:
            console.print("[yellow]В исходных папках нет файлов для обработки.[/yellow]")
        return

    if unique_copied == 0 and errors_logged:
        console.print(
            f"[red]Все обнаруженные файлы оказались недоступны. Подробности: {error_log_path}[/red]"
        )

    console.print(
        Panel.fit(
            "\n".join(
                [
                    "[bold green]Готово![/bold green]",
                    f"Скопировано уникальных файлов: {unique_copied}",
                    f"Пропущено дубликатов: {duplicates_skipped}",
                    f"Переименовано при конфликте: {renamed}",
                    f"Удалено пустых папок: {removed_dirs}",
                    (
                        f"Ошибок доступа: {errors_logged} (см. {error_log_path})"
                        if errors_logged
                        else "Ошибок доступа: 0"
                    ),
                ]
            ),
            border_style="green",
        )
    )


def prompt_directory(
    session: PromptSession,
    console: Console,
    message: str,
    *,
    must_exist: bool = True,
    allow_create: bool = False,
    require_empty: bool = False,
) -> Path:
    completer = PathCompleter(expanduser=True, only_directories=True)

    class DirValidator(Validator):
        def validate(self, document) -> None:  # type: ignore[override]
            text = document.text.strip()
            if not text:
                raise ValidationError(message="Путь не может быть пустым.")
            path = Path(text).expanduser()
            if must_exist and not path.exists():
                raise ValidationError(message="Папка не существует.")
            if path.exists() and not path.is_dir():
                raise ValidationError(message="Укажите папку, а не файл.")

    while True:
        try:
            raw = session.prompt(f"{message}:\n> ", completer=completer, validator=DirValidator(), validate_while_typing=False)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[red]Отмена пользователем.[/red]")
            sys.exit(1)

        chosen = Path(raw).expanduser()
        if must_exist:
            chosen = chosen.resolve()
        else:
            chosen = chosen.absolute()

        if not chosen.exists() and must_exist:
            console.print("[red]Папка не найдена, попробуй снова.[/red]")
            continue

        if not chosen.exists() and allow_create:
            chosen.mkdir(parents=True, exist_ok=True)
            chosen = chosen.resolve()

        if chosen.exists() and not chosen.is_dir():
            console.print("[red]Указан путь к файлу, нужна папка.[/red]")
            continue

        if chosen.exists():
            chosen = chosen.resolve()

        if require_empty and chosen.exists() and any(chosen.iterdir()):
            if not confirm_directory_cleanup(session, console, chosen):
                console.print(
                    "[yellow]Папка результата должна быть пустой — выбери другую или подтверди очистку.[/yellow]"
                )
                continue
            clear_directory(chosen)

        return chosen


def prompt_source_directories(session: PromptSession, console: Console) -> List[Path]:
    console.print(
        Panel(
            "Вводи пути к исходным папкам по одному. Пустая строка завершает ввод.\n"
            f"Допустимо от 1 до {MAX_SOURCES} директорий.",
            border_style="cyan",
        )
    )
    completer = PathCompleter(expanduser=True, only_directories=True)
    sources: List[Path] = []
    while True:
        prompt = f"Исходная папка #{len(sources) + 1}:\n> "
        try:
            raw = session.prompt(prompt, completer=completer, validate_while_typing=False)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[red]Отмена пользователем.[/red]")
            sys.exit(1)

        if not raw.strip():
            if sources:
                break
            console.print("[red]Нужно указать хотя бы одну папку.[/red]")
            continue

        path = Path(raw.strip()).expanduser()
        if not path.exists():
            console.print("[red]Папка не найдена, попробуй снова.[/red]")
            continue
        if not path.is_dir():
            console.print("[red]Укажи папку, а не файл.[/red]")
            continue

        resolved = path.resolve()
        if any(same_path(resolved, existing) for existing in sources):
            console.print("[red]Эта папка уже добавлена, пропускаю.[/red]")
            continue

        sources.append(resolved)
        if len(sources) >= MAX_SOURCES:
            console.print(f"[yellow]Достигнут максимум {MAX_SOURCES} источников.[/yellow]")
            break

    return sources


def same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except FileNotFoundError:
        return a.absolute() == b.absolute()


def validate_relationships(console: Console, sources: List[Path], dest: Path) -> None:
    blocked: List[Tuple[Path, Path]] = []
    if not sources:
        console.print("[red]Не выбрано ни одной исходной папки.[/red]")
        sys.exit(1)

    for src in sources:
        if is_subpath(dest, src) or same_path(dest, src):
            blocked.append((dest, src))
        if is_subpath(src, dest):
            blocked.append((src, dest))

    for idx, src in enumerate(sources):
        for other in sources[idx + 1 :]:
            if is_subpath(src, other):
                blocked.append((src, other))
            elif is_subpath(other, src):
                blocked.append((other, src))

    if blocked:
        lines = "\n".join(f"- {a} ⊂ {b}" for a, b in blocked)
        console.print(
            Panel(
                f"Нельзя выбирать вложенные папки или сохранять результат в исходную:\n{lines}",
                border_style="red",
            )
        )
        sys.exit(1)


def is_subpath(candidate: Path, parent: Path) -> bool:
    try:
        candidate_resolved = candidate.resolve()
        parent_resolved = parent.resolve()
    except FileNotFoundError:
        candidate_resolved = candidate.absolute()
        parent_resolved = parent.absolute()

    try:
        candidate_resolved.relative_to(parent_resolved)
        return True
    except ValueError:
        return False


def iter_all_files(
    sources: List[Path],
    error_logger: ErrorLogger,
    on_discover: Optional[Callable[[int], None]] = None,
) -> Iterable[FileEntry]:
    for root in sources:
        yield from walk_directory(root, error_logger, on_discover)


def walk_directory(
    root: Path,
    error_logger: ErrorLogger,
    on_discover: Optional[Callable[[int], None]] = None,
) -> Iterable[FileEntry]:
    stack: List[Path] = [root]
    while stack:
        current = stack.pop()
        entries = list_dir_with_retry(current, error_logger)
        if entries is None:
            continue
        files_to_yield: List[Path] = []
        for entry in entries:
            try:
                if entry.is_dir():
                    stack.append(entry)
                elif entry.is_file():
                    files_to_yield.append(entry)
            except Exception as exc:  # noqa: BLE001
                error_logger.log("stat", entry, exc)
        if files_to_yield and on_discover:
            on_discover(len(files_to_yield))
        for entry in files_to_yield:
            yield FileEntry(root=root, path=entry)


def list_dir_with_retry(directory: Path, error_logger: ErrorLogger) -> Optional[List[Path]]:
    def action() -> List[Path]:
        return list(directory.iterdir())

    return retry_path_action("iterdir", directory, action, error_logger)


def retry_path_action(
    action: str, path: Path, func: Callable[[], T], error_logger: ErrorLogger
) -> Optional[T]:
    delay = RETRY_BASE_DELAY
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            if attempt == RETRY_ATTEMPTS - 1:
                error_logger.log(action, path, exc)
                return None
            time.sleep(delay)
            delay *= 2


def confirm_directory_cleanup(session: PromptSession, console: Console, directory: Path) -> bool:
    return ask_yes_no(
        session,
        console,
        f"Папка {directory} не пуста. Очистить содержимое перед консолидацией?",
        default=False,
    )


def ask_yes_no(session: PromptSession, console: Console, prompt: str, *, default: bool) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        try:
            answer = session.prompt(f"{prompt}{suffix} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[red]Отмена пользователем.[/red]")
            sys.exit(1)

        if not answer:
            return default
        if answer in YES_ANSWERS:
            return True
        if answer in NO_ANSWERS:
            return False
        console.print("[yellow]Ответь 'y' или 'n'.[/yellow]")


def clear_directory(directory: Path) -> None:
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def process_files(
    sources: List[Path],
    dest: Path,
    console: Console,
    error_logger: ErrorLogger,
) -> Tuple[int, int, int, int]:
    best_map: Dict[str, BestRecord] = {}
    unique_copied = 0
    duplicates_skipped = 0
    renamed = 0

    progress = Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    task_id = progress.add_task("Обработка файлов", total=0)
    discovered_total = 0

    def register_new_files(count: int) -> None:
        nonlocal discovered_total
        if count <= 0:
            return
        discovered_total += count
        progress.update(task_id, total=discovered_total)

    with progress:
        for entry in iter_all_files(sources, error_logger, register_new_files):
            progress.console.log(f"[blue]Хэшируем:[/blue] {entry.path}")
            digest = compute_hash_with_retry(entry.path, error_logger)
            if digest is None:
                progress.console.log(
                    f"[red]Пропуск[/red] {entry.path} — недоступен после {RETRY_ATTEMPTS} попыток."
                )
                progress.advance(task_id)
                continue

            name_info = build_name_info(entry)

            if digest in best_map:
                best_record = best_map[digest]
                duplicates_skipped += 1
                if is_candidate_better(name_info, best_record.info):
                    new_target = dest / name_info.rel_path
                    new_target.parent.mkdir(parents=True, exist_ok=True)
                    if new_target.exists():
                        new_target = ensure_unique_name(new_target, entry.path)
                    name_info = replace(name_info, rel_path=new_target.relative_to(dest))
                    try:
                        new_target.parent.mkdir(parents=True, exist_ok=True)
                        old_path = best_record.path
                        shutil.move(old_path, new_target)
                        best_record.path = new_target
                        best_record.info = name_info
                        progress.console.log(
                            f"[cyan]Переименовано[/cyan] {old_path.name} → {new_target.name}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        error_logger.log("rename", best_record.path, exc)
                else:
                    progress.console.log(
                        f"[yellow]Дубликат[/yellow] {entry.path} → уже есть {best_record.path}"
                    )
                progress.advance(task_id)
                continue

            target_path = dest / name_info.rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)

            if target_path.exists():
                renamed += 1
                target_path = ensure_unique_name(target_path, entry.path)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                progress.console.log(
                    f"[magenta]Конфликт имён[/magenta], сохраняю как {target_path.name}"
                )

            name_info = replace(name_info, rel_path=target_path.relative_to(dest))

            if not copy_file_with_retry(entry.path, target_path, error_logger):
                progress.console.log(
                    f"[red]Не удалось скопировать[/red] {entry.path} → {target_path}"
                )
                progress.advance(task_id)
                continue

            best_map[digest] = BestRecord(path=target_path, info=name_info)
            unique_copied += 1
            progress.console.log(f"[green]Скопировано[/green] {entry.path} → {target_path}")
            progress.advance(task_id)

    return unique_copied, duplicates_skipped, renamed, discovered_total


def compute_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def compute_hash_with_retry(path: Path, error_logger: ErrorLogger) -> Optional[str]:
    return retry_path_action("read", path, lambda: compute_hash(path), error_logger)


def copy_file_with_retry(src: Path, dest: Path, error_logger: ErrorLogger) -> bool:
    def action() -> bool:
        shutil.copy2(src, dest)
        return True

    result = retry_path_action("copy", src, action, error_logger)
    return bool(result)


TIMESTAMP_FORMATS = ["%y%m%d", "%H", "%M", "%S"]


def ensure_unique_name(target_path: Path, source_path: Path) -> Path:
    suffix = target_path.suffix
    display_stem, _, _ = clean_stem_for_display(target_path.stem)
    stem = display_stem
    dt = datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc)
    depth = 1
    counter = 1
    use_full_year = False

    while True:
        timestamp_suffix = build_timestamp_suffix(dt, depth, use_full_year)
        candidate = target_path.with_name(f"{stem}{suffix}__{timestamp_suffix}")
        if not candidate.exists():
            return candidate
        if not use_full_year and is_century_conflict(candidate, dt):
            use_full_year = True
            continue
        if depth < len(TIMESTAMP_FORMATS):
            depth += 1
            continue

        candidate_with_counter = target_path.with_name(
            f"{stem}{suffix}__{timestamp_suffix}_{counter}"
        )
        if not use_full_year and is_century_conflict(candidate_with_counter, dt):
            use_full_year = True
            continue
        if not candidate_with_counter.exists():
            return candidate_with_counter
        counter += 1


def clean_stem_for_display(stem: str) -> Tuple[str, str, int]:
    lowered = stem.lower()
    tokens = stem.replace("_", " ").replace("-", " ").split()
    filtered = [token for token in tokens if token.lower() != "copy"] if tokens else []
    raw_clean = " ".join(filtered) if filtered else ""
    display = raw_clean if raw_clean else "file"
    rank_key = raw_clean.casefold()
    rank_length = len(raw_clean)
    return display, rank_key, rank_length


def build_name_info(entry: FileEntry) -> NameInfo:
    rel = entry.relative_path
    display_stem, rank_key, rank_length = clean_stem_for_display(rel.stem)
    rel_with_clean_name = rel.with_name(f"{display_stem}{rel.suffix}")
    return NameInfo(
        rel_path=rel_with_clean_name,
        display_stem=display_stem,
        rank_key=rank_key,
        rank_length=rank_length,
    )


def is_candidate_better(candidate: NameInfo, current: NameInfo) -> bool:
    if candidate.rank_length > current.rank_length:
        return True
    if candidate.rank_length < current.rank_length:
        return False
    if candidate.rank_key > current.rank_key:
        return True
    if candidate.rank_key < current.rank_key:
        return False
    return False


def build_timestamp_suffix(dt: datetime, depth: int, use_full_year: bool) -> str:
    limit = max(1, min(depth, len(TIMESTAMP_FORMATS)))
    parts: List[str] = []
    first_format = "%Y%m%d" if use_full_year else TIMESTAMP_FORMATS[0]
    parts.append(dt.strftime(first_format))
    for idx in range(1, limit):
        parts.append(dt.strftime(TIMESTAMP_FORMATS[idx]))
    return "-".join(parts)


MACOS_TRASH = {".DS_Store", ".localized"}


def prune_empty_dirs(dest: Path) -> int:
    removed = 0
    all_dirs = sorted({p for p in dest.rglob("*") if p.is_dir()}, key=lambda p: len(p.parts), reverse=True)
    for directory in all_dirs:
        remove_trash_files(directory)
        if any(directory.iterdir()):
            continue
        directory.rmdir()
        removed += 1
    remove_trash_files(dest)
    return removed


def remove_trash_files(directory: Path) -> None:
    for entry in list(directory.iterdir()):
        if entry.name not in MACOS_TRASH:
            continue
        if entry.is_file():
            entry.unlink(missing_ok=True)
        elif entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)


def is_century_conflict(existing_path: Path, new_dt: datetime) -> bool:
    if not existing_path.exists():
        return False
    try:
        existing_dt = datetime.fromtimestamp(existing_path.stat().st_mtime, tz=timezone.utc)
    except (FileNotFoundError, OSError):
        return False
    return existing_dt.year // 100 != new_dt.year // 100


if __name__ == "__main__":
    run()
