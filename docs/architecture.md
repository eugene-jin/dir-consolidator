# Consolidated Directory Builder

## Goal
Build a console-first tool that merges the contents of two source directories into a destination directory, keeping only unique files (deduplicated by hash) and ensuring no empty folders remain afterwards.

## High-Level Flow
1. **Interactive directory selection**  
   - Prompt user for two existing source directories.  
   - Prompt user for (possibly new) destination directory; ensure it is not nested inside either source and create it on demand.
2. **Inventory phase**  
   - Recursively collect file entries (path + relative path + originating root) to know the total amount of work up front.
3. **Processing phase**  
   - Hash each file with SHA-256 while streaming.  
   - Skip copying if hash already encountered; log that the file is a duplicate.  
   - When the relative path already exists in the destination but contents differ, append a short hash suffix to the filename to keep both versions.
   - Copy with metadata (`shutil.copy2`) so timestamps survive.
4. **Cleanup phase**  
   - Remove empty directories from the destination tree after all copies finish.

## UI & Feedback
- Use `prompt_toolkit` to provide autocompleting path prompts for directory selection.
- Use `rich` to show a progress bar (files processed vs total) and to stream a structured log with the name of every file as it is processed.

## Key Design Choices
- **Hash accuracy**: SHA-256 with 1 MiB chunks balances performance and collision resistance.
- **Conflict handling**: Preserve as many folder structures as possible by defaulting to the relative path; при коллизиях добавляется датовый модификатор прямо к полному имени файла (`name.ext__YYMMDD`, при конфликте столетий → `__YYYYMMDD`, затем `-HH`, `-MM`, `-SS`, далее счётчик) и всегда выбирается копия с самым длинным очищенным именем (без «copy», без учёта регистра).
- **Source intake**: Пользователь вводит пути по одному (до 100 000 директорий), а пустая строка завершает ввод; вложенные пути блокируются заранее, чтобы избежать копирования «самого в себя».
- **Hash-wise dedupe**: SHA-256 служит единственным критерием уникальности, поэтому файл копируется только один раз, даже если имена в источниках различаются.
- **Copy token cleanup**: Перед записью файла название очищается от слова `copy`, чтобы итоговые имена оставались читабельными; если после очистки возникает конфликт, включается датовый суффикс.
- **Streaming & retries**: Файлы обрабатываются потоково (без хранения полного списка), каждая операция чтения/копирования выполняется с до 5 повторов и экспоненциальной задержкой; проблемные пути логируются и не прерывают консолидацию.
- **No empty folders**: Post-copy cleanup walks directories bottom-up and deletes empty ones.
- **Clean destination**: User must pick an empty result directory; если она непуста, приложение запрашивает подтверждение и очищает содержимое, чтобы исключить «хвосты» от прошлых прогонов.
- **macOS artifacts**: Поскольку Finder создаёт `.DS_Store`/`.localized`, очистка сначала удаляет эти файлы, а уже потом решает, нужно ли удалять каталог; это гарантирует «0 пустых папок» даже при открытии результата в Finder.
- **Extensibility**: Core logic lives in `consolidator.py` with typed helpers, making it easy to plug into future automation or to add tests.
