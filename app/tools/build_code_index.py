from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple


DEFAULT_INCLUDE_EXTENSIONS: Set[str] = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".html",
    ".css",
    ".scss",
    ".svg",
}

DEFAULT_EXCLUDED_DIR_NAMES: Set[str] = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
    ".turbo",
    ".next",
    ".cache",
    "coverage",
    "__pycache__",
    ".venv",
    "venv",
    "env",
}


SYMBOL_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("python_function", re.compile(r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(") ),
    ("python_class", re.compile(r"^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]") ),
    ("ts_function", re.compile(r"^\s*export\s+function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(") ),
    ("ts_const_fn", re.compile(r"^\s*export\s+const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\(.*=>") ),
    ("ts_interface", re.compile(r"^\s*export\s+interface\s+([A-Za-z_][A-Za-z0-9_]*)\s*{") ),
    ("ts_type", re.compile(r"^\s*export\s+type\s+([A-Za-z_][A-Za-z0-9_]*)\s*=") ),
    ("js_function", re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(") ),
    ("react_component", re.compile(r"^\s*(export\s+)?(const|function|class)\s+([A-Z][A-Za-z0-9_]*)") ),
]


@dataclasses.dataclass
class FileIndexEntry:
    path: str
    size_bytes: int
    modified_time_epoch: float
    sha256: str
    extension: str
    language: Optional[str]
    line_count: int
    approx_word_count: int
    head: str
    tail: str
    symbols: List[Dict[str, str]]


def detect_language_from_extension(extension: str) -> Optional[str]:
    mapping = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescriptreact",
        ".js": "javascript",
        ".jsx": "javascriptreact",
        ".json": "json",
        ".md": "markdown",
        ".toml": "toml",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".svg": "svg",
    }
    return mapping.get(extension.lower())


def normalize_path(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except Exception:
        rel = path
    return rel.as_posix()


def iter_files(
    roots: List[Path],
    include_extensions: Set[str],
    excluded_dir_names: Set[str],
) -> Iterator[Path]:
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in excluded_dir_names]
            for filename in filenames:
                ext = Path(filename).suffix.lower()
                if include_extensions and ext not in include_extensions:
                    continue
                yield Path(dirpath) / filename


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_read_text(path: Path, max_bytes: int = 2_000_000) -> str:
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes)
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def extract_symbols(lines: List[str]) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for line in lines:
        for kind, pattern in SYMBOL_PATTERNS:
            m = pattern.search(line)
            if m:
                # Choose last group that likely contains the name
                name = m.groups()[-1]
                results.append({"kind": kind, "name": name})
    # Deduplicate while preserving order
    seen: Set[Tuple[str, str]] = set()
    deduped: List[Dict[str, str]] = []
    for s in results:
        key = (s["kind"], s["name"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return deduped


def build_entry(path: Path, repo_root: Path) -> Optional[FileIndexEntry]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None

    text = safe_read_text(path)
    lines = text.splitlines()
    line_count = len(lines)
    approx_word_count = sum(len(l.split()) for l in lines)

    head_lines = lines[:50]
    tail_lines = lines[-25:] if line_count > 75 else []

    entry = FileIndexEntry(
        path=normalize_path(path, repo_root),
        size_bytes=stat.st_size,
        modified_time_epoch=stat.st_mtime,
        sha256=compute_sha256(path),
        extension=path.suffix.lower(),
        language=detect_language_from_extension(path.suffix),
        line_count=line_count,
        approx_word_count=approx_word_count,
        head="\n".join(head_lines),
        tail="\n".join(tail_lines),
        symbols=extract_symbols(lines[:500]),
    )
    return entry


def build_index(
    repo_root: Path,
    targets: List[str],
    include_extensions: Set[str],
    excluded_dir_names: Set[str],
) -> Dict[str, object]:
    roots = [repo_root / t for t in targets]
    files: List[FileIndexEntry] = []
    started_at = time.time()
    for p in iter_files(roots, include_extensions, excluded_dir_names):
        entry = build_entry(p, repo_root)
        if entry is not None:
            files.append(entry)

    index = {
        "indexed_at_epoch": time.time(),
        "repo_root": repo_root.as_posix(),
        "targets": targets,
        "file_count": len(files),
        "total_bytes": sum(f.size_bytes for f in files),
        "files": [dataclasses.asdict(f) for f in files],
        "duration_seconds": time.time() - started_at,
        "include_extensions": sorted(include_extensions),
        "excluded_dir_names": sorted(excluded_dir_names),
    }
    return index


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lightweight code index for the repository")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Repository root directory. Defaults to the project root inferred from this script's location.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="*",
        default=["backend", "frontend"],
        help="Relative directories to index",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="code_index.json",
        help="Output JSON file path (relative to --root unless absolute)",
    )
    parser.add_argument(
        "--ext",
        type=str,
        nargs="*",
        default=sorted(DEFAULT_INCLUDE_EXTENSIONS),
        help="File extensions to include (e.g. .py .ts .tsx)",
    )
    parser.add_argument(
        "--exclude-dir",
        type=str,
        nargs="*",
        default=sorted(DEFAULT_EXCLUDED_DIR_NAMES),
        help="Directory names to exclude",
    )
    return parser.parse_args(argv)


def infer_repo_root_from_script() -> Path:
    # This file is at backend/app/tools/build_code_index.py
    # repo_root = parents[3]
    return Path(__file__).resolve().parents[3]


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.root).resolve() if args.root else infer_repo_root_from_script()
    include_exts = {e if e.startswith(".") else f".{e}" for e in args.ext}
    excluded_dirs = set(args.exclude_dir)

    index = build_index(repo_root=repo_root, targets=args.targets, include_extensions=include_exts, excluded_dir_names=excluded_dirs)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Indexed {index['file_count']} files across {', '.join(args.targets)} → {out_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

