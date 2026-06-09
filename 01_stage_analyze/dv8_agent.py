#!/usr/bin/env python3  # let macOS run this with python3 on PATH
"""Automate Depends + DV8 console workflow for a single repository."""  # short purpose line

# --- stdlib imports -----------------------------------------------------------
import argparse          # command-line argument parsing
import getpass           # secure prompt for activation code
import json              # read/write JSON artifacts
import os                # env vars, paths
import re                # simple regex checks
import shutil            # file/folder copy, archive extract
import subprocess        # run external commands
import sys               # exit codes, stdout/stderr
import tempfile          # runtime temp dir selection (DV8/JNA)
import configparser      # parse setup.cfg deterministically
import xml.etree.ElementTree as ET  # parse pom.xml deterministically
import urllib.error      # network error types
import urllib.request    # simple HTTP download
import zipfile           # zip extraction
from pathlib import Path # path objects instead of strings
from typing import Optional, Tuple, List, Dict, Any  # type hints for clarity
from urllib.parse import urlparse   # parse URLs
from typing import Optional as _Optional  # already present below, keep imports consistent
from datetime import datetime  # for temporal analysis timestamps
try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore


# --- constants/config ---------------------------------------------------------
"""LOCAL LLM CONFIGURATION
pip3 install -U openai
export OPENAI_API_KEY="sk-...your key..."
"""  # kept as note; not used in this script

CONFIG_HOME = Path.home() / ".dv8_agent"   # per-user config cache folder
DOWNLOADS_DIR = CONFIG_HOME / "downloads"  # where we keep downloaded zips
LICENSE_FILE = CONFIG_HOME / "license.json"  # cache DV8 license creds (masked in output)
DEPENDS_ARCHIVE_NAME = "depends-0.9.7-package-20221104a.zip"  # default archive filename
DEPENDS_DOWNLOAD_URL = (  # official release URL for the external depends.jar (fallback path)
    "https://github.com/multilang-depends/depends/releases/download/v0.9.7/"
    f"{DEPENDS_ARCHIVE_NAME}"
)

# NeoDepends defaults
NEODEPENDS_REPO_URL = os.environ.get("NEODEPENDS_REPO_URL", "https://github.com/FreeworkEarth/neodepends.git")
NEODEPENDS_RELEASE_API = os.environ.get(
    "NEODEPENDS_RELEASE_API",
    "https://api.github.com/repos/FreeworkEarth/neodepends/releases/latest",
)

# Deterministic "prod" scope exclusions (used when --scope prod/both is enabled).
# Keep this list small and stable to avoid surprising variability across runs.
PROD_EXCLUDE_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "build",
    "dist",
    "doc",
    "docs",
    "out",
    "test",
    "tests",
    "testing",
    "target",
    "node_modules",
}

# Some quick examples to show usage from CLI
EXAMPLE_USAGE = """
Examples:
  python3 dv8_agent.py --repo https://github.com/apache/pdfbox --ask "propagation cost"
  python3 dv8_agent.py --repo pdfbox-trunk.zip --workspace ./agent_ws --skip-arch-report
  python3 dv8_agent.py --repo ./local/repo --source-path src/main/java --ask m-score
"""

# --- helpers -----------------------------------------------------------------

def run(cmd, cwd=None, env=None, check=True, display: Optional[str] = None, hide_cmd: bool = False):
    """Run a subprocess, echo stdout/stderr, and optionally raise on failure."""
    # print a nice shell-like line so users can see what we run
    if display:
        print(f"\n$ {display}")                 # custom pretty display line
    elif hide_cmd:
        print("\n$ [command hidden]")           # conceal secrets (license args)
    else:
        print(f"\n$ {' '.join(str(x) for x in cmd)}")  # default: show full command

    # actually execute the process and capture output
    completed = subprocess.run(
        [str(x) for x in cmd],  # ensure strings
        cwd=cwd,                # optional working directory
        env=env,                # optional environment
        text=True,              # text mode, not bytes
        capture_output=True,    # capture stdout+stderr
    )
    # stream captured output back to our stdout/stderr
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    # optionally fail fast if non-zero exit
    if check and completed.returncode != 0:
        if hide_cmd:
            raise SystemExit(f"Command failed ({completed.returncode}) while running a hidden command.")
        raise SystemExit(f"Command failed ({completed.returncode}): {' '.join(str(x) for x in cmd)}")
    return completed  # return the CompletedProcess for callers that need rc/stdout

def ensure_dirs(*paths: Path) -> None:
    # make sure a list of folders exist (mkdir -p behavior)
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def guess_project_name(repo_path: Path) -> str:
    # turn folder name into a friendly project title (kebab/snake -> Title Case)
    return repo_path.resolve().name.replace('-', ' ').replace('_', ' ').title()


def parse_git_commit_datetime(date_str: str) -> datetime:
    """Parse git commit timestamps like '2026-02-23 10:58:51 -0500'."""
    try:
        return datetime.strptime((date_str or "").strip(), "%Y-%m-%d %H:%M:%S %z")
    except Exception:
        try:
            return datetime.fromisoformat((date_str or "").split()[0])
        except Exception:
            return datetime.min


def detect_language(source_root: Path, override: Optional[str] = None) -> str:
    """Detect language by file extension in the source root."""
    if override:
        return override.lower()
    counts = {}
    for ext in (".py", ".java", ".cs", ".ts"):
        counts[ext] = len(list(source_root.rglob(f"*{ext}")))
    py = counts[".py"]
    java = counts[".java"]
    cs = counts[".cs"]
    ts = counts[".ts"]
    if py == 0 and java == 0 and cs == 0 and ts == 0:
        return "unknown"
    # Mixed C# + TypeScript → route to Understand
    if cs > 0 and ts > 0 and cs + ts > py and cs + ts > java:
        return "mixed-cs-ts"
    # Pure TypeScript
    if ts > 0 and ts > py and ts > java and cs == 0:
        return "typescript"
    # Pure C#
    if cs > 0 and cs > py and cs > java and ts == 0:
        return "csharp"
    # Python vs Java (existing logic)
    if py >= java:
        return "python"
    return "java"

def auto_adjust_python_root(source_root: Path) -> Path:
    """
    Heuristic: if the provided Python root only contains a single nested project,
    switch to the directory that actually owns main.py.
    """
    if (source_root / "main.py").is_file():
        return source_root
    candidates: list[Path] = []
    for path in source_root.rglob("main.py"):
        try:
            rel = path.relative_to(source_root)
        except ValueError:
            continue
        if len(rel.parts) <= 4:
            candidates.append(path.parent)
    if len(candidates) == 1:
        return candidates[0]
    return source_root


def _has_excluded_segment(path: Path, *, root: Optional[Path] = None) -> bool:
    """Return True if any path segment matches a deterministic exclusion name."""
    try:
        rel = path.relative_to(root) if root else path
    except Exception:
        rel = path
    return any(part.lower() in PROD_EXCLUDE_DIR_NAMES for part in rel.parts)


def _symlink_or_copy_dir(src: Path, dst: Path, *, prefer_symlink: bool = True) -> None:
    """Create a directory link/copy.

    For DV8 depends parsing, prefer copying (symlinks may resolve to paths outside the inputFolder,
    resulting in empty matrices). For other tools, symlinks are OK and faster.
    """
    if prefer_symlink:
        try:
            os.symlink(src, dst, target_is_directory=True)
            return
        except Exception:
            pass
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _read_text_lossy(path: Path, *, max_bytes: int = 2_000_000) -> str:
    """Read a text file deterministically without exploding on encoding issues."""
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    if len(data) > max_bytes:
        data = data[:max_bytes]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _find_build_file(root: Path, filenames: tuple[str, ...]) -> Optional[Path]:
    """Find a build file at root or 1-level deep (deterministic, avoids expensive rglob)."""
    for name in filenames:
        cand = root / name
        if cand.is_file():
            return cand
    try:
        children = [p for p in root.iterdir() if p.is_dir()]
    except Exception:
        return None
    for child in sorted(children, key=lambda p: p.name.lower()):
        if child.name.lower() in PROD_EXCLUDE_DIR_NAMES:
            continue
        for name in filenames:
            cand = child / name
            if cand.is_file():
                return cand
    return None


def _parse_maven_modules(pom_path: Path) -> list[str]:
    """Extract <modules><module> entries from a pom.xml (namespace-agnostic)."""
    try:
        tree = ET.parse(pom_path)
        root = tree.getroot()
        modules = []
        for m in root.findall(".//{*}modules/{*}module"):
            if m.text and m.text.strip():
                modules.append(m.text.strip())
        return modules
    except Exception:
        return []


def infer_java_prod_roots(project_root: Path) -> tuple[list[Path], list[str]]:
    """Infer production Java roots from build metadata (deterministic).

    Returns (roots, notes). If empty, caller should fall back to heuristic globbing.
    """
    notes: list[str] = []
    roots: list[Path] = []

    # Maven: pom.xml modules
    pom = _find_build_file(project_root, ("pom.xml",))
    if pom:
        modules = _parse_maven_modules(pom)
        if modules:
            notes.append(f"maven: modules from {pom.relative_to(project_root) if pom.is_relative_to(project_root) else pom}")
            for mod in modules:
                mod_dir = (pom.parent / mod).resolve()
                src = mod_dir / "src" / "main" / "java"
                if src.is_dir():
                    roots.append(src)
        else:
            # Single-module Maven project
            src = pom.parent / "src" / "main" / "java"
            if src.is_dir():
                notes.append(f"maven: single-module from {pom.relative_to(project_root) if pom.is_relative_to(project_root) else pom}")
                roots.append(src)

    # Gradle: settings.gradle includes (only if Maven didn't produce anything)
    if not roots:
        settings = _find_build_file(project_root, ("settings.gradle", "settings.gradle.kts"))
        if settings:
            text = _read_text_lossy(settings)
            modules: list[str] = []
            for line in text.splitlines():
                stripped = line.split("//", 1)[0].strip()
                if "include" not in stripped:
                    continue
                if stripped.startswith("include") or stripped.startswith("includeFlat"):
                    for s in re.findall(r"['\"]([^'\"]+)['\"]", stripped):
                        s = s.strip()
                        if not s:
                            continue
                        modules.append(s)
            # Normalize to paths
            norm: list[str] = []
            for m in modules:
                m = m.strip()
                if m.startswith(":"):
                    m = m[1:]
                m = m.replace(":", "/")
                if m:
                    norm.append(m)
            norm = sorted(set(norm))
            if norm:
                notes.append(f"gradle: modules from {settings.relative_to(project_root) if settings.is_relative_to(project_root) else settings}")
                for mod in norm:
                    mod_dir = (settings.parent / mod).resolve()
                    src = mod_dir / "src" / "main" / "java"
                    if src.is_dir():
                        roots.append(src)
            # Single-module Gradle project
            if not roots:
                src = settings.parent / "src" / "main" / "java"
                if src.is_dir():
                    notes.append(f"gradle: single-module from {settings.relative_to(project_root) if settings.is_relative_to(project_root) else settings}")
                    roots.append(src)

    # De-dup while keeping deterministic order
    seen: set[str] = set()
    out: list[Path] = []
    for p in roots:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out, notes


def infer_python_prod_roots(project_root: Path) -> tuple[list[Path], list[str]]:
    """Infer production Python roots from packaging metadata (deterministic)."""
    notes: list[str] = []
    roots: list[Path] = []
    pyproject_data: dict[str, Any] = {}

    # pyproject.toml (preferred)
    pyproject = _find_build_file(project_root, ("pyproject.toml",))
    if pyproject and tomllib is not None:
        try:
            data = tomllib.loads(_read_text_lossy(pyproject))
        except Exception:
            data = {}
        if isinstance(data, dict):
            pyproject_data = data
        tool = data.get("tool") if isinstance(data, dict) else {}
        if isinstance(tool, dict):
            setuptools = tool.get("setuptools") or {}
            if isinstance(setuptools, dict):
                pkg_dir = setuptools.get("package-dir") or setuptools.get("package_dir")
                if isinstance(pkg_dir, dict):
                    v = pkg_dir.get("") or pkg_dir.get(".")
                    if isinstance(v, str) and v.strip():
                        roots.append((pyproject.parent / v.strip()).resolve())
                        notes.append("pyproject: tool.setuptools.package-dir")
                elif isinstance(pkg_dir, str) and pkg_dir.strip():
                    roots.append((pyproject.parent / pkg_dir.strip()).resolve())
                    notes.append("pyproject: tool.setuptools.package-dir")

                packages = setuptools.get("packages") or {}
                if isinstance(packages, dict):
                    find = packages.get("find") or {}
                    if isinstance(find, dict):
                        where = find.get("where")
                        if isinstance(where, str) and where.strip():
                            roots.append((pyproject.parent / where.strip()).resolve())
                            notes.append("pyproject: tool.setuptools.packages.find.where")
                        elif isinstance(where, list):
                            for w in where:
                                if isinstance(w, str) and w.strip():
                                    roots.append((pyproject.parent / w.strip()).resolve())
                                    notes.append("pyproject: tool.setuptools.packages.find.where")

            poetry = tool.get("poetry") or {}
            if isinstance(poetry, dict):
                pkgs = poetry.get("packages")
                if isinstance(pkgs, list):
                    for item in pkgs:
                        if isinstance(item, dict):
                            frm = item.get("from")
                            if isinstance(frm, str) and frm.strip():
                                roots.append((pyproject.parent / frm.strip()).resolve())
                                notes.append("pyproject: tool.poetry.packages[].from")

    # setup.cfg
    setup_cfg = _find_build_file(project_root, ("setup.cfg",))
    if setup_cfg:
        cfg = configparser.ConfigParser()
        try:
            cfg.read(setup_cfg, encoding="utf-8")
        except Exception:
            try:
                cfg.read(setup_cfg)
            except Exception:
                cfg = None
        if cfg:
            raw_pkg_dir = cfg.get("options", "package_dir", fallback="").strip()
            if raw_pkg_dir:
                for line in raw_pkg_dir.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if not k.strip() and v.strip():
                            roots.append((setup_cfg.parent / v.strip()).resolve())
                            notes.append("setup.cfg: [options] package_dir")
            raw_where = cfg.get("options.packages.find", "where", fallback="").strip()
            if raw_where:
                for line in raw_where.replace(",", "\n").splitlines():
                    v = line.strip()
                    if v:
                        roots.append((setup_cfg.parent / v).resolve())
                        notes.append("setup.cfg: [options.packages.find] where")

    # setup.py (regex only; never execute)
    setup_py = _find_build_file(project_root, ("setup.py",))
    if setup_py:
        roots_before = len(roots)
        text = _read_text_lossy(setup_py)
        m = re.search(r"package_dir\s*=\s*\{[^}]*['\"]\s*['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
        if m and m.group(1).strip():
            roots.append((setup_py.parent / m.group(1).strip()).resolve())
            notes.append("setup.py: package_dir")
        m2 = re.search(r"find_(?:namespace_)?packages\(\s*['\"]([^'\"]+)['\"]\s*\)", text)
        if m2 and m2.group(1).strip():
            roots.append((setup_py.parent / m2.group(1).strip()).resolve())
            notes.append("setup.py: find_packages(<where>)")
        m3 = re.search(r"find_(?:namespace_)?packages\([^)]*where\s*=\s*['\"]([^'\"]+)['\"]", text)
        if m3 and m3.group(1).strip():
            roots.append((setup_py.parent / m3.group(1).strip()).resolve())
            notes.append("setup.py: find_packages(where=...)")
        if len(roots) == roots_before:
            # Common pattern: packages=find_packages(include=[...]) with no 'where'/'package_dir'.
            include_m = re.search(r"include\s*=\s*\[\s*['\"]([^'\"]+)['\"]", text)
            if include_m and include_m.group(1).strip():
                pkg = include_m.group(1).strip().split(".", 1)[0]
                pkg_dir = (setup_py.parent / pkg).resolve()
                if pkg_dir.is_dir() and any(pkg_dir.rglob("*.py")):
                    roots.append(pkg_dir)
                    notes.append("setup.py: find_packages(include=[...]) -> package dir")
            if len(roots) == roots_before:
                # Fallback: use setup(name=...) mapping for the canonical import package folder.
                name_m = re.search(r"\bname\s*=\s*['\"]([^'\"]+)['\"]", text)
                if name_m and name_m.group(1).strip():
                    raw = name_m.group(1).strip()
                    for cand in (raw, raw.replace("-", "_")):
                        pkg_dir = (setup_py.parent / cand).resolve()
                        if pkg_dir.is_dir() and any(pkg_dir.rglob("*.py")):
                            roots.append(pkg_dir)
                            notes.append("setup.py: name -> package dir")
                            break

    # pyproject.toml (PEP 621 / non-setuptools backends)
    # If no explicit package roots were discovered, try a deterministic best-effort:
    # use [project].name to locate a top-level import package folder.
    if not roots and pyproject and pyproject_data:
        project = pyproject_data.get("project")
        if isinstance(project, dict):
            name = project.get("name")
            if isinstance(name, str) and name.strip():
                raw = name.strip()
                candidates = [raw, raw.replace("-", "_")]
                for cand in candidates:
                    pkg_dir = (pyproject.parent / cand).resolve()
                    if pkg_dir.is_dir() and any(pkg_dir.rglob("*.py")):
                        roots.append(pkg_dir)
                        notes.append("pyproject: project.name -> package dir")
                        break

    # Validate roots exist and contain python files
    valid: list[Path] = []
    for r in roots:
        if r.is_dir() and any(r.rglob("*.py")):
            valid.append(r)

    # De-dup deterministically
    seen: set[str] = set()
    out: list[Path] = []
    for p in valid:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    if out:
        notes = sorted(set(notes))
    return out, notes


def build_scoped_source_root(
    revision_path: Path,
    source_root: Path,
    *,
    lang: str,
    scope: str,
) -> Path:
    """Build a deterministic filtered source root for scope='prod'.

    Stage A (deterministic): infer production roots from build metadata when possible:
    - Java: Maven/Gradle module layout -> include matching 'src/main/java' roots
    - Python: pyproject.toml / setup.cfg / setup.py -> include declared package roots

    If inference yields no roots, fall back to conservative heuristics (still deterministic).
    """
    scope = (scope or "full").strip().lower()
    if scope != "prod":
        return source_root

    scoped_root = revision_path / "InputData" / "ScopedSource" / "prod"
    # Rebuild deterministically every run to avoid stale symlinks/copies.
    if scoped_root.exists():
        shutil.rmtree(scoped_root, ignore_errors=True)
    ensure_dirs(scoped_root)

    selection: dict[str, Any] = {
        "scope": "prod",
        "language": lang,
        "original_source_root": str(source_root),
        "excluded_dir_names": sorted(PROD_EXCLUDE_DIR_NAMES),
        "included_roots": [],
        "notes": [],
    }

    if lang == "java":
        include_dirs, infer_notes = infer_java_prod_roots(source_root)
        if infer_notes:
            selection["notes"].extend(infer_notes)
        if include_dirs:
            selection["notes"].append("java: prod roots inferred from build metadata")
        else:
            # Fallback: include every src/main/java we can find (no module-name filtering).
            include_dirs = sorted(source_root.glob("**/src/main/java"))
            if include_dirs:
                selection["notes"].append("java: prod roots inferred via fallback glob (**/src/main/java)")
        if not include_dirs:
            selection["notes"].append("No src/main/java directories found; using unfiltered source root.")
            (scoped_root / "scope_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
            return source_root

        for idx, cand in enumerate(include_dirs, 1):
            # Use the module dir relative path for stable, collision-resistant naming.
            module_dir = cand.parents[2]  # .../<module>/src/main/java
            try:
                module_rel = module_dir.relative_to(source_root)
                module_name = "__".join(module_rel.parts) if module_rel.parts else module_dir.name
            except Exception:
                module_name = module_dir.name
            module_name = re.sub(r"[^A-Za-z0-9._-]+", "_", module_name).strip("_") or module_dir.name
            dest = scoped_root / module_name
            if dest.exists():
                dest = scoped_root / f"{module_name}_{idx:02d}"
            _symlink_or_copy_dir(cand, dest, prefer_symlink=False)
            selection["included_roots"].append({"from": str(cand), "to": str(dest)})

    elif lang == "python":
        include_dirs, infer_notes = infer_python_prod_roots(source_root)
        if infer_notes:
            selection["notes"].extend(infer_notes)
        if include_dirs:
            selection["notes"].append("python: prod roots inferred from packaging metadata")
        else:
            # Prefer "src/" layout if it looks like real Python code.
            src_layout = source_root / "src"
            if src_layout.is_dir() and any(src_layout.rglob("*.py")):
                include_dirs = [src_layout]
                selection["notes"].append("python: prod roots inferred via fallback (src/ layout)")
            else:
                include_dirs = []

        if not include_dirs:
            # Include top-level package dirs (or code dirs) but skip obvious non-prod directories.
            try:
                for child in sorted(source_root.iterdir()):
                    if not child.is_dir():
                        continue
                    if child.name.lower() in PROD_EXCLUDE_DIR_NAMES:
                        continue
                    if any(child.rglob("*.py")):
                        include_dirs.append(child)
            except Exception:
                include_dirs = []

            if include_dirs:
                selection["notes"].append("python: prod roots inferred via fallback (top-level python dirs)")
                for idx, cand in enumerate(include_dirs, 1):
                    try:
                        rel = cand.relative_to(source_root)
                        dest_name = "__".join(rel.parts) if rel.parts else cand.name
                    except Exception:
                        dest_name = cand.name
                    dest_name = re.sub(r"[^A-Za-z0-9._-]+", "_", dest_name).strip("_") or cand.name
                    dest = scoped_root / dest_name
                    if dest.exists():
                        dest = scoped_root / f"{dest_name}_{idx:02d}"
                    _symlink_or_copy_dir(cand, dest, prefer_symlink=True)
                    selection["included_roots"].append({"from": str(cand), "to": str(dest)})
            else:
                # Fallback: we can't confidently isolate prod code; keep original root.
                selection["notes"].append("No clear Python package roots found; using unfiltered source root.")
                (scoped_root / "scope_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
                return source_root
        else:
            for idx, cand in enumerate(include_dirs, 1):
                # Ensure inferred roots stay inside the analyzed source root for reproducibility.
                try:
                    cand.relative_to(source_root)
                except Exception:
                    selection["notes"].append(f"python: skipped inferred root outside source_root: {cand}")
                    continue
                try:
                    rel = cand.relative_to(source_root)
                    dest_name = "__".join(rel.parts) if rel.parts else cand.name
                except Exception:
                    dest_name = cand.name
                dest_name = re.sub(r"[^A-Za-z0-9._-]+", "_", dest_name).strip("_") or cand.name
                dest = scoped_root / dest_name
                if dest.exists():
                    dest = scoped_root / f"{dest_name}_{idx:02d}"
                _symlink_or_copy_dir(cand, dest, prefer_symlink=True)
                selection["included_roots"].append({"from": str(cand), "to": str(dest)})
    else:
        selection["notes"].append(f"Unsupported language '{lang}' for prod scoping; using unfiltered source root.")
        (scoped_root / "scope_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        return source_root

    (scoped_root / "scope_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return scoped_root


def _parse_neodepends_version(name: str) -> Tuple[int, ...]:
    m = re.search(r"neodepends-v([0-9.]+)", name.strip())
    if not m:
        return (0,)
    return tuple(int(p) for p in m.group(1).split(".") if p.isdigit())


def _neodepends_platform_tag() -> str:
    import platform
    arch = platform.machine().lower()
    if arch in {"arm64", "aarch64"}:
        arch = "aarch64"
    elif arch in {"x86_64", "amd64"}:
        arch = "x86_64"
    plat = sys.platform.lower()
    if plat.startswith("darwin"):
        os_tag = "apple-darwin"
    elif plat.startswith("linux"):
        os_tag = "unknown-linux"
    elif plat.startswith("win"):
        os_tag = "pc-windows"
    else:
        os_tag = ""
    return f"{arch}-{os_tag}" if os_tag else arch


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def _extract_archive(archive_path: Path, out_dir: Path) -> None:
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(out_dir)
        return
    if archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix.endswith(".gz"):
        import tarfile
        with tarfile.open(archive_path) as tf:
            tf.extractall(out_dir)
        return
    raise SystemExit(f"Unsupported NeoDepends archive format: {archive_path.name}")


def _download_latest_neodepends_release(dest_root: Path) -> Optional[Path]:
    """Download the latest NeoDepends release if it is newer than what is installed.

    Always checks GitHub for the latest tag. Downloads and extracts only when
    the remote version is newer than all locally cached bundles. Falls back to
    the newest local bundle if the network check fails.
    """
    # --- Check GitHub for the latest release ---
    try:
        req = urllib.request.Request(NEODEPENDS_RELEASE_API, headers={"User-Agent": "dv8-agent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"NeoDepends release lookup failed: {e}")
        data = None

    remote_version: Tuple[int, ...] = (0,)
    chosen = None
    if data:
        assets = data.get("assets") or []
        tag_name = data.get("tag_name") or ""
        remote_version = _parse_neodepends_version(tag_name)
        platform_tag = _neodepends_platform_tag()
        for a in assets:
            aname = a.get("name") or ""
            if platform_tag and platform_tag in aname:
                chosen = a
                break
        if not chosen and assets:
            chosen = assets[0]

    # --- Find the newest locally installed bundle ---
    local_bundles: list[Path] = []
    if dest_root.exists():
        for entry in dest_root.iterdir():
            if entry.is_dir() and entry.name.strip().startswith("neodepends-v"):
                local_bundles.append(entry)
    if local_bundles:
        local_bundles.sort(key=lambda p: _parse_neodepends_version(p.name), reverse=True)
        newest_local = local_bundles[0]
        local_version = _parse_neodepends_version(newest_local.name)
    else:
        newest_local = None
        local_version = (0,)

    # --- Skip download if already up-to-date ---
    if newest_local and remote_version <= local_version:
        return newest_local

    # --- Download and extract ---
    if not chosen:
        # No network data and no local bundle
        return newest_local
    url = chosen.get("browser_download_url")
    archive_name = chosen.get("name") or "neodepends_release"
    if not url:
        return newest_local
    downloads = CONFIG_HOME / "downloads"
    archive_path = downloads / archive_name
    if not archive_path.exists():
        print(f"[neodepends] Downloading latest release ({data.get('tag_name', '')}): {url}")
        try:
            _download_file(url, archive_path)
        except Exception as e:
            print(f"[neodepends] Download failed: {e}")
            return newest_local
    # Strip double extension for .tar.gz (pathlib .stem only strips the last one)
    stem = archive_path.stem
    if stem.endswith(".tar"):
        stem = stem[:-4]
    extract_dir = dest_root / stem
    if not extract_dir.exists():
        try:
            _extract_archive(archive_path, dest_root)
        except Exception as e:
            print(f"[neodepends] Extraction failed: {e}")
            return newest_local
    if extract_dir.exists():
        print(f"[neodepends] Using release {data.get('tag_name', '')} at {extract_dir}")
        return extract_dir
    # Fallback: return newest dir after extraction
    dirs = [p for p in dest_root.iterdir() if p.is_dir()]
    if not dirs:
        return newest_local
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def _clone_neodepends_repo(dest_root: Path) -> Optional[Path]:
    repo_dir = dest_root / "neodepends_repo"
    if repo_dir.exists():
        return repo_dir
    print(f"Cloning NeoDepends: {NEODEPENDS_REPO_URL}")
    try:
        run(["git", "clone", NEODEPENDS_REPO_URL, str(repo_dir)])
        return repo_dir
    except Exception as e:
        print(f"NeoDepends clone failed: {e}")
        return None


def resolve_neodepends_root(explicit: Optional[str], source_pref: str = "auto") -> Optional[Path]:
    """Resolve NeoDepends root from explicit path, env, local bundles, or download."""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None
    env_root = os.environ.get("NEODEPENDS_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.exists():
            return p

    # Always check GitHub for a newer release binary and download it to ~/.dv8_agent/neodepends/.
    # This runs unconditionally so the newest binary is always available for resolve_neodepends_bin,
    # even when a local source repo is found below (source repo provides tools/, release provides binary).
    source_pref = (os.environ.get("NEODEPENDS_SOURCE") or source_pref or "auto").lower()
    dest_root = CONFIG_HOME / "neodepends"
    dest_root.mkdir(parents=True, exist_ok=True)
    if source_pref in {"release", "auto"}:
        _download_latest_neodepends_release(dest_root)  # side-effect: updates ~/.dv8_agent/neodepends/

    repo_root = Path(__file__).resolve().parents[2]
    base_candidates = [
        Path(__file__).resolve().parent.parent / "00_CORE" / "NEODEPENDS_DEICIDE" / "00_NEODEPENDS",
        repo_root / "TEST_AUTO" / "00_CORE" / "NEODEPENDS_DEICIDE" / "00_NEODEPENDS",
    ]
    for base in base_candidates:
        if not base.exists():
            continue
        # Prefer local source repo for the Python tools (binary comes from release bundle via resolve_neodepends_bin).
        src = base / "neodepends"
        if src.exists() and (src / "tools" / "neodepends_python_export.py").is_file():
            return src
        # Fall back to newest local release bundle in this base
        candidates = []
        for entry in base.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name.strip()
            if name.startswith("neodepends-v"):
                candidates.append(entry)
        if candidates:
            candidates.sort(key=lambda p: _parse_neodepends_version(p.name), reverse=True)
            return candidates[0]

    # Fallback: clone repo for tools/ (binary already downloaded above)
    if source_pref in {"repo", "auto"}:
        repo_dir = _clone_neodepends_repo(dest_root)
        if repo_dir and (repo_dir / "tools" / "neodepends_python_export.py").is_file():
            return repo_dir
    # Last resort: return the downloaded release dir (has binary but no tools/)
    dirs = [p for p in dest_root.iterdir() if p.is_dir() and p.name.startswith("neodepends-v")]
    if dirs:
        dirs.sort(key=lambda p: _parse_neodepends_version(p.name), reverse=True)
        return dirs[0]
    return None


def resolve_neodepends_bin(neodepends_root: Path, explicit_bin: Optional[str]) -> Path:
    """Resolve the NeoDepends executable path.

    Priority order:
    1. Explicit path or NEODEPENDS_BIN env var
    2. neodepends_root itself is a release bundle (has bin/neodepends-core)
    3. Global release bundle scan (newest version across all known locations) — BEFORE source build
    4. Source build in neodepends_root (last resort — may be old/incompatible)
    """
    if explicit_bin:
        p = Path(explicit_bin).expanduser().resolve()
        if p.exists():
            return p
    env_bin = os.environ.get("NEODEPENDS_BIN")
    if env_bin:
        p = Path(env_bin).expanduser().resolve()
        if p.exists():
            return p

    platform_tag = _neodepends_platform_tag()

    # Check if neodepends_root itself is a release bundle
    for cand in [
        neodepends_root / "neodepends",
        neodepends_root / "bin" / "neodepends-core",
        neodepends_root / "bin" / "neodepends",
    ]:
        if cand.is_file():
            return cand

    # Scan all known release bundle locations and pick the globally newest version.
    # This runs BEFORE source build so the latest downloaded release always beats
    # an old compiled binary (e.g. target/release/neodepends v0.0.13).
    release_bases = [
        CONFIG_HOME / "neodepends",  # ~/.dv8_agent/neodepends/ — downloaded releases
        Path(__file__).resolve().parents[2] / "TEST_AUTO" / "00_CORE" / "NEODEPENDS_DEICIDE" / "00_NEODEPENDS",
        Path(__file__).resolve().parent.parent / "00_CORE" / "NEODEPENDS_DEICIDE" / "00_NEODEPENDS",
    ]
    all_releases: list[Path] = []
    for base in release_bases:
        if not base.exists():
            continue
        for entry in base.iterdir():
            if entry.is_dir() and entry.name.strip().startswith("neodepends-v"):
                all_releases.append(entry)
    if all_releases:
        all_releases.sort(key=lambda p: _parse_neodepends_version(p.name), reverse=True)
        for rel_root in all_releases:
            for cand in [rel_root / "neodepends", rel_root / "bin" / "neodepends-core", rel_root / "bin" / "neodepends"]:
                if cand.is_file():
                    return cand

    # Last resort: source build (may be old / incompatible with current tools/)
    for cand in [
        neodepends_root / "target" / "release" / "neodepends",
        neodepends_root / "target" / platform_tag / "release" / "neodepends",
    ]:
        if cand.is_file():
            return cand

    # Return the most likely path for better error messages upstream.
    return neodepends_root / "bin" / "neodepends-core"


def run_neodepends_python_export(
    source_root: Path,
    output_dir: Path,
    neodepends_root: Path,
    neodepends_bin: Optional[str],
    resolver: str,
    config: Optional[str] = None,
    langs: Optional[str] = None,
    file_level: bool = True,
    logs: list | None = None,
) -> Path:
    """Run NeoDepends export to produce DV8 dependency JSON (Python/Java)."""
    script = neodepends_root / "tools" / "neodepends_python_export.py"
    if not script.is_file():
        # Some NeoDepends release bundles ship only the binary. Try to locate tools from a local
        # source checkout, or clone the repo as a last resort.
        base = Path(__file__).resolve().parent.parent / "00_CORE" / "NEODEPENDS_DEICIDE" / "00_NEODEPENDS"
        src = base / "neodepends"
        if src.is_dir() and (src / "tools" / "neodepends_python_export.py").is_file():
            script = src / "tools" / "neodepends_python_export.py"
        else:
            # network fallback (may be restricted in some environments)
            repo_dir = _clone_neodepends_repo(CONFIG_HOME / "neodepends")
            if repo_dir and (repo_dir / "tools" / "neodepends_python_export.py").is_file():
                script = repo_dir / "tools" / "neodepends_python_export.py"
    if not script.is_file():
        raise SystemExit(
            "NeoDepends export script not found. Provide --neodepends-root pointing to the NeoDepends "
            "git repo (contains tools/), or ensure a local checkout exists under TEST_AUTO/00_CORE."
        )
    bin_path = resolve_neodepends_bin(neodepends_root, neodepends_bin)
    if not bin_path.is_file():
        raise SystemExit(f"NeoDepends binary not found: {bin_path}")
    bin_path.chmod(0o755)
    output_dir.mkdir(parents=True, exist_ok=True)
    term_log = output_dir / "terminal_output.txt"
    cmd = [
        sys.executable,
        str(script),
        "--neodepends-bin",
        str(bin_path),
        "--input",
        str(source_root),
        "--output-dir",
        str(output_dir),
    ]
    if config:
        cmd += ["--config", config]
    if langs:
        cmd += ["--langs", langs]
    if file_level:
        cmd.append("--file-level-dv8")
    cmd += [
        "--resolver",
        resolver,
        "--terminal-output",
        str(term_log),
    ]
    res = run(cmd, check=False)
    if logs is not None:
        logs.append({
            "step": "neodepends_export",
            "cmd": ' '.join(str(x) for x in cmd),
            "rc": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        })
    if res.returncode != 0:
        raise SystemExit(f"NeoDepends export failed ({res.returncode})")
    dep_path = output_dir / "dependencies.full.dv8-dependency.json"
    def copy_candidate(candidate: Path) -> Path:
        try:
            shutil.copyfile(candidate, dep_path)
            return dep_path
        except OSError:
            return candidate

    def looks_like_dv8_dependency_json(candidate: Path) -> bool:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(data.get("variables"), list) and isinstance(data.get("cells"), list)

    # Prefer a file-level DSM (more comparable to Java file-based Depends),
    # even if a previous run already created dependencies.full.*.
    if file_level:
        for base in (output_dir / "details", output_dir / "data", output_dir):
            for pattern in ("dependencies.*.file.dv8-dsm-v3.json", "dependencies.*.raw_filtered_file.dv8-dsm-v3.json"):
                file_level_candidates = sorted(base.glob(pattern))
                if file_level_candidates:
                    return copy_candidate(file_level_candidates[0])
    if dep_path.is_file():
        return dep_path
    # Next: filtered DSM file that NeoDepends produces for Python.
    for base in (output_dir, output_dir / "data", output_dir / "details"):
        filtered = sorted(base.glob("dependencies.*.filtered.dv8-dsm-v3.json"))
        if filtered:
            return copy_candidate(filtered[0])
    # Broader fallback: NeoDepends may emit other DV8 JSONs.
    fallback = None
    for base in (output_dir, output_dir / "data", output_dir / "details"):
        for pattern in ("dependencies.*.dv8-dependency.json", "dependencies.*.dv8-dsm-v3.json"):
            matches = sorted(base.glob(pattern))
            if matches:
                fallback = matches[0]
                break
        if fallback:
            break
    if fallback:
        return copy_candidate(fallback)
    analysis_result = output_dir / "analysis-result.json"
    if analysis_result.is_file() and looks_like_dv8_dependency_json(analysis_result):
        return copy_candidate(analysis_result)
    raise SystemExit(f"NeoDepends export did not produce: {dep_path}")

def mask_token(token: str) -> str:
    # mask a sensitive string so we can safely print it (first/last 4 chars visible)
    token = token.strip()
    if not token:
        return ""
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}{'*' * (len(token) - 8)}{token[-4:]}"

def load_license_credentials() -> Tuple[Optional[str], Optional[str]]:
    # read cached license_key + activation_code from ~/.dv8_agent/license.json
    if not LICENSE_FILE.is_file():
        return None, None
    try:
        data = json.loads(LICENSE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None, None
    return data.get("license_key"), data.get("activation_code")

def save_license_credentials(license_key: str, activation_code: str) -> None:
    # write license creds to ~/.dv8_agent/license.json
    ensure_dirs(CONFIG_HOME)
    payload = {"license_key": license_key.strip(), "activation_code": activation_code.strip()}
    LICENSE_FILE.write_text(json.dumps(payload, indent=2))

def download_file(url: str, destination: Path) -> Path:
    # simple HTTP GET -> save to disk
    ensure_dirs(destination.parent)
    try:
        with urllib.request.urlopen(url) as response, open(destination, "wb") as fout:
            shutil.copyfileobj(response, fout)  # stream to file
    except urllib.error.URLError as exc:
        raise SystemExit(f"Failed to download {url}: {exc}")
    return destination

# --- path discovery -----------------------------------------------------------

def resolve_dv8_console(explicit: Optional[str]) -> Path:
    # find the dv8-console executable: explicit > env > common macOS locations
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("DV8_CONSOLE"):
        candidates.append(Path(os.environ["DV8_CONSOLE"]))
    if os.environ.get("DV8_HOME"):
        candidates.append(Path(os.environ["DV8_HOME"]) / "bin" / "dv8-console")
    # Common macOS install locations (both underscore and hyphen variants)
    candidates.append(Path("/Applications/DV8_Standard/bin/dv8-console"))
    candidates.append(Path("/Applications/DV8-standard/bin/dv8-console"))

    # pick the first that exists and is executable
    for binary in candidates:
        if binary.is_file() and os.access(binary, os.X_OK):
            return binary
    # nothing found -> instruct user how to set it
    raise SystemExit("dv8-console not found. Set --dv8-console or DV8_HOME/DV8_CONSOLE.")

def resolve_depends_jar(explicit: Optional[str]) -> Path:
    # locate an external depends.jar (only used when --depends-runner=jar or fallback)
    if explicit:
        jar = Path(explicit)
        if jar.is_file():
            return jar
        raise SystemExit(f"depends.jar not found at {jar}")

    # 1) environment variables
    if os.environ.get("DEPENDS_JAR"):
        jar = Path(os.environ["DEPENDS_JAR"])
        if jar.is_file():
            return jar

    if os.environ.get("DEPENDS_HOME"):
        jar = Path(os.environ["DEPENDS_HOME"]) / "depends.jar"
        if jar.is_file():
            return jar

    # 2) current working directory patterns
    for pattern in ("depends-*/depends.jar", "depends.jar"):
        for candidate in Path.cwd().glob(pattern):
            if candidate.is_file():
                return candidate.resolve()

    # 3) our own config cache
    for candidate in CONFIG_HOME.glob("depends-*/depends.jar"):
        if candidate.is_file():
            return candidate.resolve()

    # 4) download and extract if we don't have it yet
    archive_candidates = list(Path.cwd().glob("depends-*-package*.zip"))
    if archive_candidates:
        archive_path = archive_candidates[0]
    else:
        archive_path = DOWNLOADS_DIR / DEPENDS_ARCHIVE_NAME
        if not archive_path.is_file():
            print(f"Downloading Depends from {DEPENDS_DOWNLOAD_URL} ...")
            download_file(DEPENDS_DOWNLOAD_URL, archive_path)

    # extract into a stable location
    extract_root = archive_path.parent if archive_path.parent != Path.cwd() else CONFIG_HOME
    ensure_dirs(extract_root)
    print(f"Extracting {archive_path} ...")
    shutil.unpack_archive(str(archive_path), extract_dir=str(extract_root))

    # search again inside the extracted tree
    for candidate in extract_root.glob("depends-*/depends.jar"):
        if candidate.is_file():
            return candidate.resolve()

    # give up and ask user to point us to the jar
    raise SystemExit("Unable to locate depends.jar after extraction. Provide --depends-jar explicitly.")


def resolve_understand_tools(
    und_explicit: Optional[str] = None,
    upython_explicit: Optional[str] = None,
) -> tuple[Path, Path]:
    """Resolve SciTools Understand command-line tools.

    We assume licensing is already configured in the local Understand install.
    The pipeline never prompts for, stores, or manages Understand license data.
    """
    und_candidates: list[Path] = []
    upython_candidates: list[Path] = []

    if und_explicit:
        und_candidates.append(Path(und_explicit).expanduser())
    if os.environ.get("UNDERSTAND_UND"):
        und_candidates.append(Path(os.environ["UNDERSTAND_UND"]).expanduser())
    if os.environ.get("UNDERSTAND_HOME"):
        home = Path(os.environ["UNDERSTAND_HOME"]).expanduser()
        und_candidates.extend([home / "und", home / "bin" / "und"])
    und_candidates.append(Path("/Applications/Understand.app/Contents/MacOS/und"))

    und_bin = None
    for cand in und_candidates:
        cand = cand.resolve()
        if cand.is_file() and os.access(cand, os.X_OK):
            und_bin = cand
            break
    if und_bin is None:
        raise SystemExit(
            "SciTools Understand 'und' not found. Set --understand-und or UNDERSTAND_UND. "
            "On macOS the usual path is /Applications/Understand.app/Contents/MacOS/und."
        )

    if upython_explicit:
        upython_candidates.append(Path(upython_explicit).expanduser())
    if os.environ.get("UNDERSTAND_UPYTHON"):
        upython_candidates.append(Path(os.environ["UNDERSTAND_UPYTHON"]).expanduser())
    upython_candidates.append(und_bin.parent / "upython")
    upython_candidates.append(Path("/Applications/Understand.app/Contents/MacOS/upython"))

    upython_bin = None
    for cand in upython_candidates:
        cand = cand.resolve()
        if cand.is_file() and os.access(cand, os.X_OK):
            upython_bin = cand
            break
    if upython_bin is None:
        raise SystemExit(
            "SciTools Understand 'upython' not found. Set --understand-upython or UNDERSTAND_UPYTHON."
        )

    return und_bin, upython_bin


def understand_language_name(detected_language: str, override: Optional[str] = None) -> str:
    """Map our language labels to Understand's command-line language names."""
    if override:
        return override
    mapping = {
        "java": "Java",
        "python": "Python",
        "javascript": "Web",
        "typescript": "Web",
        "c": "C",
        "cpp": "C++",
        "c++": "C++",
        "csharp": "C#",
        "c#": "C#",
        "mixed-cs-ts": "C# Web",  # Understand groups JS+TS under "Web"
    }
    return mapping.get((detected_language or "").lower(), "Java")


def run_understand_export(
    source_root: Path,
    output_dir: Path,
    project_name: str,
    language: str,
    *,
    und_bin: Optional[str] = None,
    upython_bin: Optional[str] = None,
    granularity: str = "entity",
    force: bool = False,
    logs: list | None = None,
) -> Path:
    """Run Understand and export dependencies as DV8-style JSON."""
    ensure_dirs(output_dir)
    json_dep = output_dir / "dependencies.full.dv8-dependency.json"
    if json_dep.is_file() and not force:
        print("Reusing existing Understand dependency output (use --force-depends to regenerate).")
        return json_dep

    und, upython = resolve_understand_tools(und_bin, upython_bin)

    # Understand's upython crashes (SIGABRT / stack overflow) when the .und path is too long.
    # Use a short temp directory for the .und database regardless of where output_dir lives.
    import tempfile as _tempfile
    _und_tmp = _tempfile.mkdtemp(prefix="und_")
    und_db = Path(_und_tmp) / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', project_name[:20]).strip('_') or 'repo'}.und"

    # upython crashes (SIGABRT) when ANY argument path exceeds ~300 chars.
    # Symlink the source root to a short path if needed.
    import os as _os
    _src_str = str(source_root)
    if len(_src_str) > 200:
        _src_link = Path(_und_tmp) / "src"
        _os.symlink(_src_str, str(_src_link))
        _short_source_root = _src_link
    else:
        _short_source_root = source_root

    try:
        steps = [
            # Support multi-language: "C# Web" → ["-languages", "C#", "-languages", "Web"]
            ("understand:create", [und, "create", "-db", str(und_db)]
             + [arg for lang in language.split() for arg in ("-languages", lang)]),
            ("understand:add", [und, "add", str(_short_source_root), str(und_db)]),
            ("understand:analyze", [und, "analyze", str(und_db)]),
        ]
        for step, cmd in steps:
            res = run(cmd, check=False)
            if logs is not None:
                logs.append({
                    "step": step,
                    "cmd": ' '.join(str(x) for x in cmd),
                    "rc": res.returncode,
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                })
            if res.returncode != 0:
                raise SystemExit(
                    f"{step} failed ({res.returncode}). Confirm Understand is installed and licensed."
                )

        exporter = Path(__file__).resolve().parent / "understand_dependency_export.py"
        # project_name may contain spaces — sanitize so argparse receives it as one token
        safe_project_name = re.sub(r'[^A-Za-z0-9._-]+', '_', project_name).strip('_') or 'repo'
        # Write JSON to the short temp dir first, then copy to the final destination.
        tmp_json = Path(_und_tmp) / "deps.json"
        cmd = [
            upython,
            exporter,
            "--und-db", und_db,
            "--source-root", _short_source_root,
            "--output", tmp_json,
            "--project-name", safe_project_name,
            "--granularity", granularity,
        ]
        clean_env = {k: v for k, v in _os.environ.items()
                     if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONEXECUTABLE")}
        res = run(cmd, check=False, env=clean_env)
        if logs is not None:
            logs.append({
                "step": "understand:export-dependencies",
                "cmd": ' '.join(str(x) for x in cmd),
                "rc": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
            })
        if res.returncode != 0 or not tmp_json.is_file():
            raise SystemExit(f"Understand dependency export failed ({res.returncode})")
        ensure_dirs(output_dir)
        shutil.copy2(tmp_json, json_dep)
        return json_dep
    finally:
        shutil.rmtree(_und_tmp, ignore_errors=True)


def extract_zip_archive(archive_path: Path, dest_root: Path) -> Path:
    # unzip an archive and guess the extracted root folder
    ensure_dirs(dest_root)
    with zipfile.ZipFile(archive_path) as zip_file:
        top_levels = []
        for member in zip_file.namelist():
            if not member or member.endswith("/"):  # skip dirs / empty
                continue
            part = Path(member).parts[0]           # top-level folder/file
            if part and part not in top_levels:
                top_levels.append(part)
        zip_file.extractall(dest_root)             # extract everything

    # pick a sensible root to return
    candidates = [dest_root / name for name in top_levels if (dest_root / name).exists()]
    if len(candidates) == 1:
        return candidates[0].resolve()

    # fallback to <archive>.zip -> <dest>/<archive>
    fallback = dest_root / archive_path.stem
    if fallback.exists():
        return fallback.resolve()
    return dest_root.resolve()  # last resort

def obtain_zip_repo(repo_reference: str, workspace: Path) -> Path:
    # handle a .zip either from URL or local path and return the extracted folder
    if re.match(r"^https?://", repo_reference):
        archive_name = Path(urlparse(repo_reference).path).name or "archive.zip"
        destination = DOWNLOADS_DIR / archive_name
        if not destination.is_file():
            print(f"Downloading {repo_reference} ...")
            download_file(repo_reference, destination)
        archive_path = destination
    else:
        archive_path = Path(repo_reference).expanduser().resolve()
        if not archive_path.is_file():
            # try to find a near-matching file in CWD
            possible = list(Path.cwd().glob(f"{archive_path.name}*.zip"))
            if possible:
                archive_path = possible[0].resolve()
        if not archive_path.is_file():
            raise SystemExit(f"ZIP archive not found: {repo_reference}")

    # decide where to place the extracted repo
    target_dir = workspace / archive_path.stem
    if target_dir.exists():
        return target_dir.resolve()

    extracted = extract_zip_archive(archive_path, workspace)
    return extracted

# --- workflow steps -----------------------------------------------------------

def query_dv8_license_status(dv8_console: Path, env: dict[str, str]) -> Optional[dict]:
    # ask dv8-console about license status; parse JSON if returned
    result = run([dv8_console, "license:status"], env=env, check=False)
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        if result.returncode != 0:
            return None
        return None

def ensure_dv8_license(
    dv8_console: Path,
    env: dict[str, str],
    license_key: Optional[str],
    activation_code: Optional[str],
) -> None:
    # verify license and activate if needed (using cached creds or prompting)
    status = query_dv8_license_status(dv8_console, env)
    if status and status.get("status") == "valid":
        return

    stored_key, stored_code = load_license_credentials()
    if not license_key:
        license_key = stored_key
    if not activation_code:
        activation_code = stored_code

    while not license_key:
        license_key = input("DV8 license key: ").strip()
    while not activation_code:
        activation_code = getpass.getpass("DV8 activation code: ").strip()

    display = (  # mask sensitive values when echoing the command
        f"{dv8_console} license:activate {mask_token(license_key)} {mask_token(activation_code)}"
    )
    run(
        [dv8_console, "license:activate", license_key, activation_code],
        env=env,
        display=display,
        hide_cmd=True,
    )

    status = query_dv8_license_status(dv8_console, env)
    if not status or status.get("status") != "valid":
        raise SystemExit("DV8 activation failed. Please verify your license key and activation code.")

    save_license_credentials(license_key, activation_code)

def clone_or_copy(repo: str, workspace: Path, branch: Optional[str]) -> Path:
    # get a repository folder from: zip | http(s) git URL | local path
    if repo.lower().endswith(".zip"):
        return obtain_zip_repo(repo, workspace)

    if re.match(r"^https?://", repo):
        parsed = urlparse(repo)
        name = Path(parsed.path).stem or "repository"
        if name.endswith(".git"):  # strip trailing .git if present
            name = name[:-4]
        target = workspace / name
        if target.exists():
            # Only treat as present if it looks like a valid git repo
            if (target / '.git').exists():
                print(f"Repo already present at {target}")
                return target.resolve()
            else:
                # Clean up any leftover partial directory from a failed clone
                try:
                    shutil.rmtree(target)
                except Exception:
                    pass
        ensure_dirs(target.parent)

        # Discover remote branches and default HEAD to avoid cloning with a non-existent branch
        remote_heads: set[str] = set()
        default_branch: Optional[str] = None

        try:
            # Get default branch from HEAD symref
            res_head = subprocess.run(
                ["git", "ls-remote", "--symref", repo, "HEAD"],
                capture_output=True, text=True, check=False
            )
            for line in res_head.stdout.splitlines():
                line = line.strip()
                # Example: "ref: refs/heads/main\tHEAD"
                if line.startswith("ref:") and "refs/heads/" in line:
                    ref = line.split()[1] if "\t" not in line else line.split("\t")[0]
                    # normalize to branch name
                    if "refs/heads/" in ref:
                        default_branch = ref.split("refs/heads/")[-1]
                        break
        except Exception:
            pass  # non-fatal, we will still try sensible defaults

        try:
            # Enumerate all remote heads
            res_heads = subprocess.run(
                ["git", "ls-remote", "--heads", repo],
                capture_output=True, text=True, check=False
            )
            for line in res_heads.stdout.splitlines():
                parts = line.strip().split()  # "<sha>\trefs/heads/<name>"
                if len(parts) == 2 and parts[1].startswith("refs/heads/"):
                    remote_heads.add(parts[1].split("refs/heads/")[-1])
        except Exception:
            pass

        # Decide which branch to use for clone
        chosen_branch: Optional[str] = None
        if branch and branch in remote_heads:
            chosen_branch = branch
        elif branch and branch not in remote_heads:
            # Fall back to default branch or common names
            for cand in [default_branch, "main", "master", "trunk", None]:
                if cand is None:
                    chosen_branch = None
                    break
                if cand in remote_heads:
                    chosen_branch = cand
                    break
            if branch not in remote_heads:
                print(f"Warning: Remote branch '{branch}' not found. Using '{chosen_branch or 'default'}' instead.")
        else:
            # No branch specified: prefer default, then common names
            for cand in [default_branch, "main", "master", "trunk", None]:
                if cand is None:
                    chosen_branch = None
                    break
                if not remote_heads or cand in remote_heads:
                    chosen_branch = cand
                    break

        # Perform the clone (full history for temporal analysis)
        cmd = ["git", "clone"]
        if chosen_branch:
            cmd.extend(["-b", chosen_branch])
        cmd.extend([repo, str(target)])

        # Use our run() helper; let it raise on non-zero exit
        run(cmd)
        print(f"Successfully cloned using branch: {chosen_branch or 'default'}")
        return target.resolve()

    # treat as a local file/folder path
    path = Path(repo).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".zip":
        return obtain_zip_repo(str(path), workspace)
    if not path.exists():
        # explicit message; caller may switch to interactive prompt
        raise SystemExit(f"Repository path not found: {path}")
    return path

def analysis_root_for_single_run(repo_root: Path, workspace: Path) -> Path:
    """Keep generated single-run artifacts out of local source datasets."""
    try:
        repo_root.relative_to(workspace)
        return repo_root
    except ValueError:
        parts = []
        parent_names = {p.name.lower() for p in repo_root.parents}
        if "000_toy_examples" in parent_names or "arch_analysis_trainticket_toy_examples_multilang" in parent_names:
            parts.append("toy")
        if repo_root.parent.name.lower() in {"java", "python"}:
            parts.append(repo_root.parent.name.lower())
        parts.append(repo_root.name)
        analysis_name = "_".join(parts)
        return workspace / analysis_name / "single_analysis"

# alias for Optional to keep annotations concise below
from typing import Optional as _Optional

def find_existing_report_root(repo_root: Path) -> Optional[Path]:
    # Detect an existing DV8 analysis under OutputData
    arch_root = repo_root / "OutputData" / "Architecture-analysis-result"
    if arch_root.is_dir():
        for p in arch_root.rglob("dv8-analysis-result"):
            if p.is_dir():
                return arch_root
        return arch_root
    return None


def run_depends_via_dv8(
    dv8_console: Path,
    source_root: Path,
    output_dir: Path,
    basename: str,
    env: dict[str, str],
    logs: list | None = None,
) -> tuple[Path, _Optional[Path]]:
    """Use DV8's built-in depends parser task to generate JSON + mapping.

    Syntax per dv8-console depends:parser --help:
      dv8-console depends:parser -inputFolder <INPUTFOLDER> -language <LANGUAGE> 
                                 -outputFolder <OUTPUTFOLDER> -projectName <PROJECTNAME> [-detail]
    """
    ensure_dirs(output_dir)  # ensure output tree exists
    cmd = [
        dv8_console,
        "depends:parser",
        "-inputFolder", str(source_root),  # where source files are
        "-language", "java",               # language selector
        "-outputFolder", str(output_dir),  # where JSON will be written
        "-projectName", basename,          # base filename for outputs
    ]
    # ask for detailed extraction (may be unsupported on some builds)
    res = run(cmd + ["-detail"], env=env, check=False)
    if logs is not None:
        logs.append({
            "step": "depends:parser -detail",
            "cmd": ' '.join(str(x) for x in (cmd + ["-detail"])),
            "rc": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        })
    if res.returncode != 0:
        # if -detail fails, retry w/o it
        res2 = run(cmd, env=env)
        if logs is not None:
            logs.append({
                "step": "depends:parser",
                "cmd": ' '.join(str(x) for x in cmd),
                "rc": res2.returncode,
                "stdout": res2.stdout,
                "stderr": res2.stderr,
            })

    # find the JSON that was produced (exact match or a suffixed variant)
    exact = output_dir / f"{basename}.json"
    if exact.is_file():
        json_dep = exact
    else:
        matches = sorted(output_dir.glob(f"{basename}*.json"))
        if not matches:
            raise SystemExit("DV8 depends:parser did not emit the expected JSON file.")
        json_dep = matches[0]
    mapping = output_dir / "depends-dv8map.mapping"  # may or may not exist with dv8 output
    return json_dep, (mapping if mapping.is_file() else None)

def run_depends_via_jar(depends_jar: Path, source_root: Path, output_dir: Path, basename: str, logs: list | None = None) -> tuple[Path, Path]:
    """Fallback: call the external depends.jar if needed."""
    ensure_dirs(output_dir)
    cmd = [
        "java", "-Xmx4g", "-jar", str(depends_jar),  # launch the jar
        "java",                 # language
        str(source_root),       # input folder
        basename,               # project name (output base)
        "-f=json",              # JSON output
        "-m=true",              # include members
        "-s",                   # strip leading path
        "-p=dot",               # dot as path separator
        "--auto-include",       # include indirectly referenced files (if supported)
        "-d", str(output_dir),  # destination
    ]
    res = run(cmd, check=False)
    if logs is not None:
        logs.append({
            "step": "depends.jar",
            "cmd": ' '.join(str(x) for x in cmd),
            "rc": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        })
    if res.returncode != 0:
        # try again without --auto-include if that flag isn't supported
        cmd_no_auto = [
            "java", "-Xmx4g", "-jar", str(depends_jar),
            "java", str(source_root), basename,
            "-f=json", "-m=true", "-s", "-p=dot",
            "-d", str(output_dir),
        ]
        res2 = run(cmd_no_auto)
        if logs is not None:
            logs.append({
                "step": "depends.jar (no auto-include)",
                "cmd": ' '.join(str(x) for x in cmd_no_auto),
                "rc": res2.returncode,
                "stdout": res2.stdout,
                "stderr": res2.stderr,
            })

    # verify the expected files were emitted
    json_dep = output_dir / f"{basename}.json"
    mapping = output_dir / "depends-dv8map.mapping"
    if not json_dep.is_file() or not mapping.is_file():
        raise SystemExit("Depends did not emit the expected JSON and mapping files.")
    return json_dep, mapping

def convert_to_dsm(
    dv8_console: Path,
    json_dep: Path,
    mapping: _Optional[Path],
    out_dsm: Path,
    env: dict[str, str],
    logs: list | None = None,
) -> None:
    # turn depends JSON (+ optional map) into a DV8 .dv8-dsm matrix
    ensure_dirs(out_dsm.parent)
    cmd = [dv8_console, "core:convert-matrix", json_dep, "-outputFile", out_dsm]
    if mapping and mapping.is_file():
        cmd += ["-dependPath", mapping]  # include mapping if we have it
    res = run(cmd, env=env)
    if logs is not None:
        logs.append({
            "step": "core:convert-matrix",
            "cmd": ' '.join(str(x) for x in cmd),
            "rc": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        })


def export_git_history_log(
    repo_root: Path,
    out_log: Path,
    logs: list | None = None,
) -> bool:
    """Export git log in the DV8-recommended text format for HDSM generation."""
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return False
    ensure_dirs(out_log.parent)
    cmd = ["git", "log", "--numstat", "--date=iso"]
    res = run(cmd, cwd=repo_root, check=False)
    if logs is not None:
        logs.append({
            "step": "git log --numstat --date=iso",
            "cmd": ' '.join(str(x) for x in cmd),
            "rc": res.returncode,
            "stdout": "",
            "stderr": res.stderr,
            "output_file": str(out_log),
        })
    if res.returncode != 0:
        return False
    out_log.write_text(res.stdout, encoding="utf-8")
    return True


def convert_git_history_to_dsm(
    dv8_console: Path,
    git_log_txt: Path,
    out_dsm: Path,
    env: dict[str, str],
    logs: list | None = None,
) -> bool:
    """Build a raw HDSM from a git-log text export."""
    ensure_dirs(out_dsm.parent)
    cmd = [
        dv8_console,
        "scm:history:gittxt:convert-matrix",
        git_log_txt,
        "-outputFile",
        out_dsm,
    ]
    res = run(cmd, env=env, check=False)
    if logs is not None:
        logs.append({
            "step": "scm:history:gittxt:convert-matrix",
            "cmd": ' '.join(str(x) for x in cmd),
            "rc": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        })
    return res.returncode == 0 and out_dsm.exists()


def export_matrix_json(
    dv8_console: Path,
    dsm_path: Path,
    out_json: Path,
    env: dict[str, str],
    logs: list | None = None,
) -> bool:
    """Export any DV8 DSM to JSON so it can be inspected without DV8."""
    ensure_dirs(out_json.parent)
    cmd = [dv8_console, "core:export-matrix", dsm_path, "-outputFile", out_json]
    res = run(cmd, env=env, check=False)
    if logs is not None:
        logs.append({
            "step": "core:export-matrix",
            "cmd": ' '.join(str(x) for x in cmd),
            "rc": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        })
    return res.returncode == 0 and out_json.exists()


def generate_git_change_cost(
    dv8_console: Path,
    git_log_txt: Path,
    output_dir: Path,
    env: dict[str, str],
    logs: list | None = None,
) -> bool:
    """Generate DV8 change-cost artifacts from a git-log text export."""
    ensure_dirs(output_dir)
    cmd = [
        dv8_console,
        "scm:history:gittxt:change-cost",
        git_log_txt,
        "-outputFolder",
        output_dir,
    ]
    res = run(cmd, env=env, check=False)
    if logs is not None:
        logs.append({
            "step": "scm:history:gittxt:change-cost",
            "cmd": ' '.join(str(x) for x in cmd),
            "rc": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        })
    return res.returncode == 0


def _fix_change_cost_csv(change_cost_dir: Path) -> None:
    """Fix change-cost CSVs so DV8's ChangeCostParser doesn't crash.

    DV8's ChangeCostParser crashes with ArrayIndexOutOfBoundsException: -1 when:
    1. all-commits-change-cost.csv has empty Author field (gittxt:change-cost omits it)
    2. all-file-change-cost.csv has empty FileName rows (git log produces header-only rows
       for merge commits or binary files — DV8 can't handle them)

    Fixes applied:
    - all-commits-change-cost.csv: fill empty Author from Committer
    - all-file-change-cost.csv: drop rows with empty FileName
    """
    import csv, io

    def _fix_csv(csv_path: Path, drop_empty_col: str | None = None, fill_col: str | None = None, fill_from: str | None = None) -> None:
        if not csv_path.exists():
            return
        try:
            text = csv_path.read_text(encoding="utf-8")
            reader = csv.DictReader(io.StringIO(text))
            if reader.fieldnames is None:
                return
            rows = list(reader)
            original_count = len(rows)
            fixed = False
            if drop_empty_col:
                rows = [r for r in rows if r.get(drop_empty_col, "").strip()]
                if len(rows) < original_count:
                    fixed = True
                    print(f"  [change-cost] Dropped {original_count - len(rows)} empty-{drop_empty_col} rows from {csv_path.name}")
            if fill_col and fill_from:
                for row in rows:
                    if not row.get(fill_col, "").strip():
                        row[fill_col] = row.get(fill_from, "")
                        fixed = True
            if not fixed:
                return
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=reader.fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)
            csv_path.write_text(out.getvalue(), encoding="utf-8")
        except Exception as e:
            print(f"  [change-cost] Warning: could not fix {csv_path.name}: {e}")

    _fix_csv(change_cost_dir / "all-commits-change-cost.csv", fill_col="Author", fill_from="Committer")
    _fix_csv(change_cost_dir / "all-file-change-cost.csv", drop_empty_col="FileName")


def merge_dsms(
    dv8_console: Path,
    matrices: list[Path],
    out_dsm: Path,
    env: dict[str, str],
    logs: list | None = None,
) -> bool:
    """Merge structural and history DSMs into one matrix for anti-pattern detection."""
    ensure_dirs(out_dsm.parent)
    cmd = [dv8_console, "core:merge-matrix", *matrices, "-outputFile", out_dsm]
    res = run(cmd, env=env, check=False)
    if logs is not None:
        logs.append({
            "step": "core:merge-matrix",
            "cmd": ' '.join(str(x) for x in cmd),
            "rc": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        })
    return res.returncode == 0 and out_dsm.exists()

def write_params(
    params_path: Path,
    project_name: str,
    dsm_path: Path,
    output_dir: Path,
    *,
    run_file_stat: bool = False,
    run_hotspot: bool = True,
    run_archissue: bool = True,
    run_archroot: bool = True,
    run_metrics: bool = True,
    run_clustering: bool = True,
    run_report_doc: bool = True,
    change_cost_file: Optional[Path] = None,
) -> None:
    """Write a DV8 arch-report .properties file.

    change_cost_file: if provided, adds changeCostFile= so DV8 can locate
    all-file-change-cost.csv for hotspot ranking without searching for it.
    """
    dsm_posix = dsm_path.as_posix() if isinstance(dsm_path, Path) else dsm_path
    out_posix = output_dir.as_posix() if isinstance(output_dir, Path) else output_dir
    change_cost_line = (
        f"changeCostFile={change_cost_file.as_posix()}\n" if change_cost_file else ""
    )
    text = (
        f"## Auto-generated DV8 arch report parameters\n"
        f"\n"
        f"inputFolder=.\n"
        f"outputFolder={out_posix}\n"
        f"projectName={project_name}\n"
        f"\n"
        f"runMetrics={'on' if run_metrics else 'off'}\n"
        f"runArchissue={'on' if run_archissue else 'off'}\n"
        f"runArchroot={'on' if run_archroot else 'off'}\n"
        f"runHotspot={'on' if run_hotspot else 'off'}\n"
        f"runClustering={'on' if run_clustering else 'off'}\n"
        f"runReportDoc={'on' if run_report_doc else 'off'}\n"
        f"reportDocFormat=web\n"
        f"runFileStat={'on' if run_file_stat else 'off'}\n"
        f"runCompress=off\n"
        f"\n"
        f"sourceType=dsm\n"
        f"dependencyFilePath={dsm_posix}\n"
        f"{change_cost_line}"
    )
    params_path.write_text(text, encoding="utf-8")
    print(f"Wrote params file: {params_path}")

def run_arch_report(dv8_console: Path, params_path: Path, repo_root: Path, env: dict[str, str], logs: list | None = None) -> None:
    # invoke the full architecture analysis report
    cmd = [dv8_console, "arch-report", "-paramsFile", params_path]
    try:
        res = subprocess.run(
            [str(x) for x in cmd],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        print(f"\n$ {' '.join(str(x) for x in cmd)}")
        print("[timeout] arch-report exceeded 600s; continuing with fallbacks.")
        if logs is not None:
            logs.append({
                "step": "arch-report",
                "cmd": ' '.join(str(x) for x in cmd),
                "rc": -1,
                "stdout": "",
                "stderr": "timeout after 600s",
            })
        return
    if res.stdout:
        sys.stdout.write(res.stdout)
    if res.stderr:
        sys.stderr.write(res.stderr)
    if logs is not None:
        logs.append({
            "step": "arch-report",
            "cmd": ' '.join(str(x) for x in cmd),
            "rc": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        })
    if res.returncode != 0:
        # Surface a helpful hint for the common NumberFormatException on "N/A" values
        if "NumberFormatException" in (res.stderr or "") and '"N/A"' in (res.stderr or ""):
            print("Hint: arch-report failed parsing a non-numeric field (N/A). We disabled runFileStat to mitigate this. If the issue persists, try re-running with fewer report sections or use metrics + arch-issue summaries.")
        # still raise to mark the step as failed
        raise SystemExit(f"arch-report failed ({res.returncode})")

def run_additional_dv8_tasks(
    dv8_console: Path,
    dsm_path: Path,
    output_dir: Path,
    env: dict[str, str],
    arch_issue_dsm_path: Path | None = None,
    project_name: str = "project",
    logs: list | None = None,
    mv_cochange: int | None = None,
) -> None:
    """Run direct DV8 tasks to extract anti-patterns, debt cost, hotspots, and DRH.

    This complements or replaces arch-report outputs when arch-report is unstable.
    """
    def _log(res, step, cmd):
        if logs is not None:
            logs.append({
                "step": step,
                "cmd": ' '.join(str(x) for x in cmd),
                "rc": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
            })

    def _run_with_timeout(cmd, step, timeout_s: int = 180) -> None:
        try:
            res = subprocess.run(
                [str(x) for x in cmd],
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            print(f"\n$ {' '.join(str(x) for x in cmd)}")
            print(f"[timeout] {step} exceeded {timeout_s}s; skipping.")
            if logs is not None:
                logs.append({
                    "step": step,
                    "cmd": ' '.join(str(x) for x in cmd),
                    "rc": -1,
                    "stdout": "",
                    "stderr": f"timeout after {timeout_s}s",
                })
            return
        if res.stdout:
            sys.stdout.write(res.stdout)
        if res.stderr:
            sys.stderr.write(res.stderr)
        _log(res, step, cmd)

    # Ensure target folders exist
    arch_issue_dir = output_dir / "arch-issue"
    arch_issue_cost_dir = output_dir / "arch-issue-cost"
    hotspot_dir = output_dir / "hotspot"
    drh_dir = output_dir / "drh"
    drh_export_dir = output_dir / "dv8-analysis-result" / "dsm"
    dsm_export_dir = output_dir / "dsm"
    ensure_dirs(arch_issue_dir, arch_issue_cost_dir, hotspot_dir, drh_dir, drh_export_dir, dsm_export_dir)

    arch_issue_input = arch_issue_dsm_path or dsm_path

    # 1) Architecture issues (instances)
    cmd = [dv8_console, "arch-issue:arch-issue", arch_issue_input, "-outputFolder", arch_issue_dir]
    if mv_cochange is not None:
        cmd += ["-mvCochange", str(mv_cochange)]
    res = run(cmd, env=env, check=False)
    _log(res, "arch-issue:arch-issue", cmd)

    # 2) Debt cost summary for issues (structural only — no -changeCost).
    # DV8's ChangeCostParser crashes with ArrayIndexOutOfBoundsException when the
    # gittxt:change-cost CSV has empty Author/FileName fields (a known DV8 bug).
    # Running without -changeCost produces structural cost (FileCount per instance)
    # which is what we need for % coverage. Churn data is sourced from our own pipeline.
    cmd = [dv8_console, "debt:arch-issue-cost", arch_issue_input, "-asDir", arch_issue_dir, "-outputFolder", arch_issue_cost_dir]
    res = run(cmd, env=env, check=False)
    _log(res, "debt:arch-issue-cost", cmd)

    # 3) Design Rule Hierarchy (clustering)
    drh_clsx = drh_export_dir / "drh-clustering.dv8-clsx"
    drh_json = drh_export_dir / "drh-clustering.json"
    cmd = [dv8_console, "dr-hier:dr-hier", dsm_path, "-outputFile", drh_clsx]
    res = run(cmd, env=env, check=False)
    _log(res, "dr-hier:dr-hier", cmd)
    cmd = [dv8_console, "core:export-cluster", drh_clsx, "-outputFile", drh_json]
    res = run(cmd, env=env, check=False)
    _log(res, "core:export-cluster", cmd)

    # 4) Export matrix  DSM(JSON) — useful for downstream tooling
    matrix_json = dsm_export_dir / "matrix.json"
    cmd = [dv8_console, "core:export-matrix", dsm_path, "-outputFile", matrix_json]
    res = run(cmd, env=env, check=False)
    _log(res, "core:export-matrix", cmd)

    # 5) Hotspots and hotspot cost (requires change-cost input)
    # Resolve the file-level CSV specifically — arch-report and standalone hotspot both need
    # all-file-change-cost.csv (per-file churn). all-commits-change-cost.csv is commits-level
    # and causes ChangeCostParser crashes when passed to hotspot commands.
    change_cost = next(
        (p for p in output_dir.rglob("all-file-change-cost.csv") if p.is_file()),
        None,
    )
    if change_cost:
        cmd = [dv8_console, "hotspot:hotspot", dsm_path, "-changeCost", change_cost, "-outputFolder", hotspot_dir]
        _run_with_timeout(cmd, "hotspot:hotspot", timeout_s=180)

        cmd = [
            dv8_console,
            "hotspot:hotspot-cost",
            dsm_path,
            "-changeCost",
            change_cost,
            "-hotspotFolder",
            hotspot_dir,
            "-outputFolder",
            hotspot_dir,
        ]
        _run_with_timeout(cmd, "hotspot:hotspot-cost", timeout_s=180)

        # 5b) Run arch-report with runHotspot=on to get ranked hotspot CSVs
        # (active-hotspot-index.csv, active-hotspot-roi.csv etc.).
        # The standalone hotspot:hotspot command only produces change-matrix.dv8-dsm —
        # ranked output is only generated by arch-report's internal hotspot pipeline.
        roi_csv = hotspot_dir / "active-hotspot-roi.csv"
        if not roi_csv.exists():
            params_dir = output_dir.parent / "InputData"
            params_dir.mkdir(parents=True, exist_ok=True)
            hotspot_params = params_dir / "archreport_hotspot.properties"
            write_params(
                hotspot_params,
                project_name,
                dsm_path,
                output_dir,
                run_file_stat=False,
                run_hotspot=True,
                run_archissue=False,
                run_archroot=False,
                run_metrics=False,
                run_clustering=False,
                run_report_doc=False,
                change_cost_file=change_cost,
            )
            try:
                run_arch_report(dv8_console, hotspot_params, output_dir.parent, env, logs=logs)
            except SystemExit:
                print("  [hotspot] arch-report hotspot run failed; falling back to computed hotspot CSV.")
                try:
                    from generate_hotspot_csv import generate_hotspot_csv as _gen_hotspot
                    written = _gen_hotspot(output_dir, force=False)
                    if written:
                        print("  [hotspot] Generated active-hotspot-roi.csv from DSM + churn data (arch-report fallback)")
                except Exception as _exc:
                    print(f"  [hotspot] Fallback hotspot generation failed: {_exc}")
    else:
        print("  [hotspot] all-file-change-cost.csv not found; skipping hotspot generation.")


def summarize_outputs(output_dir: Path) -> None:
    """Print a concise summary of key outputs for quick navigation."""
    try:
        print("\nSummary of outputs:")
        # Metrics and DSM
        dsm = output_dir / "repo.dv8-dsm"
        metrics_dir = output_dir / "metrics"
        print(f"  DSM: {dsm if dsm.exists() else 'not found'}")
        print(f"  Metrics dir: {metrics_dir if metrics_dir.exists() else 'not found'}")

        # Arch-report (HTML)
        report_dirs = list(output_dir.rglob("dv8-analysis-result"))
        if report_dirs:
            print(f"  Arch-report: {report_dirs[0]}")
        else:
            print("  Arch-report: not found (see tool logs)")

        # Direct DV8 tasks
        ai = output_dir / "arch-issue"
        aic = output_dir / "arch-issue-cost"
        hs = output_dir / "hotspot"
        drh = output_dir / "drh"
        dsm_json = output_dir / "dsm" / "matrix.json"
        history_dsm = output_dir / "history-dsm" / "history_raw.dv8-dsm"
        history_dsm_json = output_dir / "history-dsm" / "history_raw.matrix.json"
        merged_dsm = output_dir / "history-dsm" / "struct_plus_history.dv8-dsm"
        merged_dsm_json = output_dir / "history-dsm" / "struct_plus_history.matrix.json"
        print(f"  Arch-issues: {ai if ai.exists() else 'not found'}")
        print(f"  Debt cost: {aic if aic.exists() else 'not found'}")
        print(f"  Hotspots: {hs if hs.exists() else 'not found'}")
        print(f"  DR-Hierarchy: {drh if drh.exists() else 'not found'}")
        print(f"  Matrix JSON: {dsm_json if dsm_json.exists() else 'not found'}")
        print(f"  History DSM: {history_dsm if history_dsm.exists() else 'not found'}")
        print(f"  History DSM JSON: {history_dsm_json if history_dsm_json.exists() else 'not found'}")
        print(f"  Merged anti-issue DSM: {merged_dsm if merged_dsm.exists() else 'not found'}")
        print(f"  Merged anti-issue JSON: {merged_dsm_json if merged_dsm_json.exists() else 'not found'}")

        # Tool logs
        tool_log = output_dir / "tool_logs" / "DEPENDS_DV8_OUTPUT.json"
        print(f"  Tool logs: {tool_log if tool_log.exists() else 'not found'}")
    except Exception:
        pass

def run_metric_task(
    dv8_console: Path,
    metric: str,
    dsm_path: Path,
    output_dir: Path,
    env: dict[str, str],
) -> Path:
    # run an individual metric directly against a DSM (bypass report if needed)
    key = metric.lower().strip().replace("_", " ").replace("-", " ")
    # normalize common aliases
    if key in {"m", "m score", "mscore"}:
        metric_key = "m-score"
    elif "decoupling" in key:
        metric_key = "decoupling-level"
    elif "independence" in key:
        metric_key = "independence-level"
    elif "propagation" in key and "cost" in key:
        metric_key = "propagation-cost"
    else:
        metric_key = metric.lower().strip()

    metric_map = {
        "m-score": "metrics:m-score",
        "m": "metrics:m-score",
        "m score": "metrics:m-score",
        "mscore": "metrics:m-score",
        "decoupling-level": "metrics:decoupling-level",
        "decoupling level": "metrics:decoupling-level",
        "dl": "metrics:decoupling-level",
        "independence-level": "metrics:independence-level",
        "independence level": "metrics:independence-level",
        "il": "metrics:independence-level",
        "propagation-cost": "metrics:propagation-cost",
        "propagation cost": "metrics:propagation-cost",
        "propagation_cost": "metrics:propagation-cost",
        "pc": "metrics:propagation-cost",
    }
    task = metric_map.get(metric_key)
    if not task:
        raise SystemExit(f"Unsupported metric query: {metric}")

    metrics_dir = output_dir / "metrics"
    ensure_dirs(metrics_dir)
    out_file = metrics_dir / (task.split(":", 1)[1] + ".json")  # file name based on task

    cmd = [dv8_console, task, dsm_path, "-outputFile", out_file]
    run(cmd, env=env)
    return out_file


def compute_all_metrics(
    dv8_console: Path,
    dsm_path: Path,
    output_dir: Path,
    env: dict[str, str],
    logs: list | None = None,
) -> dict:
    metrics = {}
    metrics_root = output_dir / "metrics"
    ensure_dirs(metrics_root)
    tasks = [
        ("propagation-cost", "metrics:propagation-cost"),
        ("decoupling-level", "metrics:decoupling-level"),
        ("independence-level", "metrics:independence-level"),
        ("m-score", "metrics:m-score"),
    ]
    for key, task in tasks:
        out_file = metrics_root / f"{key}.json"
        cmd = [dv8_console, task, dsm_path, "-outputFile", out_file]
        res = run(cmd, env=env)
        if logs is not None:
            logs.append({
                "step": task,
                "cmd": ' '.join(str(x) for x in cmd),
                "rc": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
            })
        try:
            metrics[key] = json.loads(out_file.read_text())
        except json.JSONDecodeError:
            metrics[key] = {"file": str(out_file)}
    # write combined file for downstream consumers
    combined = metrics_root / "all-metrics.json"
    combined.write_text(json.dumps(metrics, indent=2))
    return metrics

def fetch_metric(report_root: Path, query: str) -> Optional[dict]:
    # read metric JSONs from the arch-report output tree and return the one requested
    metrics_dir = None
    for candidate in report_root.rglob("dv8-analysis-result"):  # find analysis root
        maybe = candidate / "metrics"
        if maybe.is_dir():
            metrics_dir = maybe
            break
    if not metrics_dir:
        return None

    data: dict[str, dict] = {}
    for name in ["m-score.json", "decoupling-level.json", "propagation-cost.json", "independence-level.json"]:
        file_path = metrics_dir / name
        if file_path.is_file():
            try:
                data[name] = json.loads(file_path.read_text())
            except json.JSONDecodeError:
                pass  # ignore malformed

    key = query.lower().strip()
    if key in {"m", "m-score", "mscore"}:
        return data.get("m-score.json")
    if key in {"dl", "decoupling", "decoupling level", "decoupling-level"}:
        return data.get("decoupling-level.json")
    if key in {"pc", "propagation", "propagation cost", "propagation-cost"}:
        return data.get("propagation-cost.json")
    if key in {"il", "independence", "independence level", "independence-level"}:
        return data.get("independence-level.json")
    return data or None  # if user asked something else, return whatever we collected


# --- temporal analysis -------------------------------------------------------

def get_commit_history(
    repo_path: Path,
    branch: str = "main",
    count: int = 10,
    intelligent: bool = False,
    min_months_apart: int = 0,
    min_commits_apart: int = 0,
    since_date: str | None = None,
    until_date: str | None = None,
    spacing_mode: str = "alltime",
) -> List[Dict[str, str]]:
    """
    Get commits - TWO SIMPLE MODES:
    1. ALL-TIME: First ever, last ever, interpolated in between
    2. RECENT-MAJOR: Recent commits with min_months_apart spacing

    Args:
        repo_path: Path to the Git repository
        branch: Branch name to analyze
        count: Number of commits to retrieve
        intelligent: IGNORED (kept for compatibility)
        min_months_apart: 0=all-time mode, >0=recent-major mode with N months spacing
        spacing_mode: "alltime" or "recent"
    """
    repo_path = repo_path.resolve()

    # Determine mode and fetch count
    if spacing_mode == "smart":
        fetch_count = 100000  # Need all commits to rank by change size
    elif min_months_apart > 0:
        spacing_mode = "recent"
        # For recent mode with spacing, need to look back far enough
        fetch_count = count * min_months_apart * 100  # Look back enough commits
    elif min_commits_apart > 0:
        spacing_mode = "commits"
        # Need at least (count-1)*gap + 1 commits
        fetch_count = max((count - 1) * min_commits_apart + 1, count)
    else:
        spacing_mode = "alltime"
        # For alltime mode, get everything to find first/last
        fetch_count = 100000  # Get all commits

    # Try specified branch, then common alternatives, then trunk (for SVN-style repos like PDFBox), then HEAD
    for br in [branch, "main", "master", "trunk", "develop", "HEAD"]:
        try:
            # Get commits with numstat to see file changes
            cmd = ["git", "log", f"{br}", f"-{fetch_count}", "--pretty=format:%H|%ai|%an|%s", "--numstat"]
            if since_date:
                cmd.insert(3, f"--since={since_date}")
            if until_date:
                cmd.insert(4 if since_date else 3, f"--until={until_date}")
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout.strip():  # Make sure we got actual commits
                break
        except subprocess.CalledProcessError:
            continue
    else:
        raise RuntimeError(f"Could not retrieve git log from {repo_path}. Is it a git repo?")

    commits = []
    lines = result.stdout.strip().split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line or '|' not in line:
            i += 1
            continue

        parts = line.split('|', 3)
        if len(parts) == 4:
            commit_hash, date_str, author, message = parts

            # Count file changes for this commit
            files_changed = 0
            i += 1
            while i < len(lines) and lines[i] and '|' not in lines[i]:
                # Lines like: "5    3    path/to/file.java"
                files_changed += 1
                i += 1

            commit_data = {
                'hash': commit_hash,
                'short_hash': commit_hash[:8],
                'date': date_str,
                'author': author,
                'message': message.strip(),
                'files_changed': files_changed
            }

            commits.append(commit_data)  # Collect all commits first
        else:
            i += 1

    # Git log order can be topological around merge commits, which may not be
    # strictly chronological. Sort explicitly so temporal sampling is monotonic.
    commits.sort(key=lambda c: parse_git_commit_datetime(c.get('date', '')), reverse=True)

    # Apply spacing mode strategies
    if spacing_mode == "alltime":
        # MODE 1: ALL-TIME - First ever, last ever, interpolated in between
        if len(commits) < count:
            # Not enough commits, return what we have
            return commits

        selected = [commits[0]]  # Most recent
        if count > 2:
            # Calculate indices for even spacing across ALL history
            total_commits = len(commits)
            step = (total_commits - 1) / (count - 1)
            for i in range(1, count - 1):
                idx = int(i * step)
                selected.append(commits[idx])
        if count > 1:
            selected.append(commits[-1])  # Oldest (first ever)
        return selected

    elif spacing_mode == "recent":
        # MODE 2: RECENT-MAJOR - Recent commits with min_months_apart spacing
        min_days = min_months_apart * 30
        selected = []

        for commit in commits:
            # Check time spacing
            time_ok = True
            if selected:
                try:
                    last_date = datetime.fromisoformat(selected[-1]['date'].split()[0])
                    curr_date = datetime.fromisoformat(commit['date'].split()[0])
                    days_diff = abs((last_date - curr_date).days)
                    time_ok = days_diff >= min_days
                except:
                    time_ok = True

            if time_ok:
                selected.append(commit)

            if len(selected) >= count:
                break

        # If we didn't get enough, include first commit even if spacing is violated
        if len(selected) < count and len(commits) > len(selected):
            if commits[-1] not in selected:
                selected.append(commits[-1])

        return selected

    elif spacing_mode == "commits":
        # MODE 3: RECENT-COMMITS - Select commits with fixed commit-count gaps
        selected = []
        if not commits:
            return selected
        for i_gap in range(count):
            idx = i_gap * max(1, min_commits_apart)
            if idx < len(commits):
                selected.append(commits[idx])
            else:
                break
        # If we didn't gather enough due to shallow history, try to append oldest
        if len(selected) < count and commits[-1] not in selected:
            selected.append(commits[-1])
        return selected

    elif spacing_mode == "smart":
        # MODE 4: SMART - Select commits with the most file changes (largest diffs).
        # Runs git log --numstat to count added+removed lines per commit, then picks top N.
        import subprocess as _sp
        result2 = _sp.run(
            ["git", "log", branch, "--pretty=format:%H", "--numstat"],
            cwd=repo_path, capture_output=True, text=True
        )
        change_counts: dict = {}
        current_hash = None
        for line in result2.stdout.split('\n'):
            line = line.strip()
            if not line:
                continue
            if len(line) == 40 and all(c in '0123456789abcdef' for c in line):
                current_hash = line
                change_counts[current_hash] = 0
            elif current_hash and '\t' in line:
                parts = line.split('\t')
                try:
                    added = int(parts[0]) if parts[0] != '-' else 0
                    removed = int(parts[1]) if parts[1] != '-' else 0
                    change_counts[current_hash] += added + removed
                except (ValueError, IndexError):
                    pass
        ranked = sorted(commits, key=lambda c: change_counts.get(c['hash'], 0), reverse=True)
        selected = ranked[:count]
        selected.sort(key=lambda c: c['date'], reverse=True)  # newest-first like other modes
        return selected

    # Default fallback
    return commits[:count]


def _is_empty_revision(metrics: dict) -> bool:
    """Return True if a revision produced no usable DV8 metrics (empty/trivial repo state)."""
    keys = ("m-score", "propagation-cost", "decoupling-level", "independence-level")
    vals = [metrics.get(k) for k in keys]
    if all(v is None for v in vals):
        return True
    numeric = [v for v in vals if isinstance(v, (int, float))]
    if numeric and all(v == 0.0 for v in numeric):
        return True
    return False


def checkout_commit_to_folder(repo_path: Path, commit_hash: str, commit_date: str, workspace: Path, revision_number: int) -> Path:
    """Checkout a specific commit to a numbered folder using git archive (clean, no .git folder)."""
    repo_name = repo_path.name

    # Format date and time as DDMMYYYY_HHMM from "2025-10-08 12:34:11 +0000"
    try:
        # Split: "2025-10-08" and "12:34:11"
        date_part, time_part = commit_date.split()[:2]
        date_obj = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")
        formatted_datetime = date_obj.strftime("%d%m%Y_%H%M")
    except Exception:
        formatted_datetime = commit_hash[:8]

    # Create directory name: 01_pdfbox_08102025_1234
    checkout_dir = workspace / f"{revision_number:02d}_{repo_name}_{formatted_datetime}"

    if checkout_dir.exists():
        print(f"  Checkout already exists: {checkout_dir.name}")
        return checkout_dir

    print(f"  Checking out to {checkout_dir.name}...")

    # Use git archive to export clean source code without .git folder
    # This is much more efficient than git clone!
    ensure_dirs(checkout_dir)

    # Extract the archive directly using pipe (no temp files)
    # git archive outputs binary tar data, so we pipe it directly to tar
    subprocess.run(
        f"cd '{repo_path}' && git archive '{commit_hash}' | tar -x -C '{checkout_dir}'",
        shell=True,
        check=True
    )

    print(f"    ✓ Extracted source (clean, no .git folder)")

    return checkout_dir


def checkout_commit_to_worktree(
    repo_path: Path,
    commit_hash: str,
    commit_date: str,
    workspace: Path,
    revision_number: int,
) -> Path:
    """Checkout a specific commit using git worktree (keeps .git for arch-report)."""
    repo_name = repo_path.name

    try:
        date_part, time_part = commit_date.split()[:2]
        date_obj = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")
        formatted_datetime = date_obj.strftime("%d%m%Y_%H%M")
    except Exception:
        formatted_datetime = commit_hash[:8]

    checkout_dir = workspace / f"{revision_number:02d}_{repo_name}_{formatted_datetime}"

    if checkout_dir.exists():
        print(f"  Worktree already exists: {checkout_dir.name}")
        return checkout_dir

    print(f"  Checking out worktree to {checkout_dir.name}...")
    ensure_dirs(checkout_dir.parent)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(checkout_dir), commit_hash],
        cwd=repo_path,
        check=True,
    )
    return checkout_dir


def analyze_single_revision(
    revision_path: Path,
    dv8_console: Path,
    env: dict,
    source_path: Optional[str] = None,
    fine_grain: bool = False,
    skip_arch_report: bool = False,
    scope: str = "full",
    language: Optional[str] = None,
    neodepends_root: Optional[str] = None,
    neodepends_bin: Optional[str] = None,
    neodepends_resolver: str = "stackgraphs",
    java_depends: bool = False,
    depends_runner: str = "dv8",
    understand_und: Optional[str] = None,
    understand_upython: Optional[str] = None,
    understand_language: Optional[str] = None,
    understand_granularity: str = "entity",
    mv_cochange: int | None = None,
) -> Dict[str, Any]:
    """Run complete DV8 analysis pipeline on a single revision. Returns metrics dict."""
    print(f"\n  Analyzing {revision_path.name}...")

    # Determine source root
    source_root = None
    if source_path:
        p = Path(source_path)
        if not p.is_absolute():
            p = revision_path / p
        if p.exists():
            source_root = p

    if source_root is None:
        for rel in ("src/main/java", "src", "SourceCode"):
            cand = revision_path / rel
            if cand.exists():
                source_root = cand
                break

    if source_root is None:
        source_root = revision_path

    # Setup directories
    input_data = revision_path / "InputData"
    depends_output = input_data / "DependsOutput" / "json"
    neodepends_output = input_data / "NeoDependsOutput"
    output_data = revision_path / "OutputData"
    history_output = output_data / "history-dsm"
    ensure_dirs(input_data, depends_output, output_data, neodepends_output)

    project_name = guess_project_name(revision_path)
    basename = f"{project_name}-depends"

    # Collect step logs for troubleshooting
    step_logs: list = []

    try:
        # Step 1: Run dependency extraction (Python/Java via NeoDepends; others via DV8 Depends)
        lang = detect_language(source_root, language)
        if lang == "python":
            adjusted = auto_adjust_python_root(source_root)
            if adjusted != source_root:
                print(f"  Python source root auto-adjusted to: {adjusted}")
                source_root = adjusted
        # Optional deterministic scoping (e.g., prod-only) before dependency extraction
        scoped = build_scoped_source_root(revision_path, source_root, lang=lang, scope=scope)
        if scoped != source_root:
            print(f"  Scope '{scope}' source root: {scoped}")
            source_root = scoped
        json_dep: Path
        mapping: _Optional[Path] = None
        # Auto-select Understand for C#/TypeScript languages (no other extractor supports them)
        if depends_runner != "understand" and lang in ("csharp", "typescript", "mixed-cs-ts"):
            depends_runner = "understand"
            print(f"  [deps] Auto-selecting Understand for {lang} source.")
        if depends_runner == "understand":
            understand_output = input_data / "UnderstandOutput"
            print(
                "  Using SciTools Understand for dependency extraction "
                f"({understand_language_name(lang, understand_language)})."
            )
            json_dep = run_understand_export(
                source_root=source_root,
                output_dir=understand_output,
                project_name=project_name,
                language=understand_language_name(lang, understand_language),
                und_bin=understand_und,
                upython_bin=understand_upython,
                granularity=understand_granularity,
                logs=step_logs,
            )
            mapping = None
        else:
            use_neodepends = lang in {"python", "java"} and not (lang == "java" and java_depends)
            if use_neodepends:
                nd_root = resolve_neodepends_root(neodepends_root)
                if nd_root:
                    if lang == "java":
                        print("  Java source detected. Using NeoDepends for dependency extraction.")
                    else:
                        print("  Python source detected. Using NeoDepends for dependency extraction.")
                    json_dep = run_neodepends_python_export(
                        source_root=source_root,
                        output_dir=neodepends_output,
                        neodepends_root=nd_root,
                        neodepends_bin=neodepends_bin,
                        resolver=neodepends_resolver,
                        config="default",
                        langs=lang,
                        logs=step_logs,
                    )
                elif lang == "java":
                    print("  NeoDepends not available for Java. Falling back to DV8 Depends.")
                    json_dep, mapping = run_depends_via_dv8(
                        dv8_console, source_root, depends_output, basename, env, logs=step_logs
                    )
                else:
                    raise SystemExit("NeoDepends root not found. Set --neodepends-root or NEODEPENDS_ROOT.")
            else:
                json_dep, mapping = run_depends_via_dv8(
                    dv8_console, source_root, depends_output, basename, env, logs=step_logs
                )

        # Step 2: Convert to DSM
        dsm_path = output_data / "repo.dv8-dsm"
        convert_to_dsm(dv8_console, json_dep, mapping, dsm_path, env, logs=step_logs)
        if not dsm_path.is_file():
            raise SystemExit(f"core:convert-matrix exited 0 but did not produce {dsm_path}. "
                             "Try --understand-granularity file if using Understand.")

        # Step 2b: Build history DSM from git log and merge it with the structural DSM
        history_log_path = history_output / "git-history.txt"
        history_dsm_path = history_output / "history_raw.dv8-dsm"
        history_dsm_json_path = history_output / "history_raw.matrix.json"
        merged_arch_issue_dsm_path = history_output / "struct_plus_history.dv8-dsm"
        merged_arch_issue_json_path = history_output / "struct_plus_history.matrix.json"
        has_history_dsm = False
        if export_git_history_log(revision_path, history_log_path, logs=step_logs):
            generate_git_change_cost(
                dv8_console,
                history_log_path,
                history_output / "change-cost",
                env,
                logs=step_logs,
            )
            _fix_change_cost_csv(history_output / "change-cost")
            has_history_dsm = convert_git_history_to_dsm(
                dv8_console,
                history_log_path,
                history_dsm_path,
                env,
                logs=step_logs,
            )
            if has_history_dsm:
                export_matrix_json(
                    dv8_console,
                    history_dsm_path,
                    history_dsm_json_path,
                    env,
                    logs=step_logs,
                )
                has_history_dsm = merge_dsms(
                    dv8_console,
                    [dsm_path, history_dsm_path],
                    merged_arch_issue_dsm_path,
                    env,
                    logs=step_logs,
                )
                if has_history_dsm:
                    export_matrix_json(
                        dv8_console,
                        merged_arch_issue_dsm_path,
                        merged_arch_issue_json_path,
                        env,
                        logs=step_logs,
                    )
        arch_issue_dsm_path = merged_arch_issue_dsm_path if has_history_dsm else dsm_path

        # Step 3: Compute metrics
        all_metrics = compute_all_metrics(dv8_console, dsm_path, output_data, env, logs=step_logs)

        # Optional: Full arch-report (anti-patterns, clustering, hotspots)
        if fine_grain:
            if not skip_arch_report:
                params_dir = revision_path / "InputData"
                params_dir.mkdir(parents=True, exist_ok=True)

                # Only attempt hotspot if the file-level change-cost CSV exists.
                # Use all-file-change-cost.csv specifically — arch-report's hotspot needs
                # per-file churn data, not the commits-level all-commits-change-cost.csv
                # (which alphabetically sorts first and was previously picked by the rglob).
                change_cost = next(
                    (p for p in output_data.rglob("all-file-change-cost.csv") if p.is_file()),
                    None,
                )
                run_hotspot = change_cost is not None

                params_path = params_dir / "archreport.properties"
                write_params(
                    params_path,
                    project_name,
                    dsm_path,
                    output_data,
                    run_file_stat=False,
                    run_hotspot=run_hotspot,
                    run_archissue=False,
                    run_archroot=False,  # NPE in createVersionsLanguageOverview when runArchissue=off seeds DB first
                    run_metrics=True,
                    run_clustering=True,
                    run_report_doc=False,  # requires runArchroot=on, skip to avoid cascade NPE
                    change_cost_file=change_cost,
                )
                try:
                    run_arch_report(dv8_console, params_path, revision_path, env, logs=step_logs)
                except SystemExit:
                    # Best-effort only; fall back to direct tasks below.
                    pass

        # Always run direct DV8 tasks for anti-patterns/debt/hotspots/DRH — regardless of
        # fine_grain flag. This ensures hotspot ROI CSVs are produced whenever a DSM exists,
        # and arch-issue/cost data is always available for the QA pipeline.
        run_additional_dv8_tasks(
            dv8_console,
            dsm_path,
            output_data,
            env,
            arch_issue_dsm_path=arch_issue_dsm_path,
            project_name=project_name,
            logs=step_logs,
            mv_cochange=mv_cochange,
        )

        # Extract values
        metrics = {}
        metrics_dir = output_data / "metrics"

        def _parse_percent(v):
            if v is None:
                return None
            if isinstance(v, (int, float)):
                try:
                    if v != v or v == float('inf') or v == float('-inf'):
                        return None
                    return float(v)
                except Exception:
                    return None
            if isinstance(v, str):
                s = v.strip()
                if s in {"∞", "inf", "-inf", "Infinity", "-Infinity"}:
                    return None
                if s.endswith('%'):
                    s = s[:-1]
                try:
                    return float(s)
                except Exception:
                    return None
            return None

        def _parse_int(v):
            try:
                if v is None:
                    return None
                if isinstance(v, (int, float)):
                    return int(v)
                s = str(v).strip()
                # remove commas if present
                s = s.replace(',', '')
                return int(float(s))
            except Exception:
                return None

        for metric_name in ["propagation-cost", "m-score", "decoupling-level", "independence-level"]:
            metric_file = metrics_dir / f"{metric_name}.json"
            if metric_file.exists():
                with open(metric_file) as f:
                    data = json.load(f)
                    if metric_name == "propagation-cost":
                        raw_pc = data.get("propagationCost")
                        val = _parse_percent(raw_pc)
                        # Handle INF or None by remapping when counts are zero
                        if val is None:
                            iso = _parse_int(data.get("numberOfIsolatedItems"))
                            total = _parse_int(data.get("numberOfItems") or data.get("totalItems") or data.get("itemCount"))
                            if (iso is not None and iso == 0) or (total is not None and total == 0):
                                val = 0.0
                            else:
                                # Try alternate field excluding isolated items
                                alt = _parse_percent(data.get("propagationCostExcludeIsolatedItems"))
                                if alt is not None:
                                    val = alt
                    elif metric_name == "m-score":
                        val = _parse_percent(data.get("mScore"))
                    elif metric_name == "decoupling-level":
                        val = _parse_percent(data.get("decouplingLevel"))
                    elif metric_name == "independence-level":
                        val = _parse_percent(data.get("independenceLevel"))
                    else:
                        val = _parse_percent(data.get("value"))

                    metrics[metric_name] = val
            else:
                metrics[metric_name] = None

        # Persist tool logs
        try:
            logs_dir = output_data / "tool_logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / "DEPENDS_DV8_OUTPUT.json").write_text(json.dumps(step_logs, indent=2))
        except Exception:
            pass

        # Print a quick navigation summary
        summarize_outputs(output_data)

        return metrics

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Persist any logs we captured to help diagnose errors
        try:
            logs_dir = output_data / "tool_logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / "DEPENDS_DV8_OUTPUT.json").write_text(json.dumps(step_logs, indent=2))
        except Exception:
            pass
        summarize_outputs(output_data)
        return {
            "propagation-cost": None,
            "m-score": None,
            "decoupling-level": None,
            "independence-level": None,
            "error": str(e)
        }


def generate_plots(timeseries_file: Path, output_dir: Path) -> Path:
    """Generate beautiful plots with threshold zones using metric_plotter.py."""
    # Create plots directory inside INPUT_INTERPRETATION
    plots_dir = output_dir / "INPUT_INTERPRETATION" / "plots" / "time_evolution_modularity_metrics"
    ensure_dirs(plots_dir)

    # Call the existing metric_plotter.py script
    plotter_script = Path(__file__).parent / "metric_plotter.py"

    if not plotter_script.exists():
        raise RuntimeError(f"metric_plotter.py not found at {plotter_script}")

    # Convert to absolute paths for subprocess
    abs_timeseries = timeseries_file.resolve()
    abs_plots_dir = plots_dir.resolve()
    abs_plotter = plotter_script.resolve()

    # Run the plotter with correct arguments and absolute paths
    result = subprocess.run(
        ["python3", str(abs_plotter), "--json", str(abs_timeseries), "--output", str(abs_plots_dir)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Plotting failed:\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}")

    print(f"Plots generated with threshold zones.")
    # Print summary output from plotter
    for line in result.stdout.split('\n'):
        if 'Saved:' in line or 'All plots' in line:
            print(f"   {line.strip()}")

    # Also attempt anti-pattern plots if arch-issue summaries exist
    try:
        anti_plotter = Path(__file__).parent / "anti_pattern_plotter.py"
        if anti_plotter.exists():
            result2 = subprocess.run(
                ["python3", str(anti_plotter), "--temporal", str(output_dir.parent), "--output", str(plots_dir / "antipatterns")],
                capture_output=True,
                text=True
            )
            # Print minimal summary from anti-plotter
            for line in result2.stdout.split('\n'):
                if 'Saved:' in line or 'All anti-pattern plots' in line:
                    print(f"   {line.strip()}")
    except Exception as e:
        print(f"Note: Anti-pattern plotting skipped ({e})")
    return plots_dir


def run_temporal_analysis(
    repo_path: Path,
    revision_count: int,
    branch: str,
    workspace: Path,
    source_path: Optional[str],
    dv8_console: Path,
    env: dict,
    scope: str = "full",
    language: Optional[str] = None,
    neodepends_root: Optional[str] = None,
    neodepends_bin: Optional[str] = None,
    neodepends_resolver: str = "stackgraphs",
    java_depends: bool = False,
    depends_runner: str = "dv8",
    understand_und: Optional[str] = None,
    understand_upython: Optional[str] = None,
    understand_language: Optional[str] = None,
    understand_granularity: str = "entity",
    analysis_tag: Optional[str] = None,
    intelligent: bool = False,
    min_months_apart: int = 0,
    min_commits_apart: int = 0,
    since_date: str | None = None,
    until_date: str | None = None,
    fine_grain: bool = False,
    skip_arch_report: bool = False,
    use_worktree: bool = True,
    spacing_mode: str = "intelligent",
    mv_cochange: int | None = None,
) -> Path:
    """Analyze multiple Git revisions and save time-series data with flexible spacing strategies."""

    if not (repo_path / ".git").exists():
        raise RuntimeError(f"Not a git repository: {repo_path}")

    repo_name = repo_path.name

    print(f"\n{'='*60}")
    print(f"TEMPORAL ANALYSIS: {repo_name}")
    print(f"  Revisions: {revision_count}")
    print(f"  Branch: {branch}")
    if scope and scope != "full":
        print(f"  Scope: {scope}")
    print(f"{'='*60}\n")
    if fine_grain:
        print(f"  (Fine-grain ON: arch-report + anti-patterns will run for each revision)\n")

    # Make sure we're on a proper branch (not detached HEAD)
    print(f"Checking out branch '{branch}'...")
    try:
        result = subprocess.run(["git", "checkout", branch], cwd=repo_path, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Note: Could not checkout '{branch}': {result.stderr.strip()}")
            print(f"  Attempting to use current HEAD state...")
    except Exception as e:
        print(f"  Note: Git checkout failed: {e}")

    # Get commits
    if intelligent:
        print(f"Intelligently selecting {revision_count} meaningful commits (releases, fixes, major changes)...")
    else:
        if min_months_apart > 0:
            print(f"Selecting last {revision_count} commits with ≥{min_months_apart} month spacing...")
        elif min_commits_apart > 0:
            print(f"Selecting last {revision_count} commits with {min_commits_apart} commits spacing...")
        else:
            print(f"Selecting all-time: first, last, and evenly spaced in between ({revision_count} total)...")
    commits = get_commit_history(
        repo_path,
        branch,
        revision_count,
        intelligent=intelligent,
        min_months_apart=min_months_apart,
        min_commits_apart=min_commits_apart,
        since_date=since_date,
        until_date=until_date,
        spacing_mode=spacing_mode,
    )
    print(f"Found {len(commits)} commits to analyze\n")

    if len(commits) == 0:
        raise RuntimeError(f"No commits found on branch '{branch}'. Try a different branch (e.g., 'trunk', 'master').")
    if len(commits) < revision_count:
        print(f"Warning: Only {len(commits)} commits available (requested {revision_count})\n")

    # Create workspace for revisions with descriptive folder name
    # MODE 1: alltime => temporal_analysis_alltime
    # MODE 2: recent with X months => temporal_analysis_5revisions_3month_diff
    # MODE 3: recent with N commit gaps => temporal_analysis_5revisions_100commits_diff
    repo_folder = workspace / (f"{repo_name}_{analysis_tag}" if analysis_tag else repo_name)

    # Derive a simple tag from inputs to name the folder consistently regardless of CLI spacing_mode value
    mode_tag = (
        "commits" if min_commits_apart > 0 else
        ("recent" if min_months_apart > 0 else "alltime")
    )

    # Compute date range (YYYY-MM) from selected commits (min..max month)
    def _month_from(s: str) -> str:
        try:
            # Expect formats like 'YYYY-MM-DD ...'
            return s.strip().split()[0][:7]
        except Exception:
            return "unknown"
    months = [_month_from(c['date']) for c in commits]
    months = [m for m in months if m and m != 'unknown']
    start_month = min(months) if months else "unknown"
    end_month = max(months) if months else "unknown"
    range_suffix = f"_{start_month}_to_{end_month}" if start_month != "unknown" and end_month != "unknown" else ""

    # Add timestamp to distinguish multiple runs
    from datetime import datetime
    run_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    if mode_tag == "alltime":
        folder_name = f"temporal_analysis_alltime{range_suffix}_{run_timestamp}"
    elif mode_tag == "recent":
        folder_name = f"temporal_analysis_{revision_count}revisions_{min_months_apart}month_diff{range_suffix}_{run_timestamp}"
    elif mode_tag == "commits":
        folder_name = f"temporal_analysis_{revision_count}revisions_{min_commits_apart}commits_diff{range_suffix}_{run_timestamp}"
    else:
        folder_name = f"temporal_analysis_{revision_count}revisions{range_suffix}_{run_timestamp}"

    if scope and scope != "full":
        folder_name = f"{folder_name}_scope-{scope}"

    revisions_workspace = repo_folder / folder_name
    ensure_dirs(revisions_workspace)
    # All numbered repo snapshots go into data_repositories/ to keep the analysis root clean
    data_repos_dir = revisions_workspace / "data_repositories"
    ensure_dirs(data_repos_dir)

    # Analyze each revision
    timeseries_data = []
    arch_report_enabled = not skip_arch_report
    revision_counter = 0  # increments only for non-empty revisions → contiguous revision_numbers

    def _has_arch_summary(out_dir: Path) -> bool:
        direct = out_dir / "dv8-analysis-result" / "analysis-summary.html"
        if direct.exists():
            return True
        found = list(out_dir.glob("**/dv8-analysis-result/analysis-summary.html"))
        return bool(found)

    for i, commit in enumerate(commits, 1):
        print(f"\n[{i}/{len(commits)}] Commit {commit['short_hash']}: {commit['message'][:60]}")
        print(f"  Date: {commit['date']}")
        print(f"  Author: {commit['author']}")

        # Checkout (prefer worktree to retain .git for arch-report/hotspot)
        if use_worktree:
            try:
                revision_path = checkout_commit_to_worktree(
                    repo_path, commit['hash'], commit['date'], data_repos_dir, i
                )
            except Exception as e:
                print(f"  Worktree failed ({e}); falling back to archive snapshot.")
                revision_path = checkout_commit_to_folder(repo_path, commit['hash'], commit['date'], data_repos_dir, i)
        else:
            revision_path = checkout_commit_to_folder(repo_path, commit['hash'], commit['date'], data_repos_dir, i)

        # Analyze
        metrics = analyze_single_revision(
            revision_path,
            dv8_console,
            env,
            source_path,
            fine_grain=fine_grain,
            skip_arch_report=not arch_report_enabled,
            scope=scope,
            language=language,
            neodepends_root=neodepends_root,
            neodepends_bin=neodepends_bin,
            neodepends_resolver=neodepends_resolver,
            java_depends=java_depends,
            depends_runner=depends_runner,
            understand_und=understand_und,
            understand_upython=understand_upython,
            understand_language=understand_language,
            understand_granularity=understand_granularity,
            mv_cochange=mv_cochange,
        )

        if fine_grain and arch_report_enabled:
            out_dir = revision_path / "OutputData"
            if not _has_arch_summary(out_dir):
                print("  Arch-report output missing; disabling arch-report for remaining revisions.")
                arch_report_enabled = False

        # Skip empty/trivial revisions (e.g. cvs2svn init commit, empty repo state)
        if _is_empty_revision(metrics):
            print(f"  ⚠  Skipping commit {commit['short_hash']} ({commit['date'][:10]}): "
                  f"all metrics are None/zero — likely an empty or trivial repository state.")
            continue

        # Store results — revision_number is contiguous across non-empty revisions only
        revision_counter += 1
        timeseries_data.append({
            'revision_number': revision_counter,
            'commit_hash': commit['hash'],
            'commit_date': commit['date'],
            'commit_author': commit['author'],
            'commit_message': commit['message'],
            'metrics': metrics
        })

        print(f"  Metrics:")
        for k, v in metrics.items():
            if k != "error":
                print(f"    {k}: {v}")

    # Save time-series data into INPUT_INTERPRETATION for clean folder structure
    interp_input_dir = revisions_workspace / "INPUT_INTERPRETATION"
    ensure_dirs(interp_input_dir)
    output_file = interp_input_dir / "timeseries.json"
    with open(output_file, 'w') as f:
        json.dump({
            'repo': repo_name,
            'timestamp': datetime.now().isoformat(),
            'mode': mode_tag,
            'scope': scope or "full",
            'start_month': start_month,
            'end_month': end_month,
            'revision_count': revision_count,
            'min_months_apart': min_months_apart,
            'min_commits_apart': min_commits_apart,
            'revisions': timeseries_data
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"  Analyzed: {len(timeseries_data)} revisions")
    print(f"  Output: {output_file}")
    print(f"{'='*60}\n")

    # Generate plots
    print(f"\nGenerating plots...")
    try:
        plots_dir = generate_plots(output_file, revisions_workspace)
        print(f"Plots generated: {plots_dir}")
    except Exception as e:
        print(f"Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
        print(f"    Install matplotlib if needed: pip3 install matplotlib")

    return output_file


# --- main --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    # define CLI interface
    parser = argparse.ArgumentParser(description="Automate Depends + DV8 console pipeline.")
    parser.add_argument("--repo", help="Git URL, local path, or ZIP to analyze.")
    parser.add_argument(
        "--branch",
        default=None,
        help="Branch or tag when cloning (default: remote default).",
    )
    # Default workspace is REPOS folder (sibling to 01_STAGE_ANALYZE)
    script_parent = Path(__file__).resolve().parent
    default_ws = str(script_parent.parent / "REPOS_ANALYZED")
    parser.add_argument("--workspace", default=default_ws, help=f"Working directory for clones (default: {default_ws}).")
    parser.add_argument("--project-name", help="Override project name shown in reports.")
    parser.add_argument("--source-path", help="Optional path (relative or absolute) to feed into Depends.")
    parser.add_argument(
        "--scope",
        choices=["full", "prod", "both"],
        default="full",
        help="Analysis scope: full (default, includes tests/tools), prod (deterministic prod-only heuristics), or both (run full+prod).",
    )
    parser.add_argument("--depends-jar", help="Path to depends.jar (optional; used when --depends-runner=jar or fallback).")
    parser.add_argument("--dv8-console", help="Path to dv8-console executable (optional).")
    parser.add_argument("--dv8-license-key", help="DV8 license key (optional; otherwise prompt or reuse cached).")
    parser.add_argument("--dv8-activation-code", help="DV8 activation code (optional; otherwise prompt or reuse cached).")
    parser.add_argument("--ask", help="Metric query: one of 'propagation cost', 'decoupling-level', 'independence-level', 'm-score', or 'all'.")
    parser.add_argument("--skip-arch-report", action="store_true", help="Skip arch-report step (useful for metrics only).")
    parser.add_argument("--force-depends", action="store_true", help="Regenerate Depends output even if cached files exist.")
    parser.add_argument(
        "--depends-runner",
        choices=["dv8", "jar", "understand", "auto"],
        default="dv8",
        help="How to run dependency extraction: dv8 (built-in), jar (external depends.jar), understand (SciTools), or auto (try dv8 then jar).",
    )
    parser.add_argument("--language", help="Override language detection: java, python, csharp, typescript, mixed-cs-ts (default: auto).")
    parser.add_argument("--understand-und", help="Path to SciTools Understand und executable.")
    parser.add_argument("--understand-upython", help="Path to SciTools Understand upython executable.")
    parser.add_argument(
        "--understand-language",
        help="Understand language name, e.g. Java, Python, C++, JavaScript, TypeScript, C#.",
    )
    parser.add_argument(
        "--understand-granularity",
        choices=["entity", "file"],
        default="entity",
        help="Understand export granularity. Use entity for Depends-like full dependencies; file for file-only DSMs.",
    )
    parser.add_argument("--neodepends-root", help="Path to NeoDepends root (release or repo; contains neodepends + tools/).")
    parser.add_argument("--neodepends-bin", help="Path to NeoDepends binary (optional).")
    parser.add_argument(
        "--neodepends-resolver",
        choices=["depends", "stackgraphs"],
        default="stackgraphs",
        help="NeoDepends resolver for Python (default: stackgraphs).",
    )
    parser.add_argument(
        "--java-depends",
        action="store_true",
        help="Force Java dependency extraction via DV8 Depends instead of NeoDepends.",
    )
    parser.add_argument(
        "--analysis-tag",
        help="Optional tag to suffix temporal output folder (useful for multi-language runs).",
    )
    parser.add_argument("--examples", action="store_true", help="Show usage examples and exit.")
    # Temporal analysis options
    parser.add_argument("--temporal", action="store_true", help="Run temporal analysis on multiple Git revisions.")
    parser.add_argument("--revisions", type=int, default=10, help="Number of revisions to analyze in temporal mode (default: 10).")
    parser.add_argument("--intelligent-selection", action="store_true", help="In temporal mode, intelligently select meaningful commits (releases, major changes) instead of just the last N.")
    parser.add_argument("--min-months-apart", type=int, default=0, help="Minimum months between revisions (0=disabled, 1=1 month, 3=3 months recommended for architectural changes).")
    parser.add_argument("--min-commits-apart", type=int, default=0, help="Minimum number of commits between selected revisions (0=disabled). Example: 100 means pick HEAD, then ~100 commits back, etc.")
    parser.add_argument("--since-date", help="Limit analysis to commits on/after this date (YYYY or YYYY-MM or YYYY-MM-DD).")
    parser.add_argument("--until-date", help="Limit analysis to commits on/before this date (YYYY or YYYY-MM or YYYY-MM-DD).")
    parser.add_argument("--no-fine-grain", action="store_true", help="Skip full arch-report (anti-patterns, clustering, hotspots) per revision. Fine-grain is ON by default.")
    parser.add_argument("--mv-cochange", type=int, default=None, help="Modularity Violation co-change threshold (passed to DV8 arch-issue:arch-issue as -mvCochange). Default: DV8's built-in default (2). Use higher values (e.g. 5) to filter out process-noise co-changes (license headers, formatting) and keep only semantically meaningful co-changes.")
    parser.add_argument("--spacing-mode", choices=["intelligent", "interpolate", "all-time-major", "smart", "uniform"], default="intelligent",
                       help="Revision selection mode: smart (most file changes), uniform/alltime (evenly spaced), intelligent (default filtering).")
    parser.add_argument("--no-temporal-worktree", action="store_true",
                       help="Disable git worktrees in temporal mode (use git archive snapshots instead).")
    # Single-commit focus analysis options
    parser.add_argument("--commit", help="Analyze a specific commit hash with full pipeline (use with --fine-grain for arch-report).")
    parser.add_argument("--commit2", help="Analyze a second commit hash (analyze both, placed into a focus folder).")
    return parser.parse_args()

def main() -> None:
    # high-level orchestration: prep -> repo -> depends -> DSM -> report -> metrics
    args = parse_args()
    # ...resolve dv8_console/depends_jar and obtain repository before determining source_root...

    if args.examples:             # print sample commands if requested
        print(EXAMPLE_USAGE.strip())
        return

    # If user didn’t pass --repo, ask interactively
    if not args.repo:
        args.repo = input("Repository path or URL: ").strip()
    if not args.repo:
        raise SystemExit("Repository path or URL is required.")

    # workspace is where we clone/extract the repo; defaults to this script’s folder
    workspace = Path(args.workspace).expanduser().resolve()
    ensure_dirs(workspace)
    ensure_dirs(CONFIG_HOME, DOWNLOADS_DIR)

    # locate dv8-console; optional depends.jar only when needed
    dv8_console = resolve_dv8_console(args.dv8_console)
    depends_jar: Optional[Path] = None
    if args.depends_runner in {"jar", "auto"}:
        try:
            depends_jar = resolve_depends_jar(args.depends_jar)
        except SystemExit as e:
            if args.depends_runner == "jar":
                raise     # jar-only mode cannot proceed
            depends_jar = None  # auto mode: keep going with dv8 runner

    # seed environment so dv8-console is on PATH
    env = os.environ.copy()
    env["PATH"] = f"{dv8_console.parent}{os.pathsep}{env.get('PATH', '')}" if env.get("PATH") else str(dv8_console.parent)
    # keep DV8 logs + temp within a writable path (override with DV8_RUNTIME_DIR if needed)
    #
    # Important: avoid spaces in paths passed via JAVA_TOOL_OPTIONS; the JVM splits it on whitespace.
    runtime_override = os.environ.get("DV8_RUNTIME_DIR")
    if runtime_override:
        dv8_runtime = Path(runtime_override).expanduser().resolve()
    else:
        dv8_runtime = Path(tempfile.gettempdir()) / "dv8_agent_runtime"
    dv8_tmp = dv8_runtime / "tmp"
    try:
        ensure_dirs(dv8_runtime, dv8_tmp)
    except PermissionError:
        # Fallback: last resort to workspace (may contain spaces, so we avoid JAVA_TOOL_OPTIONS below).
        dv8_runtime = workspace / ".dv8_runtime"
        dv8_tmp = dv8_runtime / "tmp"
        ensure_dirs(dv8_runtime, dv8_tmp)
    log4j_config = dv8_runtime / "log4j2.xml"
    if not log4j_config.exists():
        log4j_config.write_text(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<Configuration status=\"WARN\">\n"
            "  <Appenders>\n"
            "    <Console name=\"Console\" target=\"SYSTEM_OUT\">\n"
            "      <PatternLayout pattern=\"%d{HH:mm:ss} %-5p %c{1} - %m%n\"/>\n"
            "    </Console>\n"
            "  </Appenders>\n"
            "  <Loggers>\n"
            "    <Root level=\"info\">\n"
            "      <AppenderRef ref=\"Console\"/>\n"
            "    </Root>\n"
            "  </Loggers>\n"
            "</Configuration>\n",
            encoding="utf-8",
        )
    env["LOG4J_CONFIGURATION_FILE"] = str(log4j_config)
    env["TMPDIR"] = str(dv8_tmp)
    env["TMP"] = str(dv8_tmp)
    env["TEMP"] = str(dv8_tmp)
    env["JNA_TMPDIR"] = str(dv8_tmp)
    # Only set JAVA_TOOL_OPTIONS if the tmp path has no spaces.
    if " " not in str(dv8_tmp):
        jvm_opts = env.get("JAVA_TOOL_OPTIONS", "")
        extra_opts = f"-Xmx4g -Djava.io.tmpdir={dv8_tmp} -Djna.tmpdir={dv8_tmp}"
        env["JAVA_TOOL_OPTIONS"] = f"{jvm_opts} {extra_opts}".strip()

    # make sure DV8 license is active (prompts if needed)
    ensure_dv8_license(dv8_console, env, args.dv8_license_key, args.dv8_activation_code)

    # obtain the repo folder (clone, unzip, or use local path). If missing, prompt user.
    while True:
        try:
            repo_root = clone_or_copy(args.repo, workspace, args.branch).resolve()
            break
        except SystemExit as e:
            msg = str(e)
            if "not found" in msg.lower():
                print(msg)
                new_repo = input("Enter a Git URL, local folder, or ZIP path (Enter for current folder, or 'q' to quit): ").strip()
                if not new_repo:
                    new_repo = "."
                if new_repo.lower() in {"q", "quit", "exit"}:
                    raise
                args.repo = new_repo
                continue
            else:
                raise

    # friendly name for reports if not provided
    project_name = args.project_name or guess_project_name(repo_root)

    # Focus analysis on explicit commits (no temporal series)
    if args.commit:
        def _git_date(repo: Path, h: str) -> str:
            try:
                r = subprocess.run(["git", "show", "-s", "--format=%ci", h], cwd=repo, capture_output=True, text=True, check=True)
                return r.stdout.strip() or ""
            except Exception:
                return ""

        short1 = args.commit[:8]
        short2 = args.commit2[:8] if args.commit2 else None
        focus_name = f"focus_commits_{short1}" + (f"_to_{short2}" if short2 else "")
        focus_ws = (Path(args.workspace).expanduser().resolve() / repo_root.name / focus_name)
        ensure_dirs(focus_ws)

        commits = [args.commit] + ([args.commit2] if args.commit2 else [])
        for i, h in enumerate(commits, 1):
            date = _git_date(repo_root, h)
            rev_dir = checkout_commit_to_folder(repo_root, h, date, focus_ws, i)
            _ = analyze_single_revision(
                rev_dir,
                dv8_console,
                env,
                args.source_path,
                fine_grain=not args.no_fine_grain,
                scope=args.scope,
                language=args.language,
                neodepends_root=args.neodepends_root,
                neodepends_bin=args.neodepends_bin,
                neodepends_resolver=args.neodepends_resolver,
                java_depends=args.java_depends,
                depends_runner=args.depends_runner,
                understand_und=args.understand_und,
                understand_upython=args.understand_upython,
                understand_language=args.understand_language,
                understand_granularity=args.understand_granularity,
                mv_cochange=args.mv_cochange if hasattr(args, "mv_cochange") else None,
            )

        print(f"\nFocus commit analysis complete: {focus_ws}")
        return

    # If temporal analysis is requested, run it and exit
    if args.temporal:
        branch = args.branch or "main"
        auto_fine = not args.no_fine_grain  # fine-grain (arch-report per revision) is ON by default; --no-fine-grain disables it
        scopes = ["full", "prod"] if args.scope == "both" else [args.scope]
        last_output: Optional[Path] = None
        for sc in scopes:
            last_output = run_temporal_analysis(
                repo_path=repo_root,
                revision_count=args.revisions,
                branch=branch,
                workspace=workspace,
                source_path=args.source_path,
                dv8_console=dv8_console,
                env=env,
                scope=sc,
                language=args.language,
                neodepends_root=args.neodepends_root,
                neodepends_bin=args.neodepends_bin,
                neodepends_resolver=args.neodepends_resolver,
                java_depends=args.java_depends,
                depends_runner=args.depends_runner,
                understand_und=args.understand_und,
                understand_upython=args.understand_upython,
                understand_language=args.understand_language,
                understand_granularity=args.understand_granularity,
                analysis_tag=args.analysis_tag,
                intelligent=args.intelligent_selection,
                min_months_apart=args.min_months_apart,
                min_commits_apart=args.min_commits_apart,
                since_date=args.since_date,
                until_date=args.until_date,
                fine_grain=auto_fine,
                skip_arch_report=args.skip_arch_report,
                use_worktree=not args.no_temporal_worktree,
                spacing_mode=args.spacing_mode,
                mv_cochange=args.mv_cochange if hasattr(args, "mv_cochange") else None,
            )

        print(f"\nTemporal analysis complete!")
        if last_output:
            print(f"   Time-series data: {last_output}")
        print(f"   Revision folders: {workspace / repo_root.name}/")

        # Clean up cloned repo source files — keep only .git and temporal_analysis_* folders.
        # The .git dir is kept so subsequent runs can skip re-cloning.
        repo_folder = workspace / repo_root.name
        if repo_folder.exists():
            removed = []
            for item in list(repo_folder.iterdir()):
                if item.name == ".git" or item.name.startswith("temporal_analysis"):
                    continue
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    removed.append(item.name)
                except Exception as e:
                    print(f"  Warning: could not remove {item.name}: {e}")
            if removed:
                print(f"  Cleaned up source files: {', '.join(removed)}")
        return

    # try to detect a sensible source root automatically (Maven/Gradle/common)
    source_root = None
    if args.source_path:
        p = Path(args.source_path).expanduser()
        if not p.is_absolute():
            p = repo_root / p
        if p.exists():
            source_root = p
    if source_root is None:
        for rel in ("src/main/java", "src", "SourceCode", "InputData/SourceCode"):
            cand = repo_root / rel
            if cand.exists():
                source_root = cand
                break
    if source_root is None:
        source_root = repo_root  # last resort: entire repo
    print(f"Using source root: {source_root}")

    analysis_root = analysis_root_for_single_run(repo_root, workspace).resolve()
    if analysis_root != repo_root:
        print(f"Writing analysis artifacts to: {analysis_root}")

    # ensure DV8’s expected folder structure inside the analysis root
    input_data = analysis_root / "InputData"
    depends_output = input_data / "DependsOutput" / "json"
    neodepends_output = input_data / "NeoDependsOutput"
    output_data = analysis_root / "OutputData"
    ensure_dirs(input_data, depends_output, output_data, neodepends_output)

    # prepare file names for depends outputs
    basename = "repo-json-dep"
    default_json = depends_output / f"{basename}.json"
    json_candidates = sorted(depends_output.glob(f"{basename}*.json"))
    json_dep = json_candidates[0] if json_candidates else default_json
    mapping: _Optional[Path] = (depends_output / "depends-dv8map.mapping") if (depends_output / "depends-dv8map.mapping").is_file() else None

    # Auto-detect language for dependency extraction
    detected_language = detect_language(source_root, args.language)
    if detected_language == "python":
        adjusted = auto_adjust_python_root(source_root)
        if adjusted != source_root:
            print(f"Auto-adjusted Python source root: {adjusted}")
            source_root = adjusted
    # Auto-select Understand for C#/TypeScript (no other extractor supports them)
    effective_depends_runner = args.depends_runner
    if effective_depends_runner != "understand" and detected_language in ("csharp", "typescript", "mixed-cs-ts"):
        effective_depends_runner = "understand"
        print(f"[deps] Auto-selecting Understand for {detected_language} source.")
    use_understand = effective_depends_runner == "understand"
    use_neodepends = (
        detected_language in {"python", "java"}
        and not use_understand
        and not (detected_language == "java" and args.java_depends)
    )
    if use_understand:
        print(
            "Using SciTools Understand for dependency extraction "
            f"({understand_language_name(detected_language, args.understand_language)})."
        )
    if use_neodepends:
        print(f"Detected {detected_language} source. Using NeoDepends for dependency extraction.")

    # check for existing analysis and branch
    existing_report = find_existing_report_root(analysis_root)
    if existing_report and not args.force_depends and not args.skip_arch_report:
        print(f"Found existing DV8 analysis: {existing_report}")
        ans = input("Re-run full analysis? [y/N]: ").strip().lower()
        if ans not in {"y", "yes"}:
            ask = args.ask
            if not ask:
                ask = input("Return a specific metric (m-score/propagation cost/decoupling-level/independence-level) or 'all': ").strip() or "all"
            result = fetch_metric(existing_report, ask)
            if result is None:
                # Try computing metrics directly from DSM if present
                dsm_path = output_data / "repo.dv8-dsm"
                if dsm_path.is_file():
                    try:
                        metric_file = run_metric_task(dv8_console, ask, dsm_path, output_data, env)
                        try:
                            result = json.loads(Path(metric_file).read_text())
                        except json.JSONDecodeError:
                            result = {"file": str(metric_file)}
                    except Exception as e:
                        print(f"Metric computation failed: {e}")
                        result = None
            print("\n=== Metric result ===")
            if result is None:
                print(f"No metric data available for query: {ask}")
            else:
                print(json.dumps(result, indent=2))
            return

    # reuse cached outputs unless user forced regeneration
    needs_depends_fallback = False
    if use_understand:
        understand_output = input_data / "UnderstandOutput"
        json_dep = run_understand_export(
            source_root=source_root,
            output_dir=understand_output,
            project_name=project_name,
            language=understand_language_name(detected_language, args.understand_language),
            und_bin=args.understand_und,
            upython_bin=args.understand_upython,
            granularity=args.understand_granularity,
            force=args.force_depends,
        )
        mapping = None
    elif use_neodepends:
        neodep_json = neodepends_output / "dependencies.full.dv8-dependency.json"
        if neodep_json.is_file() and not args.force_depends:
            print("Reusing existing NeoDepends output (use --force-depends to regenerate).")
            json_dep = neodep_json
            mapping = None
        else:
            nd_root = resolve_neodepends_root(args.neodepends_root)
            if nd_root:
                json_dep = run_neodepends_python_export(
                    source_root=source_root,
                    output_dir=neodepends_output,
                    neodepends_root=nd_root,
                    neodepends_bin=args.neodepends_bin,
                    resolver=args.neodepends_resolver,
                    config="default",
                    langs=detected_language,
                )
                mapping = None
            elif detected_language == "java":
                print("NeoDepends not available for Java. Falling back to DV8 Depends.")
                needs_depends_fallback = True
            else:
                raise SystemExit("NeoDepends root not found. Set --neodepends-root or NEODEPENDS_ROOT.")
    if (not use_neodepends) or needs_depends_fallback:
        cache_available = json_dep.is_file()
        if cache_available and not args.force_depends:
            print("Reusing existing Depends output (use --force-depends to regenerate).")
        else:
            # Choose how to run dependency extraction
            runner = args.depends_runner
            last_err: Optional[Exception] = None
            if runner in {"dv8", "auto"}:
                try:
                    json_dep, mapping = run_depends_via_dv8(dv8_console, source_root, depends_output, basename, env)
                except Exception as e:
                    last_err = e
                    if runner == "dv8":
                        raise
            # If dv8 path didn't produce a JSON, try the jar fallback (if allowed and available)
            if runner in {"jar", "auto"} and not any(depends_output.glob(f"{basename}*.json")):
                if depends_jar is None:
                    if last_err:
                        print(f"DV8 depends:parser failed: {last_err}")
                    raise SystemExit("depends.jar is required for fallback but was not found. Set --depends-jar or DEPENDS_HOME.")
                json_dep, mapping = run_depends_via_jar(depends_jar, source_root, depends_output, basename)
            # handle dv8 emitting a suffixed JSON name by reselecting the first match
            if not Path(json_dep).is_file():
                candidates = sorted(depends_output.glob(f"{basename}*.json"))
                if candidates:
                    json_dep = candidates[0]

    # convert the depends JSON (+optional mapping) into a DV8 DSM file
    dsm_path = output_data / "repo.dv8-dsm"
    convert_to_dsm(dv8_console, json_dep, mapping, dsm_path, env)

    # write an arch-report properties file inside InputData
    params_path = input_data / "archreport.properties"
    output_dir = Path("OutputData/Architecture-analysis-result")  # relative path used by DV8
    write_params(params_path, project_name, dsm_path.relative_to(analysis_root), output_dir)

    # If the user wants all metrics, compute them directly from the DSM first (independent of report)
    ask_key = (args.ask or "").strip().lower()
    if ask_key in {"all", "metrics", "all metrics", "all-metrics"}:
        all_metrics = compute_all_metrics(dv8_console, dsm_path, output_data, env)
        print("\n=== All Metrics ===")
        print(json.dumps(all_metrics, indent=2))

    # run the full architecture analysis if not skipped
    if not args.skip_arch_report:
        try:
            run_arch_report(dv8_console, params_path.relative_to(analysis_root), analysis_root, env)
        except SystemExit as e:
            print(f"Arch report failed: {e}")
            # If user only asked for a metric, try computing it directly from the DSM
            if args.ask and ask_key not in {"all", "metrics", "all metrics", "all-metrics"}:
                try:
                    metric_file = run_metric_task(dv8_console, args.ask, dsm_path, output_data, env)
                    try:
                        result = json.loads(Path(metric_file).read_text())
                    except json.JSONDecodeError:
                        result = {"file": str(metric_file)}
                    print("\n=== Metric result (fallback) ===")
                    print(json.dumps(result, indent=2))
                    return
                except Exception as me:
                    print(f"Direct metric fallback also failed: {me}")
            # No metric requested; just stop
            return

    # If a metric was requested, locate it in the report outputs or compute directly
    if args.ask:
        if ask_key in {"all", "metrics", "all metrics", "all-metrics"}:
            # already printed combined metrics above
            pass
        else:
            answer = fetch_metric(analysis_root / output_dir, args.ask)
            if answer is None:
                # Try direct metric even if report ran but didn't emit expected files
                try:
                    metric_file = run_metric_task(dv8_console, args.ask, dsm_path, output_data, env)
                    try:
                        answer = json.loads(Path(metric_file).read_text())
                    except json.JSONDecodeError:
                        answer = {"file": str(metric_file)}
                except Exception:
                    answer = None
            print("\n=== Metric result ===")
            if answer is None:
                print(f"No metric data available for query: {args.ask}")
            else:
                print(json.dumps(answer, indent=2))

if __name__ == "__main__":  # standard Python module guard
    main()  # run the orchestrator
