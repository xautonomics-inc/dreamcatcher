"""Check image executable references against CMake declarations, without compiling."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path


class CheckError(ValueError):
    pass


def cmake_commands(text: str) -> list[tuple[str, list[str]]]:
    """Read balanced commands, respecting strings and CMake bracket/line comments."""
    token = re.compile(
        r'#[^\n]*|"(?:\\.|[^"\\])*"|'
        r'\[(=*)\[[\s\S]*?\]\1\]|[()]|[^\s()"#]+'
    )
    # Bracket comments must be removed before tokenizing their contents.
    text = re.sub(r"#\[(=*)\[[\s\S]*?\]\1\]", "", text)
    tokens = [m.group() for m in token.finditer(text) if not m.group().startswith("#")]
    commands: list[tuple[str, list[str]]] = []
    i = 0
    while i + 1 < len(tokens):
        name = tokens[i].lower()
        if tokens[i + 1] != "(":
            i += 1
            continue
        i += 2
        depth = 1
        args: list[str] = []
        while i < len(tokens) and depth:
            value = tokens[i]
            if value == "(":
                depth += 1
            elif value == ")":
                depth -= 1
            if depth:
                args.append(
                    value[1:-1]
                    if value.startswith('"') and value.endswith('"')
                    else value
                )
            i += 1
        if depth:
            raise CheckError("unbalanced CMake command")
        commands.append((name, args))
    return commands


def resolve(value: str, variables: dict[str, str]) -> str:
    for _ in range(8):
        expanded = re.sub(r"\$\{([^}]+)\}", lambda m: variables.get(m[1], m[0]), value)
        if expanded == value:
            break
        value = expanded
    return value


def declared_targets(root: Path) -> set[str]:
    """Find literal executables and locally resolved set(TARGET ... ) declarations."""
    targets: set[str] = set()
    paths = sorted(set(root.rglob("CMakeLists.txt")) | set(root.rglob("*.cmake")))
    for path in paths:
        if any(
            part.startswith(".") or part.startswith("build")
            for part in path.relative_to(root).parts
        ):
            continue
        variables: dict[str, str] = {}
        for name, args in cmake_commands(path.read_text(errors="replace")):
            if name == "set" and len(args) >= 2:
                variables[args[0]] = resolve(args[1], variables)
            elif name == "add_executable" and args:
                value = resolve(args[0], variables)
                if len(args) > 1 and args[1].upper() in ("ALIAS", "IMPORTED"):
                    continue
                if re.fullmatch(r"[A-Za-z0-9_.+-]+", value):
                    targets.add(value)
            elif (
                name == "set_target_properties"
                and "PROPERTIES" in args
                and "OUTPUT_NAME" in args
            ):
                at = args.index("OUTPUT_NAME")
                if at + 1 < len(args) and resolve(args[0], variables) in targets:
                    value = resolve(args[at + 1], variables)
                    if re.fullmatch(r"[A-Za-z0-9_.+-]+", value):
                        targets.add(value)
    return targets


def instructions(path: Path) -> list[tuple[int, str, str]]:
    result: list[tuple[int, str, str]] = []
    joined = ""
    start = 0
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not joined:
            start = number
        joined += line[:-1] + " " if line.endswith("\\") else line
        if line.endswith("\\"):
            continue
        parts = joined.split(None, 1)
        result.append((start, parts[0].upper(), parts[1] if len(parts) > 1 else ""))
        joined = ""
    if joined:
        raise CheckError("unterminated Containerfile continuation")
    return result


def words(value: str) -> list[str]:
    if value.lstrip().startswith("["):
        parsed: object = json.loads(value)
        if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
            raise CheckError("expected JSON array of strings")
        return [str(x) for x in parsed]
    return shlex.split(value, comments=True)


def check(root: Path, paths: list[Path], target: str | None) -> dict[str, object]:
    declared = declared_targets(root)
    scripts = {
        p.name
        for p in root.rglob("*")
        if p.is_file() and p.suffix in (".sh", ".py") and ".git" not in p.parts
    }
    checked: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for path in paths:
        display = path.relative_to(root).as_posix()
        try:
            rows = instructions(path)
            stage = ""
            stages: dict[str, str] = {}
            staged: list[tuple[str, int, str, str]] = []
            for line, op, args in rows:
                if op == "FROM":
                    fields = shlex.split(args)
                    fields = [x for x in fields if not x.startswith("--")]
                    if not fields:
                        raise CheckError("missing FROM image")
                    stage = (
                        fields[-1]
                        if len(fields) >= 3 and fields[-2].upper() == "AS"
                        else str(len(stages))
                    )
                    stages[stage] = fields[0]
                staged.append((stage, line, op, args))
            selected = set(stages)
            if target is not None:
                if target not in stages:
                    raise CheckError("requested stage not found")
                selected = set()
                current = target
                while current in stages and current not in selected:
                    selected.add(current)
                    current = stages[current]
            references: list[tuple[int, str, str]] = []
            for stage, line, op, args in staged:
                if stage not in selected:
                    continue
                if op == "COPY":
                    # Strip Docker flags before parsing either shell or JSON form.
                    from_stage = args.startswith("--") and "--from=" in args
                    args = re.sub(r"^(?:--\S+\s+)+", "", args)
                    fields = words(args)
                    if len(fields) < 2:
                        raise CheckError("COPY requires source and destination")
                    for source in fields[:-1]:
                        name = Path(source).name
                        if any(c in source for c in "$*?["):
                            raise CheckError(
                                "COPY source must use literal artifact paths"
                            )
                        if from_stage and name in ("bin", "full"):
                            raise CheckError(
                                "enumerate binaries instead of copying a directory"
                            )
                        if name.startswith("llama-") and not Path(name).suffix:
                            references.append((line, "COPY", name))
                        elif (
                            ("/bin/" in source or from_stage)
                            and name
                            and name not in ("*", "bin", "lib")
                            and not Path(name).suffix
                        ):
                            references.append((line, "COPY", name))
                elif op == "ENTRYPOINT":
                    fields = words(args)
                    if not fields:
                        continue  # Docker permits clearing an inherited entrypoint.
                    if not args.lstrip().startswith("["):
                        raise CheckError(
                            "shell-form ENTRYPOINT needs exec-form for static checking"
                        )
                    executable = Path(fields[0]).name
                    if executable in ("sh", "bash", "python", "python3"):
                        if len(fields) < 2 or fields[1].startswith("-"):
                            raise CheckError(
                                "interpreter ENTRYPOINT needs a literal script path"
                            )
                        executable = Path(fields[1]).name
                    references.append((line, "ENTRYPOINT", executable))
            if not references:
                raise CheckError(
                    "no executable references; cannot establish image entrypoint"
                )
            for line, source, name in references:
                if any(c in name for c in "$*?["):
                    status = "unresolved_executable"
                elif name in declared:
                    status = "declared_target"
                elif name in scripts and Path(name).suffix in (".sh", ".py"):
                    status = "tracked_script_requires_image_smoke"
                else:
                    status = "missing_build_target"
                row: dict[str, object] = {
                    "file": display,
                    "line": line,
                    "instruction": source,
                    "executable": name,
                    "status": status,
                }
                checked.append(row)
                if status in ("missing_build_target", "unresolved_executable"):
                    errors.append(row)
        except (ValueError, OSError) as exc:
            errors.append(
                {
                    "file": display,
                    "status": "unsupported_or_invalid",
                    "reason": str(exc),
                }
            )
    return {
        "schema_version": 1,
        "ok": bool(paths) and not errors,
        "target": target,
        "declared_target_count": len(declared),
        "checked": checked,
        "errors": errors,
        "scope": "static_declarations_only_not_configure_build_or_runtime_proof",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--target", help="Check this image stage and its FROM ancestors"
    )
    parser.add_argument("containerfiles", nargs="*", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        paths = (
            [p.resolve() for p in args.containerfiles]
            if args.containerfiles
            else sorted((root / "docker").rglob("*.Containerfile"))
        )
        if not paths:
            raise CheckError("no Containerfiles found")
        result = check(root, paths, args.target)
    except (ValueError, OSError) as exc:
        result = {"schema_version": 1, "ok": False, "error": str(exc)}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
