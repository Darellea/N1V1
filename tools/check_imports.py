#!/usr/bin/env python3
"""
Static import checker for the repository.

Scans all .py files, parses import statements, and reports imports that
look like they reference local project modules but whose target files
do not exist on disk.

This is a static check only (uses AST) and does not execute project code.
"""
import ast
import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__) + os.sep + "..")
ROOT = os.path.normpath(ROOT)

def find_py_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # skip hidden dirs like .git, __pycache__
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
        for f in filenames:
            if f.endswith('.py'):
                yield os.path.join(dirpath, f)

def module_candidates(module):
    """
    Given a dotted module name like "utils.config_loader", return possible filesystem
    paths to check (relative to ROOT), e.g. "utils/config_loader.py" and
    "utils/config_loader/__init__.py".
    """
    parts = module.split('.')
    file_path = os.path.join(*parts) + '.py'
    pkg_init = os.path.join(*parts, '__init__.py')
    return [file_path, pkg_init]

def exists_in_project(relpath):
    path = os.path.normpath(os.path.join(ROOT, relpath))
    return os.path.exists(path)

def resolve_relative(from_file, level, module):
    """
    Resolve a relative import (level >= 1) to one or more candidate paths.
    from_file: absolute path to the file containing the import
    level: integer level (1 = from . import x)
    module: module name or None
    """
    # directory containing the file
    cur_dir = os.path.dirname(from_file)
    # go up 'level' times
    for _ in range(level - 1):
        cur_dir = os.path.dirname(cur_dir)
    base_rel = os.path.relpath(cur_dir, ROOT)
    if module:
        full = os.path.join(base_rel, *module.split('.'))
    else:
        full = base_rel
    # normalize and produce candidates
    parts = os.path.normpath(full).split(os.sep) if full != '.' else []
    if parts == ['.']:
        parts = []
    if parts:
        file_path = os.path.join(*parts) + '.py'
        pkg_init = os.path.join(*parts, '__init__.py')
    else:
        file_path = '__init__.py'
        pkg_init = '__init__.py'
    return [file_path, pkg_init]

def is_local_module(module_name, project_top_level_dirs):
    """
    Heuristic: consider a module local if its top-level name is one of the project dirs
    or if it starts with a dot (handled separately). This avoids flagging stdlib/external packages.
    """
    if not module_name:
        return False
    top = module_name.split('.')[0]
    return top in project_top_level_dirs

def main():
    # identify top-level project package names (directories with Python files)
    project_dirs = []
    for name in os.listdir(ROOT):
        p = os.path.join(ROOT, name)
        if os.path.isdir(p) and not name.startswith('.') and name != '__pycache__':
            # consider it a project package if it contains any .py files
            for f in os.listdir(p):
                if f.endswith('.py'):
                    project_dirs.append(name)
                    break

    # also include top-level .py filenames (modules) as possible locals
    top_level_modules = [os.path.splitext(f)[0] for f in os.listdir(ROOT) if f.endswith('.py')]

    project_top_level = set(project_dirs + top_level_modules)

    missing = []
    scanned = 0
    for py in find_py_files(ROOT):
        scanned += 1
        rel_py = os.path.relpath(py, ROOT)
        try:
            with open(py, 'r', encoding='utf-8') as fh:
                src = fh.read()
            tree = ast.parse(src, filename=py)
        except Exception as e:
            print(f"[WARN] Failed to parse {rel_py}: {e}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name  # e.g. 'utils.config_loader' or 'requests'
                    if is_local_module(name, project_top_level):
                        candidates = module_candidates(name)
                        found = any(exists_in_project(c) for c in candidates)
                        if not found:
                            missing.append((rel_py, 'import', name, candidates))
            elif isinstance(node, ast.ImportFrom):
                module = node.module  # can be None for "from . import x"
                level = node.level  # 0 for absolute imports
                if level and level >= 1:
                    # relative import: resolve
                    candidates = resolve_relative(py, level, module)
                    found = any(exists_in_project(c) for c in candidates)
                    if not found:
                        # represent module as relative (e.g., .types or ..sub.pkg)
                        mod_repr = ('.' * level) + (module or '')
                        missing.append((rel_py, 'from', mod_repr, candidates))
                else:
                    # absolute import
                    if module and is_local_module(module, project_top_level):
                        candidates = module_candidates(module)
                        found = any(exists_in_project(c) for c in candidates)
                        if not found:
                            missing.append((rel_py, 'from', module, candidates))

    # Print results
    print("Checked repository at:", ROOT)
    print("Python files scanned:", scanned)
    if not missing:
        print("\nNo missing local imports detected. (No obvious deletions of files that are imported.)")
        sys.exit(0)
    else:
        print("\nMissing local import targets detected:")
        for rel_py, kind, mod, candidates in missing:
            print(f"- File: {rel_py}")
            print(f"  Statement type: {kind}")
            print(f"  Module: {mod}")
            print(f"  Expected candidate paths (relative to repo root):")
            for c in candidates:
                print(f"    - {c}")
            print("")
        print(f"Total missing import targets: {len(missing)}")
        sys.exit(2)

if __name__ == '__main__':
    main()
