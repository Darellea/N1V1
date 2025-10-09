"""Utilities for efficient code auditing - incremental analysis and caching"""

import hashlib
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import git


class AuditCache:
    """Cache system for audit results with file modification time tracking"""

    def __init__(self, cache_dir: str = ".audit_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file contents and modification time"""
        try:
            stat = os.stat(file_path)
            mtime = stat.st_mtime
            size = stat.st_size

            # Include file path, mtime, and size in hash
            hash_input = f"{file_path}:{mtime}:{size}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except OSError:
            return ""

    def _get_cache_path(self, tool: str, file_path: str) -> Path:
        """Get cache file path for a tool and file"""
        file_hash = self._get_file_hash(file_path)
        if not file_hash:
            return self.cache_dir / f"{tool}_invalid.cache"

        return self.cache_dir / f"{tool}_{file_hash[:8]}.cache"

    def get_cached_result(self, tool: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached result if still valid"""
        cache_path = self._get_cache_path(tool, file_path)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)

            # Verify the cached result is still valid
            if self._get_file_hash(file_path) == cached_data.get("file_hash"):
                return cached_data.get("result")

        except (pickle.UnpicklingError, KeyError, EOFError):
            # Cache corrupted, remove it
            cache_path.unlink(missing_ok=True)

        return None

    def cache_result(self, tool: str, file_path: str, result: Dict[str, Any]) -> None:
        """Cache a result"""
        cache_path = self._get_cache_path(tool, file_path)

        cached_data = {
            "file_hash": self._get_file_hash(file_path),
            "result": result,
            "timestamp": os.path.getmtime(file_path)
            if os.path.exists(file_path)
            else 0,
        }

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cached_data, f)
        except (OSError, pickle.PicklingError):
            # If caching fails, just continue
            pass

    def clear_cache(self) -> None:
        """Clear all cached results"""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink(missing_ok=True)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.cache"))
        return {
            "total_cached_results": len(cache_files),
            "cache_size_mb": sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
        }


class GitUtils:
    """Git utilities for incremental analysis"""

    def __init__(self, repo_path: str = "."):
        try:
            self.repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError:
            self.repo = None

    def get_changed_files(
        self, base_branch: str = "origin/main", include_untracked: bool = True
    ) -> Set[str]:
        """Get files changed since base branch"""
        if not self.repo:
            return set()

        try:
            # Get the base commit
            base_commit = self.repo.commit(base_branch)

            # Get current HEAD
            head_commit = self.repo.head.commit

            # Get diff between base and HEAD
            diff = base_commit.diff(head_commit)

            changed_files = set()

            # Files modified in diff
            for item in diff:
                if item.a_path:
                    changed_files.add(item.a_path)
                if item.b_path and item.b_path != item.a_path:
                    changed_files.add(item.b_path)

            # Also check for files that are different from working directory
            if include_untracked:
                # Get untracked files
                untracked = set(self.repo.untracked_files)

                # Get modified files in working directory
                modified = set()
                for item in self.repo.index.diff("HEAD"):
                    modified.add(item.a_path)

                changed_files.update(untracked)
                changed_files.update(modified)

            # Filter to Python files only
            python_files = {f for f in changed_files if f.endswith(".py")}

            return python_files

        except (git.GitCommandError, ValueError):
            # Fallback: return empty set if git operations fail
            return set()

    def get_python_files_in_repo(self) -> Set[str]:
        """Get all Python files in the repository"""
        if not self.repo:
            return set()

        python_files = set()

        # Walk through all files in repo
        for root, dirs, files in os.walk(self.repo.working_dir):
            # Skip .git directory and common exclusions
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".") and d not in ["__pycache__", "node_modules"]
            ]

            for file in files:
                if file.endswith(".py"):
                    # Get relative path from repo root
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.repo.working_dir)
                    python_files.add(rel_path)

        return python_files


class ParallelProcessor:
    """Parallel processing utilities for file-level operations"""

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, os.cpu_count() or 4)

    def process_files_parallel(
        self, files: List[str], processor_func, *args, **kwargs
    ) -> Dict[str, Any]:
        """Process files in parallel using the given function"""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(processor_func, file_path, *args, **kwargs): file_path
                for file_path in files
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results[file_path] = result
                except Exception as exc:
                    results[file_path] = {"error": str(exc)}

        return results


class IncrementalAnalyzer:
    """Incremental analysis with caching and parallel processing"""

    def __init__(self, cache_dir: str = ".audit_cache"):
        self.cache = AuditCache(cache_dir)
        self.git_utils = GitUtils()
        self.parallel_processor = ParallelProcessor()

    def analyze_files_incremental(
        self, files: List[str], analysis_func, tool_name: str, use_cache: bool = True
    ) -> Dict[str, Any]:
        """Analyze files incrementally with caching"""
        results = {}
        files_to_analyze = []
        cached_results = {}

        if use_cache:
            # Check cache for each file
            for file_path in files:
                cached_result = self.cache.get_cached_result(tool_name, file_path)
                if cached_result is not None:
                    cached_results[file_path] = cached_result
                else:
                    files_to_analyze.append(file_path)
        else:
            files_to_analyze = files

        # Analyze files that aren't cached or cache disabled
        if files_to_analyze:
            print(f"Analyzing {len(files_to_analyze)} files with {tool_name}...")
            analyzed_results = self.parallel_processor.process_files_parallel(
                files_to_analyze, analysis_func
            )

            # Cache the results
            for file_path, result in analyzed_results.items():
                if use_cache and "error" not in result:
                    self.cache.cache_result(tool_name, file_path, result)
                results[file_path] = result
        else:
            print(f"All {len(files)} files already cached for {tool_name}")

        # Merge cached and new results
        results.update(cached_results)

        return results

    def run_incremental_audit(
        self, base_branch: str = "origin/main", mode: str = "incremental"
    ) -> Dict[str, Any]:
        """Run incremental audit based on changed files"""
        if mode == "full":
            # Full analysis - analyze all Python files
            target_files = list(self.git_utils.get_python_files_in_repo())
            print(f"Running FULL audit on {len(target_files)} Python files")
        else:
            # Incremental analysis - only changed files
            target_files = list(self.git_utils.get_changed_files(base_branch))
            print(
                f"Running INCREMENTAL audit on {len(target_files)} changed Python files"
            )

        if not target_files:
            print("No Python files to analyze")
            return {"files_analyzed": 0, "results": {}}

        # For demonstration, we'll analyze complexity on changed files
        # In a full implementation, this would integrate with the main analysis pipeline
        from .static_analysis import StaticAnalysis

        analyzer = StaticAnalysis()

        # Run radon cc on changed files only
        print(f"Running complexity analysis on {len(target_files)} files...")

        # Filter to existing files
        existing_files = [f for f in target_files if os.path.exists(f)]

        if existing_files:
            # For incremental mode, analyze individual files
            complexity_results = self.analyze_files_incremental(
                existing_files,
                lambda f: self._analyze_file_complexity(f, analyzer),
                "radon_cc",
            )

            return {
                "mode": mode,
                "base_branch": base_branch,
                "files_analyzed": len(existing_files),
                "results": {"complexity": complexity_results},
            }
        else:
            return {
                "mode": mode,
                "base_branch": base_branch,
                "files_analyzed": 0,
                "results": {},
            }

    def _analyze_file_complexity(
        self, file_path: str, analyzer: "StaticAnalysis"
    ) -> Dict[str, Any]:
        """Analyze complexity of a single file"""
        try:
            # Run radon cc on single file
            returncode, stdout, stderr = analyzer.run_command(
                ["radon", "cc", "-s", file_path, "-j"]
            )

            if returncode == 0:
                data = json.loads(stdout)
                return {
                    "status": "success",
                    "complexity_data": data,
                    "functions": len(data.get(file_path, [])),
                }
            else:
                return {"status": "error", "error": stderr or "Analysis failed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit Utilities - Incremental analysis tools"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="incremental",
        help="Analysis mode",
    )
    parser.add_argument(
        "--base", default="origin/main", help="Base branch for incremental analysis"
    )
    parser.add_argument("--clear-cache", action="store_true", help="Clear audit cache")
    parser.add_argument(
        "--cache-stats", action="store_true", help="Show cache statistics"
    )

    args = parser.parse_args()

    analyzer = IncrementalAnalyzer()

    if args.clear_cache:
        analyzer.cache.clear_cache()
        print("Cache cleared")
        return

    if args.cache_stats:
        stats = analyzer.cache.get_cache_stats()
        print(
            f"Cache stats: {stats['total_cached_results']} cached results, "
            f"{stats['cache_size_mb']:.2f} MB"
        )
        return

    results = analyzer.run_incremental_audit(args.base, args.mode)

    print(
        f"Analysis complete: {results['files_analyzed']} files analyzed in {args.mode} mode"
    )


if __name__ == "__main__":
    main()
