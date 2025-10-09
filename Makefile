# Auditor - Code Quality Automation Makefile

.PHONY: help install audit audit-fast lint lint-fix static deps report clean ci-setup test

# Default target
help:
	@echo "Auditor - Code Quality Automation"
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install development dependencies"
	@echo "  audit        Run full code quality audit"
	@echo "  audit-fast   Run fast audit on auditor module only"
	@echo "  lint         Run code linting checks"
	@echo "  lint-fix     Run code linting and auto-fix"
	@echo "  static       Run static analysis"
	@echo "  deps         Run dependency analysis"
	@echo "  report       Generate audit reports"
	@echo "  clean        Clean generated files"
	@echo "  ci-setup     Setup CI environment"
	@echo "  test         Run tests"
	@echo "  pre-commit   Install and run pre-commit hooks"

# Installation
install:
	pip install -r requirements-dev.txt
	pip install pre-commit

# Full audit (comprehensive but slower)
audit:
	python -m auditor.audit_manager --target .

# Fast audit (quick checks on specific modules)
audit-fast:
	python -m auditor.audit_manager --target auditor/

# Linting
lint:
	python -m auditor.code_linter --mode check

lint-fix:
	python -m auditor.code_linter --mode fix

# Static analysis
static:
	python -m auditor.static_analysis

static-fast:
	python -m auditor.static_analysis --target auditor/

# Dependency analysis
deps:
	python -m auditor.dependency_checker

# Report generation
report:
	python -m auditor.code_quality_report

# Pre-commit hooks
pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

pre-commit: pre-commit-install pre-commit-run

# CI setup
ci-setup: install pre-commit-install

# Testing
test:
	pytest

test-coverage:
	pytest --cov=auditor --cov-report=html

# Cleaning
clean:
	rm -rf reports/ htmlcov/ .coverage .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Development helpers
format:
	black auditor/
	isort auditor/

check-format:
	black --check auditor/
	isort --check-only auditor/

# CI simulation
ci-simulate:
	@echo "Simulating CI pipeline..."
	make clean
	make install
	make pre-commit-run
	make audit-fast
	make report
	@echo "✅ CI simulation complete!"

# Docker support
docker-build:
	docker build -t auditor .

docker-run:
	docker run --rm -v $(PWD):/app auditor make audit

# Quick development checks
dev-check: lint static-fast deps
	@echo "✅ Development checks passed!"
