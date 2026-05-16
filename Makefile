.PHONY: smoke ui-smoke browser-smoke browser-agent-smoke health health-live app-smoke eval-gate answer-eval eval-all verify-live verify-demo rerank-benchmark vector-artifact verify-vector-artifact install-vector-artifact export-local

PYTHON ?= .venv/bin/python

smoke:
	$(PYTHON) -m py_compile app.py xiao_copilot/*.py scripts/*.py
	$(PYTHON) scripts/test_app_scope.py
	$(PYTHON) scripts/ui_contract_smoke.py
	$(PYTHON) scripts/test_vector_artifacts.py
	$(PYTHON) scripts/test_health_checks.py
	OFFLINE_EVAL=1 STRICT_EVAL=1 EVAL_LIMIT=9 $(PYTHON) scripts/eval_smoke.py

ui-smoke:
	$(PYTHON) scripts/ui_contract_smoke.py

browser-smoke:
	$(PYTHON) scripts/browser_smoke.py

browser-agent-smoke:
	$(PYTHON) scripts/browser_smoke.py --run-query --timeout-ms $${BROWSER_SMOKE_TIMEOUT_MS:-120000}

health:
	$(PYTHON) scripts/health_check.py

health-live:
	$(PYTHON) scripts/health_check.py --live --app --strict-warnings

app-smoke:
	$(PYTHON) scripts/app_smoke.py

eval-gate:
	./scripts/eval_gate.sh

answer-eval:
	$(PYTHON) scripts/eval_answer_quality.py

eval-all: eval-gate answer-eval

verify-live: smoke health-live app-smoke eval-all

verify-demo: verify-live
	$(MAKE) browser-agent-smoke PYTHON=$(PYTHON)

rerank-benchmark:
	$(PYTHON) scripts/benchmark_rerank_window.py --windows $${RERANK_BENCHMARK_WINDOWS:-900,1600,2400,3200}

vector-artifact:
	$(PYTHON) scripts/package_vector_artifact.py

verify-vector-artifact:
	$(PYTHON) scripts/verify_vector_artifact.py

install-vector-artifact:
	test -n "$$VECTOR_INDEX_ARCHIVE_URL"
	$(PYTHON) scripts/install_vector_artifact.py --url "$$VECTOR_INDEX_ARCHIVE_URL" --sha256 "$$VECTOR_INDEX_ARCHIVE_SHA256"

export-local:
	$(PYTHON) scripts/export_local_changes.py
