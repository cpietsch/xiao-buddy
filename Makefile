.PHONY: smoke health health-live eval-gate answer-eval eval-all rerank-benchmark vector-artifact install-vector-artifact

PYTHON ?= .venv/bin/python

smoke:
	$(PYTHON) -m py_compile app.py xiao_copilot/*.py scripts/*.py
	$(PYTHON) scripts/test_vector_artifacts.py
	$(PYTHON) scripts/test_health_checks.py
	OFFLINE_EVAL=1 STRICT_EVAL=1 EVAL_LIMIT=9 $(PYTHON) scripts/eval_smoke.py

health:
	$(PYTHON) scripts/health_check.py

health-live:
	$(PYTHON) scripts/health_check.py --live --app

eval-gate:
	./scripts/eval_gate.sh

answer-eval:
	$(PYTHON) scripts/eval_answer_quality.py

eval-all: eval-gate answer-eval

rerank-benchmark:
	$(PYTHON) scripts/benchmark_rerank_window.py --windows $${RERANK_BENCHMARK_WINDOWS:-900,1600,2400,3200}

vector-artifact:
	$(PYTHON) scripts/package_vector_artifact.py

install-vector-artifact:
	test -n "$$VECTOR_INDEX_ARCHIVE_URL"
	$(PYTHON) scripts/install_vector_artifact.py --url "$$VECTOR_INDEX_ARCHIVE_URL" --sha256 "$$VECTOR_INDEX_ARCHIVE_SHA256"
