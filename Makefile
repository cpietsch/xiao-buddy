.PHONY: smoke health health-live eval-gate answer-eval eval-all rerank-benchmark

PYTHON ?= .venv/bin/python

smoke:
	$(PYTHON) -m py_compile app.py xiao_copilot/*.py scripts/*.py
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
