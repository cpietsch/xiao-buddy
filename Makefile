.PHONY: eval-gate answer-eval eval-all rerank-benchmark

eval-gate:
	./scripts/eval_gate.sh

answer-eval:
	.venv/bin/python scripts/eval_answer_quality.py

eval-all: eval-gate answer-eval

rerank-benchmark:
	.venv/bin/python scripts/benchmark_rerank_window.py --windows $${RERANK_BENCHMARK_WINDOWS:-900,1600,2400,3200}
