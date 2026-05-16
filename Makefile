.PHONY: smoke ui-smoke corpus-scope browser-smoke browser-agent-smoke health health-live health-functional app-smoke eval-gate answer-eval eval-all verify-live verify-demo verify-handoff import-wiki-xiao import-wiki-all build-hnsw build-faiss-pq rerank-benchmark rerank-quality rerank-quality-all vector-artifact verify-vector-artifact verify-vector-artifact-restore install-vector-artifact export-local

PYTHON ?= .venv/bin/python

smoke:
	$(PYTHON) -m py_compile app.py xiao_copilot/*.py scripts/*.py
	$(PYTHON) scripts/test_config.py
	$(PYTHON) scripts/test_env_template.py
	$(PYTHON) scripts/test_clients.py
	$(PYTHON) scripts/test_retrieval.py
	$(PYTHON) scripts/test_app_scope.py
	$(PYTHON) scripts/test_pipeline_failover.py
	$(PYTHON) scripts/test_corpus_scope.py
	$(PYTHON) scripts/ui_contract_smoke.py
	$(PYTHON) scripts/test_vector_artifacts.py
	$(PYTHON) scripts/test_health_checks.py
	OFFLINE_EVAL=1 STRICT_EVAL=1 EVAL_LIMIT=9 $(PYTHON) scripts/eval_smoke.py

ui-smoke:
	$(PYTHON) scripts/ui_contract_smoke.py

corpus-scope:
	$(PYTHON) scripts/test_corpus_scope.py

browser-smoke:
	$(PYTHON) scripts/browser_smoke.py

browser-agent-smoke:
	$(PYTHON) scripts/browser_smoke.py --run-query --timeout-ms $${BROWSER_SMOKE_TIMEOUT_MS:-120000}

health:
	$(PYTHON) scripts/health_check.py

health-live:
	$(PYTHON) scripts/health_check.py --live --app --strict-warnings

health-functional:
	$(PYTHON) scripts/endpoint_smoke.py

app-smoke:
	$(PYTHON) scripts/app_smoke.py

eval-gate:
	./scripts/eval_gate.sh

answer-eval:
	$(PYTHON) scripts/eval_answer_quality.py

eval-all: eval-gate answer-eval

verify-live: smoke health-live app-smoke eval-all

verify-demo: verify-live
	$(MAKE) health-functional PYTHON=$(PYTHON)
	$(MAKE) rerank-quality-all PYTHON=$(PYTHON)
	$(MAKE) browser-agent-smoke PYTHON=$(PYTHON)

verify-handoff: verify-demo verify-vector-artifact-restore
	$(MAKE) export-local PYTHON=$(PYTHON)

import-wiki-xiao:
	$(PYTHON) scripts/import_seeed_wiki.py --scope xiao --refresh

import-wiki-all:
	$(PYTHON) scripts/import_seeed_wiki.py --scope all --refresh

build-hnsw:
	$(PYTHON) scripts/build_wiki_vector_index.py --backend hnsw --batch-size $${BATCH_SIZE:-32}

build-faiss-pq:
	$(PYTHON) scripts/build_wiki_vector_index.py --backend faiss-pq --batch-size $${BATCH_SIZE:-32}

rerank-benchmark:
	$(PYTHON) scripts/benchmark_rerank_window.py --windows $${RERANK_BENCHMARK_WINDOWS:-900,1600,2400,3200}

rerank-quality:
	$(PYTHON) scripts/eval_reranker_quality.py

rerank-quality-all:
	$(PYTHON) scripts/eval_reranker_quality.py --all --min-pass-rate $${RERANK_QUALITY_ALL_MIN_PASS_RATE:-1.0} --max-failures $${RERANK_QUALITY_ALL_MAX_FAILURES:-0} --json-output dist/reranker-quality/all.json

vector-artifact:
	$(PYTHON) scripts/package_vector_artifact.py

verify-vector-artifact:
	$(PYTHON) scripts/verify_vector_artifact.py

verify-vector-artifact-restore: verify-vector-artifact
	$(PYTHON) scripts/verify_vector_artifact_restore.py

install-vector-artifact:
	test -n "$$VECTOR_INDEX_ARCHIVE_URL"
	$(PYTHON) scripts/install_vector_artifact.py --url "$$VECTOR_INDEX_ARCHIVE_URL" --sha256 "$$VECTOR_INDEX_ARCHIVE_SHA256"

export-local:
	$(PYTHON) scripts/export_local_changes.py
