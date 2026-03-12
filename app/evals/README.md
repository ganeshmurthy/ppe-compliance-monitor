# PPE Compliance Monitor - LLM Evals

Evaluates the `/api/chat` endpoint by sending predefined questions and detection contexts from `eval_dataset.json`, then scoring responses against golden answers using DeepEval's GEval (Correctness) metric with a VLLM-backed judge model.

## Usage

The full stack must already be running (`make local-build-up` or equivalent) before running evals:

```bash
make eval
```

Results are printed as a summary table with per-entry scores and an overall pass rate.
