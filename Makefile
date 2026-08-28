.PHONY: dev eval offline-test demo

dev:
	docker-compose up api web

eval:
	# Run the batch evaluation mode
	satquery eval --manifest data/manifest.jsonl --task vqa --out preds.jsonl

offline-test:
	# Run tests ensuring no network access
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest tests/

demo:
	# Run the demo bundle
	docker-compose up

test-resume:
	python training/run_checkpoint_test.py

