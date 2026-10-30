.PHONY: test figures check

check:
	python -m grain --check

test:
	python -m unittest discover -s tests -v

figures:
	python -m grain --all --out results
