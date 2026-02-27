.PHONY: install lint test notebook clean help

help:
	@echo "GeoTop — available commands:"
	@echo "  make install    Install all dependencies"
	@echo "  make lint       Lint source code"
	@echo "  make test       Run unit tests"
	@echo "  make notebook   Execute main GeoTop notebook"
	@echo "  make clean      Remove cache and executed notebooks"

install:
	pip install -r requirements.txt

lint:
	flake8 Code/ --max-line-length=127 --count --statistics

test:
	pytest tests/ -v --tb=short

notebook:
	jupyter nbconvert --to notebook --execute GeoTop.ipynb \
		--output GeoTop_executed.ipynb \
		--ExecutePreprocessor.timeout=600

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name ".DS_Store" -delete 2>/dev/null; true
	find . -name "*_executed.ipynb" -delete 2>/dev/null; true
	find . -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null; true
