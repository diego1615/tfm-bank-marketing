PYTHON ?= python

.PHONY: reproduce test notebook
reproduce:
	PYTHONPATH=src MPLCONFIGDIR=/tmp/tfm-matplotlib OMP_NUM_THREADS=2 $(PYTHON) -m bank_marketing.run

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

notebook:
	PYTHONPATH=src MPLCONFIGDIR=/tmp/tfm-matplotlib $(PYTHON) scripts/build_notebook.py
