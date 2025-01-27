PYTHONVERSION = python
LATEXVERSION = pdflatex

PRISONERS_PY = prisoners.py
RPS_PY = rock_paper_scissors.py

REPORT_PDF = report.pdf
REPORT_TEX = report.tex


PRISONERS_RES = prisoners_results.png
RPS_RES = rps_results.png

all: $(PRISONERS_RES) $(RPS_RES) $(REPORT_PDF)

$(PRISONERS_RES): $(PRISONERS_PY)
	python3 $(PRISONERS_PY)

$(RPS_RES): $(RPS_PY)
	python3 $(RPS_PY)

$(REPORT_PDF): $(REPORT_TEX)
	pdflatex report.tex

clean:
	rm $(PRISONERS_RES) $(RPS_RES) report.aux report.log