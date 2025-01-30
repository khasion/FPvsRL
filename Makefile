PYTHONVERSION = python
LATEXVERSION = pdflatex

PRISONERS_PY = prisoners.py
RPS_PY = rock_paper_scissors.py

REPORT_PDF = report.pdf
REPORT_TEX = report.tex


PRISONERS_RES = pd_behavior.png pd_cumulative.png pd_rewards.png
RPS_RES = rps_cumulative.png rps_fp_strategy.png rps_ql_strategy.png rps_rewards_comparison.png rps_rewards.png

all: $(PRISONERS_RES) $(RPS_RES) $(REPORT_PDF)

$(PRISONERS_RES): $(PRISONERS_PY)
	python3 $(PRISONERS_PY)

$(RPS_RES): $(RPS_PY)
	python3 $(RPS_PY)

$(REPORT_PDF): $(REPORT_TEX)
	pdflatex report.tex
	pdflatex report.tex

clean:
	rm $(PRISONERS_RES) $(RPS_RES) report.aux report.log report.out report.toc