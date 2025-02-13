PYTHONVERSION = python
LATEXVERSION = pdflatex

MP_PY = matching_pennies.py
RPS_PY = rock_paper_scissors.py

REPORT_PDF = report.pdf
REPORT_TEX = report.tex


MP_RES = mp_simulation_data.csv
RPS_RES = rps_simulation_data.csv

all: $(MP_RES) $(RPS_RES) $(REPORT_PDF)

$(MP_RES): $(MP_PY)
	python3 $(MP_PY)

$(RPS_RES): $(RPS_PY)
	python3 $(RPS_PY)

$(REPORT_PDF): $(REPORT_TEX)
	pdflatex report.tex
	pdflatex report.tex

clean:
	rm rps-plots/* mp-plots/* *.csv report.aux report.log report.out