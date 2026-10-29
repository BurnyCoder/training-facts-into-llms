LATEXMK ?= latexmk
BUILD_DIR := ../../artifacts/papers/qwen35-revised
FINAL_PDF := ../../output/pdf/teaching-one-synthetic-fact-qwen35-revised.pdf
SOURCES := main.tex references.bib ../paper-preamble.tex $(wildcard sections/*.tex) $(wildcard generated/*)

.PHONY: all clean
all: $(FINAL_PDF)

$(FINAL_PDF): $(BUILD_DIR)/main.pdf
	mkdir -p $(dir $(FINAL_PDF))
	cp $< $@

$(BUILD_DIR)/main.pdf: $(SOURCES)
	mkdir -p $(BUILD_DIR)
	$(LATEXMK) -pdf -bibtex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=$(BUILD_DIR) main.tex

clean:
	$(LATEXMK) -C -outdir=$(BUILD_DIR) main.tex
