LATEXMK ?= latexmk
BUILD_DIR := build
FINAL_DIR := ../output/pdf
FINAL_PDF := $(FINAL_DIR)/teaching-one-synthetic-fact-qwen35.pdf
SOURCES := main.tex references.bib \
	$(wildcard sections/*.tex) \
	$(wildcard figures/*.tex) \
	$(wildcard appendices/*.tex)

.PHONY: all clean

all: $(FINAL_PDF)

$(FINAL_PDF): $(SOURCES)
	mkdir -p $(BUILD_DIR) $(FINAL_DIR)
	$(LATEXMK) -pdf -bibtex -interaction=nonstopmode -halt-on-error \
		-file-line-error -outdir=$(BUILD_DIR) main.tex
	cp $(BUILD_DIR)/main.pdf $(FINAL_PDF)

clean:
	$(LATEXMK) -C -outdir=$(BUILD_DIR) main.tex
	rm -rf $(BUILD_DIR)
