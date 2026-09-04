LATEXMK ?= latexmk
BUILD_DIR := ../../paper/build/qwen38-minimal
FINAL_DIR := ../../output/pdf
FINAL_PDF := ../../output/pdf/teaching-one-synthetic-fact-qwen38-minimal.pdf
SOURCES := main.tex references.bib \
	$(wildcard sections/*.tex) \
	$(wildcard appendices/*.tex)

.PHONY: all clean

all: $(FINAL_PDF)

$(FINAL_PDF): $(BUILD_DIR)/main.pdf
	mkdir -p $(FINAL_DIR)
	cp $(BUILD_DIR)/main.pdf $(FINAL_PDF)

$(BUILD_DIR)/main.pdf: $(SOURCES)
	mkdir -p $(BUILD_DIR)
	$(LATEXMK) -pdf -bibtex -interaction=nonstopmode -halt-on-error \
		-file-line-error -outdir=$(BUILD_DIR) main.tex

clean:
	$(LATEXMK) -C -outdir=$(BUILD_DIR) main.tex
	rm -rf $(BUILD_DIR)
