# PANDOC VARIABLES
PANDOC := pandoc
FILTERS := pandoc-crossref pandoc-citeproc 
PFLAGS := $(foreach filter, $(FILTERS),-F $(filter))

# GENERAL VARIABLES
SRCS := header.md memoria.md
DEPS := citas.bib style.csl
OUTPDF := ../memoria.pdf

.PHONY: all clean

all: $(OUTPDF)

$(OUTPDF): $(SRCS) $(DEPS)
	$(PANDOC) $(PFLAGS) $(SRCS) -o $@

clean:
	rm -rf $(OUTPDF)
