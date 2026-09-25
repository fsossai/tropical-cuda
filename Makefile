BUILD_DIR := build
DATA_DIR := data
CONVERTER := $(BUILD_DIR)/graph_convert
SNAP_URL := 
GRAPH_NAMES := \
	web-Google \
	roadNet-CA \
	soc-LiveJournal1 \
	wiki-Talk \
	email-EuAll \
	web-BerkStan \
	web-NotreDame \
	amazon0601 \
	roadNet-PA \
	roadNet-TX \
	web-Stanford \
	soc-pokec-relationships

CSR_BINARIES := $(GRAPH_NAMES:%=$(DATA_DIR)/%.csrbin)

.PHONY: all clean inputs

all:
	cmake --preset default
	cmake --build --preset default -j$$(nproc)

inputs: $(CSR_BINARIES)

$(CONVERTER):
	cmake --preset default
	cmake --build --preset default --target graph_convert -j$$(nproc)

$(DATA_DIR):
	mkdir -p $@

$(DATA_DIR)/%.txt.gz: | $(DATA_DIR)
	curl --fail --location --retry 3 --output $@ https://snap.stanford.edu/data/$*.txt.gz

$(DATA_DIR)/%.txt: $(DATA_DIR)/%.txt.gz
	gzip --decompress --stdout $< >$@.tmp
	mv $@.tmp $@

$(DATA_DIR)/%.csrbin: $(DATA_DIR)/%.txt $(CONVERTER)
	$(CONVERTER) $< $@

clean:
	rm -rf build
