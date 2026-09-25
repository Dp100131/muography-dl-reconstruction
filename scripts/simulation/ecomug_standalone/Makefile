# Standalone EcoMug generator. Requires g++ (build-essential) and git.
# EcoMug is header-only: we fetch the single header at a pinned commit.
# TODO(freeze): pin ECOMUG_REF to a commit hash after first successful build
# and record it somewhere durable.

ECOMUG_REF ?= main
ECOMUG_DIR := ecomug-src

ecomug_gen: main.cc $(ECOMUG_DIR)/EcoMug.h
	g++ -O2 -std=c++17 -I$(ECOMUG_DIR) -o $@ main.cc

$(ECOMUG_DIR)/EcoMug.h:
	git clone --depth 1 --branch $(ECOMUG_REF) https://github.com/dr4kan/EcoMug.git $(ECOMUG_DIR)-repo
	mkdir -p $(ECOMUG_DIR)
	cp $(ECOMUG_DIR)-repo/EcoMug.h $(ECOMUG_DIR)/ 2>/dev/null || cp $(ECOMUG_DIR)-repo/include/EcoMug.h $(ECOMUG_DIR)/
	rm -rf $(ECOMUG_DIR)-repo

clean:
	rm -rf ecomug_gen $(ECOMUG_DIR)
