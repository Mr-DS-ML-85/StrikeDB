.PHONY: build test prove-it

build:
	cargo build --release

test:
	cargo test --release

# Public evidence engine: crash durability, bulk atomicity, concurrency,
# mixed-workload invariants, per-subsystem durability, FLUSHALL undo.
# RESP-only, stdlib-only. Exit 0 = every proof held.
prove-it: build
	python3 scripts/prove_it.py
