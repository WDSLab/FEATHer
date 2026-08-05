# x86 golden-vector build (week 1-2). ARM targets arrive in week 3;
# flags must stay identical on both (EDGE_EXPERIMENT.md §7).
CC      ?= gcc
CFLAGS  = -O2 -ffp-contract=off -fno-fast-math -Wall -Wextra \
          -Isrc -Igolden
LDLIBS  = -lm

BUILD   = build
KERN    = src/fea_kernels.c src/fea_kernels.h golden/fea_twiddle.h

$(BUILD)/test_golden: src/feather_infer.c test/test_golden.c \
                      src/feather_infer.h $(KERN) \
                      golden/feather_weights.h golden/feather_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DFEA_CAPTURE_STAGES src/fea_kernels.c \
	      src/feather_infer.c test/test_golden.c -o $@ $(LDLIBS)

$(BUILD)/test_stsf: src/stsf_infer.c test/test_stsf.c \
                    src/stsf_infer.h $(KERN) \
                    golden/stsf_weights.h golden/stsf_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) src/fea_kernels.c src/stsf_infer.c \
	      test/test_stsf.c -o $@ $(LDLIBS)

$(BUILD)/test_dipe: src/dipe_infer.c test/test_dipe.c \
                    src/dipe_infer.h $(KERN) \
                    golden/dipe_weights.h golden/dipe_vectors.h \
                    golden/dipe_twiddle.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) src/fea_kernels.c src/dipe_infer.c \
	      test/test_dipe.c -o $@ $(LDLIBS)

$(BUILD)/test_pers: test/test_pers.c src/pers_infer.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) test/test_pers.c -o $@ $(LDLIBS)

$(BUILD)/test_fits: src/fits_infer.c test/test_fits.c \
                    src/fits_infer.h $(KERN) \
                    golden/fits_weights.h golden/fits_vectors.h \
                    golden/fits_twiddle.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) src/fea_kernels.c src/fits_infer.c \
	      test/test_fits.c -o $@ $(LDLIBS)

$(BUILD)/test_batch: test/test_batch.c $(KERN) \
                     src/feather_infer.c src/stsf_infer.c \
                     src/dipe_infer.c src/fits_infer.c src/pers_infer.h \
                     golden/batch_ref.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) src/fea_kernels.c src/feather_infer.c \
	      src/stsf_infer.c src/dipe_infer.c src/fits_infer.c \
	      test/test_batch.c -o $@ $(LDLIBS)

.PHONY: test clean
test: $(BUILD)/test_golden $(BUILD)/test_stsf $(BUILD)/test_dipe \
      $(BUILD)/test_pers $(BUILD)/test_fits $(BUILD)/test_batch
	./$(BUILD)/test_golden
	./$(BUILD)/test_stsf
	./$(BUILD)/test_dipe
	./$(BUILD)/test_pers
	./$(BUILD)/test_fits
	./$(BUILD)/test_batch

clean:
	rm -rf $(BUILD)
