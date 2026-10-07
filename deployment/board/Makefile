# x86 golden-vector build (week 1-2). ARM targets arrive in week 3;
# flags must stay identical on both (EDGE_EXPERIMENT.md §7).
CC      ?= gcc
CFLAGS  = -O2 -ffp-contract=off -fno-fast-math -Wall -Wextra \
          -Isrc -Igolden
LDLIBS  = -lm

# horizon / channel overrides, e.g.  make feather2 FEA2_GEOM="-DFEA2_H=720"
FEA2_GEOM ?=
DL_GEOM   ?=
CN_GEOM   ?=

BUILD   = build
KERN    = src/fea_kernels.c src/fea_kernels.h golden/fea_twiddle.h

$(BUILD)/test_golden: src/feather_infer.c test/test_golden.c \
                      src/feather_infer.h $(KERN) \
                      golden/feather_weights.h golden/feather_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DFEA_CAPTURE_STAGES src/fea_kernels.c \
	      src/feather_infer.c test/test_golden.c -o $@ $(LDLIBS)

$(BUILD)/test_feather2: src/feather2_infer.c test/test_feather2.c                     src/feather2_infer.h $(KERN)                     golden/feather2_weights.h golden/feather2_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DFEA2_CAPTURE_STAGES $(FEA2_GEOM) src/fea_kernels.c 	      src/feather2_infer.c test/test_feather2.c -o $@ $(LDLIBS)

# FEATHer-CI shares golden/feather2_weights.h with the mixing variant, but the
# CI export writes it at WIDTH 1 - so this target and `test` cannot be satisfied
# by one export, and it deliberately stays out of `test`.  Regenerate first:
#   python deployment/board/export_golden_feather2ci.py --pred_len H --d_model D
#   make feather2ci FEA2_GEOM="-DFEA2_D=1 -DFEA2CI_D=D -DFEA2_H=H"
# then re-run export_golden_feather2.py before `make test`.
$(BUILD)/test_feather2ci: src/feather2_infer.c src/feather2ci_infer.c \
                    test/test_feather2ci.c src/feather2ci_infer.h $(KERN) \
                    golden/feather2_weights.h golden/feather2ci_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(FEA2_GEOM) src/fea_kernels.c src/feather2_infer.c \
	      src/feather2ci_infer.c test/test_feather2ci.c -o $@ $(LDLIBS)

# FEATHer in1 (2026-09-15 candidate): mixer reads the input, sensors finish one
# at a time.  Both schedules are built; see src/feather2in_infer.h.  Kept out of
# `test` because it is not the frozen model.  Regenerate the goldens first:
#   python deployment/board/export_golden_feather2in.py [--pred_len H --d_model D]
FEA2I_GEOM ?=
$(BUILD)/test_feather2in: src/feather2in_infer.c test/test_feather2in.c \
                    src/feather2in_infer.h $(KERN) \
                    golden/feather2in_weights.h golden/feather2in_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DFEA2I_CAPTURE_STAGES -DFEA2I_RECOMPUTE_NORM=1 $(FEA2I_GEOM) \
	      src/fea_kernels.c src/feather2in_infer.c test/test_feather2in.c -o $@ $(LDLIBS)

$(BUILD)/test_feather2in_strict: src/feather2in_infer.c test/test_feather2in.c \
                    src/feather2in_infer.h $(KERN) \
                    golden/feather2in_weights.h golden/feather2in_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DFEA2I_CAPTURE_STAGES -DFEA2I_RECOMPUTE_NORM=0 $(FEA2I_GEOM) \
	      src/fea_kernels.c src/feather2in_infer.c test/test_feather2in.c -o $@ $(LDLIBS)

$(BUILD)/test_dlinear: src/dlinear_infer.c test/test_dlinear.c src/dlinear_infer.h golden/dlinear_weights.h golden/dlinear_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DDL_CAPTURE_STAGES $(DL_GEOM) src/dlinear_infer.c test/test_dlinear.c -o $@ $(LDLIBS)

# CycleNet (2026-09-30): ported for fairness like DLinear; weights fit at H = 96/192
$(BUILD)/test_cyclenet: src/cyclenet_infer.c test/test_cyclenet.c src/cyclenet_infer.h golden/cyclenet_weights.h golden/cyclenet_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DCN_CAPTURE_STAGES $(CN_GEOM) src/cyclenet_infer.c test/test_cyclenet.c -o $@ $(LDLIBS)

# CMoS (2026-10-07): structural values (seg, num_map, kernel, stride) come from the
# exported weights header; regenerate with export_golden_cmos.py --config <setting>
CMOS_GEOM ?=
$(BUILD)/test_cmos: src/cmos_infer.c test/test_cmos.c src/cmos_infer.h golden/cmos_weights.h golden/cmos_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DCMOS_CAPTURE_STAGES $(CMOS_GEOM) src/cmos_infer.c test/test_cmos.c -o $@ $(LDLIBS)

# TimeBase (2026-10-07): period / basis / individual come from the exported weights header
TB_GEOM ?=
$(BUILD)/test_timebase: src/timebase_infer.c test/test_timebase.c src/timebase_infer.h golden/timebase_weights.h golden/timebase_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DTB_CAPTURE_STAGES $(TB_GEOM) src/timebase_infer.c test/test_timebase.c -o $@ $(LDLIBS)

# PhaseFormer (2026-10-07): widths come from the exported weights header
PF_GEOM ?=
$(BUILD)/test_phaseformer: src/phaseformer_infer.c test/test_phaseformer.c src/phaseformer_infer.h golden/phaseformer_weights.h golden/phaseformer_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DPF_CAPTURE_STAGES $(PF_GEOM) src/phaseformer_infer.c test/test_phaseformer.c -o $@ $(LDLIBS)

# SAMformer (2026-10-07): the channel-mixing reference; hid_dim comes from the weights header
SAM_GEOM ?=
$(BUILD)/test_samformer: src/samformer_infer.c test/test_samformer.c src/samformer_infer.h golden/samformer_weights.h golden/samformer_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DSAM_CAPTURE_STAGES $(SAM_GEOM) src/samformer_infer.c test/test_samformer.c -o $@ $(LDLIBS)

$(BUILD)/test_realcheck: src/feather2_infer.c test/test_realcheck.c src/feather2_infer.h $(KERN) golden/feather2_weights.h golden/real_ref.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(FEA2_GEOM) src/fea_kernels.c src/feather2_infer.c test/test_realcheck.c -o $@ $(LDLIBS)

# the shipping port of the re-frozen model (recompute schedule, 2026-09-22/23);
# must be bit-identical to feather2_infer.c on the golden
$(BUILD)/test_feather2r: src/feather2_infer.c src/feather2r_infer.c test/test_feather2r.c \
                    src/feather2r_infer.h src/feather2_infer.h $(KERN) \
                    golden/feather2_weights.h golden/feather2_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(FEA2_GEOM) src/fea_kernels.c src/feather2_infer.c \
	      src/feather2r_infer.c test/test_feather2r.c -o $@ $(LDLIBS)

# real-data replay through the shipping recompute port.  Export first:
#   python deployment/board/verify_deployed_accuracy.py --ckpt C --mlp_rank 4 ...
#   python deployment/board/export_golden_feather2.py --ckpt C --mlp_rank 4 --pred_len H
#   make realcheck_r FEA2_GEOM="-DFEA2_MLP_R=4 -DFEA2_H=H"
$(BUILD)/test_realcheck_r: src/feather2r_infer.c test/test_realcheck.c src/feather2r_infer.h $(KERN) golden/feather2_weights.h golden/real_ref.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DREALCHECK_RECOMPUTE $(FEA2_GEOM) src/fea_kernels.c src/feather2r_infer.c test/test_realcheck.c -o $@ $(LDLIBS)

$(BUILD)/test_stsf: src/stsf_infer.c test/test_stsf.c \
                    src/stsf_infer.h $(KERN) \
                    golden/stsf_weights.h golden/stsf_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DSTSF_CAPTURE_STAGES src/fea_kernels.c src/stsf_infer.c \
	      test/test_stsf.c -o $@ $(LDLIBS)

$(BUILD)/test_stsf_mlp: src/stsf_mlp_infer.c test/test_stsf_mlp.c \
                    src/stsf_mlp_infer.h $(KERN) \
                    golden/stsf_mlp_weights.h golden/stsf_mlp_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DSMLP_CAPTURE_STAGES src/fea_kernels.c \
	      src/stsf_mlp_infer.c test/test_stsf_mlp.c -o $@ $(LDLIBS)

$(BUILD)/test_mixlinear: src/mixlinear_infer.c test/test_mixlinear.c \
                    src/mixlinear_infer.h $(KERN) \
                    golden/mixlinear_weights.h golden/mixlinear_vectors.h \
                    golden/mixlinear_twiddle.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DML_CAPTURE_STAGES src/fea_kernels.c src/mixlinear_infer.c \
	      test/test_mixlinear.c -o $@ $(LDLIBS)

$(BUILD)/test_dipe: src/dipe_infer.c test/test_dipe.c \
                    src/dipe_infer.h $(KERN) \
                    golden/dipe_weights.h golden/dipe_vectors.h \
                    golden/dipe_twiddle.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DDIPE_CAPTURE_STAGES src/fea_kernels.c src/dipe_infer.c \
	      test/test_dipe.c -o $@ $(LDLIBS)

$(BUILD)/test_pers: test/test_pers.c src/pers_infer.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) test/test_pers.c -o $@ $(LDLIBS)

$(BUILD)/test_fits: src/fits_infer.c test/test_fits.c \
                    src/fits_infer.h $(KERN) \
                    golden/fits_weights.h golden/fits_vectors.h \
                    golden/fits_twiddle.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DFITS_CAPTURE_STAGES src/fea_kernels.c src/fits_infer.c \
	      test/test_fits.c -o $@ $(LDLIBS)

$(BUILD)/test_batch: test/test_batch.c $(KERN) \
                     src/feather_infer.c src/stsf_infer.c \
                     src/dipe_infer.c src/fits_infer.c src/pers_infer.h \
                     golden/batch_ref.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) src/fea_kernels.c src/feather_infer.c \
	      src/stsf_infer.c src/dipe_infer.c src/fits_infer.c \
	      test/test_batch.c -o $@ $(LDLIBS)

.PHONY: test clean feather2 feather2r feather2ci feather2in dlinear cyclenet cmos timebase phaseformer samformer realcheck realcheck_r
feather2in: $(BUILD)/test_feather2in $(BUILD)/test_feather2in_strict
	./$(BUILD)/test_feather2in
	./$(BUILD)/test_feather2in_strict

realcheck: $(BUILD)/test_realcheck
	./$(BUILD)/test_realcheck

realcheck_r: $(BUILD)/test_realcheck_r
	./$(BUILD)/test_realcheck_r

feather2r: $(BUILD)/test_feather2r
	./$(BUILD)/test_feather2r

dlinear: $(BUILD)/test_dlinear
	./$(BUILD)/test_dlinear

cyclenet: $(BUILD)/test_cyclenet
	./$(BUILD)/test_cyclenet

cmos: $(BUILD)/test_cmos
	./$(BUILD)/test_cmos

timebase: $(BUILD)/test_timebase
	./$(BUILD)/test_timebase

phaseformer: $(BUILD)/test_phaseformer
	./$(BUILD)/test_phaseformer

samformer: $(BUILD)/test_samformer
	./$(BUILD)/test_samformer

feather2: $(BUILD)/test_feather2
	./$(BUILD)/test_feather2

feather2ci: $(BUILD)/test_feather2ci
	./$(BUILD)/test_feather2ci

test: $(BUILD)/test_feather2 $(BUILD)/test_feather2r $(BUILD)/test_dlinear $(BUILD)/test_cyclenet $(BUILD)/test_cmos $(BUILD)/test_timebase $(BUILD)/test_phaseformer $(BUILD)/test_samformer $(BUILD)/test_golden $(BUILD)/test_stsf $(BUILD)/test_stsf_mlp \
      $(BUILD)/test_mixlinear $(BUILD)/test_dipe $(BUILD)/test_pers \
      $(BUILD)/test_fits $(BUILD)/test_batch
	./$(BUILD)/test_feather2
	./$(BUILD)/test_feather2r
	./$(BUILD)/test_dlinear
	./$(BUILD)/test_cyclenet
	./$(BUILD)/test_cmos
	./$(BUILD)/test_timebase
	./$(BUILD)/test_phaseformer
	./$(BUILD)/test_samformer
	./$(BUILD)/test_golden
	./$(BUILD)/test_stsf
	./$(BUILD)/test_stsf_mlp
	./$(BUILD)/test_mixlinear
	./$(BUILD)/test_dipe
	./$(BUILD)/test_pers
	./$(BUILD)/test_fits
	./$(BUILD)/test_batch

clean:
	rm -rf $(BUILD)
