# x86 golden-vector tests of the C ports (make test).  Same flags as the ARM build.
CC      ?= gcc
CFLAGS  = -O2 -ffp-contract=off -fno-fast-math -Wall -Wextra \
          -Isrc -Igolden
LDLIBS  = -lm

# horizon / channel overrides, e.g.  make feather FEA2_GEOM="-DFEA2_D=1 -DFEA2CI_D=7 -DFEA2_H=720"
# (default = export_golden_feather.py's default cell: width 1, 7 channels, H 96)
FEA2_GEOM ?= -DFEA2_D=1 -DFEA2CI_D=7

BUILD   = build
KERN    = src/mcu_kernels.c src/mcu_kernels.h src/mcu_pt.c src/mcu_pt.h
POCKET  = src/mcu_pocket.c src/mcu_pocket.h golden/mcu_rfft96_plan.h   # pocketfft's transforms

# FEATHer-CI = feather_core.c compiled at width 1, driven per channel.  Regenerate with
#   python deployment/board/export/export_golden_feather.py --pred_len H --d_model D
#   make feather FEA2_GEOM="-DFEA2_D=1 -DFEA2CI_D=D -DFEA2_H=H"
$(BUILD)/test_feather: src/feather_core.c src/feather_infer.c \
                    test/test_feather.c src/feather_infer.h $(KERN) $(POCKET) \
                    golden/feather_weights.h golden/feather_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(FEA2_GEOM) src/mcu_kernels.c src/mcu_pocket.c src/feather_core.c \
	      src/feather_infer.c test/test_feather.c src/mcu_pt.c -o $@ $(LDLIBS)

# CMoS: structural values (seg, num_map, kernel, stride) come from the
# exported weights header; regenerate with export_golden_cmos.py --config <setting>
CMOS_GEOM ?=
$(BUILD)/test_cmos: src/cmos_infer.c test/test_cmos.c src/cmos_infer.h golden/cmos_weights.h golden/cmos_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DCMOS_CAPTURE_STAGES $(CMOS_GEOM) src/cmos_infer.c test/test_cmos.c src/mcu_pt.c -o $@ $(LDLIBS)

# TimeBase: period / basis / individual come from the exported weights header
TB_GEOM ?=
$(BUILD)/test_timebase: src/timebase_infer.c test/test_timebase.c src/timebase_infer.h golden/timebase_weights.h golden/timebase_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DTB_CAPTURE_STAGES $(TB_GEOM) src/timebase_infer.c test/test_timebase.c src/mcu_pt.c -o $@ $(LDLIBS)

# PhaseFormer: widths come from the exported weights header
PF_GEOM ?=
$(BUILD)/test_phaseformer: src/phaseformer_infer.c test/test_phaseformer.c src/phaseformer_infer.h golden/phaseformer_weights.h golden/phaseformer_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DPF_CAPTURE_STAGES $(PF_GEOM) src/phaseformer_infer.c test/test_phaseformer.c src/mcu_pt.c -o $@ $(LDLIBS)

STSF_GEOM ?=
$(BUILD)/test_stsf: src/stsf_infer.c test/test_stsf.c \
                    src/stsf_infer.h $(KERN) \
                    golden/stsf_weights.h golden/stsf_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DSTSF_CAPTURE_STAGES $(STSF_GEOM) src/mcu_kernels.c src/stsf_infer.c \
	      test/test_stsf.c src/mcu_pt.c -o $@ $(LDLIBS)

SMLP_GEOM ?=
$(BUILD)/test_stsf_mlp: src/stsf_mlp_infer.c test/test_stsf_mlp.c \
                    src/stsf_mlp_infer.h $(KERN) \
                    golden/stsf_mlp_weights.h golden/stsf_mlp_vectors.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DSMLP_CAPTURE_STAGES $(SMLP_GEOM) src/mcu_kernels.c \
	      src/stsf_mlp_infer.c test/test_stsf_mlp.c src/mcu_pt.c -o $@ $(LDLIBS)

ML_GEOM ?=
$(BUILD)/test_mixlinear: src/mixlinear_infer.c test/test_mixlinear.c \
                    src/mixlinear_infer.h $(KERN) \
                    golden/mixlinear_weights.h golden/mixlinear_vectors.h \
                    golden/mixlinear_fft.h $(POCKET)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DML_CAPTURE_STAGES $(ML_GEOM) src/mcu_kernels.c src/mcu_pocket.c src/mixlinear_infer.c \
	      test/test_mixlinear.c src/mcu_pt.c -o $@ $(LDLIBS)

FITS_GEOM ?=
$(BUILD)/test_fits: src/fits_infer.c test/test_fits.c \
                    src/fits_infer.h $(KERN) \
                    golden/fits_weights.h golden/fits_vectors.h \
                    golden/fits_fft.h $(POCKET)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DFITS_CAPTURE_STAGES $(FITS_GEOM) src/mcu_kernels.c src/mcu_pocket.c src/fits_infer.c \
	      test/test_fits.c src/mcu_pt.c -o $@ $(LDLIBS)

.PHONY: test clean feather cmos timebase phaseformer stsf stsf_mlp mixlinear fits
feather: $(BUILD)/test_feather
	./$(BUILD)/test_feather

cmos: $(BUILD)/test_cmos
	./$(BUILD)/test_cmos

timebase: $(BUILD)/test_timebase
	./$(BUILD)/test_timebase

phaseformer: $(BUILD)/test_phaseformer
	./$(BUILD)/test_phaseformer

stsf: $(BUILD)/test_stsf
	./$(BUILD)/test_stsf

stsf_mlp: $(BUILD)/test_stsf_mlp
	./$(BUILD)/test_stsf_mlp

mixlinear: $(BUILD)/test_mixlinear
	./$(BUILD)/test_mixlinear

fits: $(BUILD)/test_fits
	./$(BUILD)/test_fits

TESTS = feather cmos timebase phaseformer stsf stsf_mlp mixlinear fits
test: $(addprefix $(BUILD)/test_,$(TESTS))
	@for t in $(TESTS); do echo ./$(BUILD)/test_$$t; ./$(BUILD)/test_$$t || exit 1; done

clean:
	rm -rf $(BUILD)
