# Cortex-M3 firmware build for QEMU lm3s6965evb.
# Run from WSL with arm-none-eabi-gcc installed.

CROSS      ?= arm-none-eabi
CC         := $(CROSS)-gcc
LD         := $(CROSS)-gcc
SIZE       := $(CROSS)-size
OBJCOPY    := $(CROSS)-objcopy

# RAM budget overridable: make RAM_KB=32
RAM_KB     ?= 64

CFLAGS     := -Os -g -mthumb -mcpu=cortex-m3 \
              -ffunction-sections -fdata-sections \
              -Wall -Wextra -std=c11 \
              -DARENA_MAX_KB=$(RAM_KB)
LDFLAGS    := -mthumb -mcpu=cortex-m3 -nostartfiles \
              --specs=nosys.specs --specs=rdimon.specs \
              -Wl,--gc-sections \
              -T lm3s6965_$(RAM_KB)k.ld

OBJS       := main.o

firmware.elf: $(OBJS) lm3s6965_$(RAM_KB)k.ld
	$(LD) $(LDFLAGS) -o $@ $(OBJS) -lc -lrdimon
	$(SIZE) $@

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

run: firmware.elf
	qemu-system-arm -M lm3s6965evb -kernel firmware.elf \
	    -nographic -semihosting

clean:
	rm -f *.o *.elf

.PHONY: run clean
