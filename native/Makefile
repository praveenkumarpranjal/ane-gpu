DYLIB = ../anegpu/libanegpu.dylib

$(DYLIB): ane_ffn.m
	xcrun clang -O2 -fobjc-arc -dynamiclib \
		-framework Foundation -framework IOSurface \
		-ldl -o $(DYLIB) ane_ffn.m

clean:
	rm -f $(DYLIB)
