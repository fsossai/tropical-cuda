.PHONY: all

all:
	cmake --preset default
	cmake --build --preset default

clean:
	rm -rf build