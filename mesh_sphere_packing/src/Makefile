CC = gcc

#   TRILIBDEFS = -DTRILIBRARY -DREDUCED -DCDT_ONLY
TRILIBDEFS = -DTRILIBRARY

default: libwraptri.so

all: libwraptri.so

libwraptri.so:
	$(CC) $(TRILIBDEFS) -shared -o libwraptri.so -fPIC wraptri.c triangle.c

test_libwraptri: libwraptri.so
	$(CC) -L./ -Wl,-rpath=./ -o test_libwraptri test.c -lwraptri -lm

test: test_libwraptri
	./test_libwraptri

clean:
	rm -f libwraptri.so
	rm -f test_libwraptri
