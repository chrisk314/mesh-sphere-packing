CC = gcc
CFLAGS = -Wall -g -I./
LDFLAGS = -L/usr/lib/x86_64-linux-gnu
LIBS = -lhdf5

default: libwrapmfh5.so

all: libwrapmfh5.so

libwrapmfh5.so:
	$(CC) $(CFLAGS) $(LDFLAGS) -shared -o libwrapmfh5.so -fPIC mfh5.c $(LIBS)

clean:
	rm -f libwrapmfh5.so
