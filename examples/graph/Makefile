.POSIX:
.SUFFIXES:
.SUFFIXES: .gv
.SUFFIXES: .svg

0 = \
analitical.svg \
coins.svg \
three.svg \

all: $0
.gv.svg:; dot $< -Tsvg -o $@
clean:; rm -f $0
