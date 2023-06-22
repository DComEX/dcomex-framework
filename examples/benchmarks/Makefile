.POSIX:
.SUFFIXES:
.SUFFIXES: .gp .pdf .png

F = \
strong.gp \
strong.multi.gp \
weak.gp \
weak.multi.gp \

all: $(F:.gp=.png) $(F:.gp=.pdf)

.gp.pdf:; t=/tmp/$$$$.bench && GNUTERM=pdf gnuplot $< > $$t && mv $$t $@
.gp.png:; t=/tmp/$$$$.bench && GNUTERM=pngcairo gnuplot $< > $$t && mv $$t $@
