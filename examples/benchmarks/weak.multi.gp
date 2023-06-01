set monochrome
set log x
set xlabel "logical cores for workers"
set ylabel "efficiency, %"
set ytics 20
set xtics rotate by -45
set key center

plot [][0:110] \
"<awk '{t[$1] = $2; print $1 - 1, 100 * t[2]/$2}' bio/weak.multi.24" u 1:2:xtic(1) w lp pt 4 ps 2 lw 3 t "draws per core: 24"
