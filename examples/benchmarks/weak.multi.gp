set term pngcairo monochrome
set log x
set output "weak.multi.png"
set xlabel "logical cores for workers"
set ylabel "efficiency, %"
set key center
set ytics 20
set xtics rotate by -45

plot [:][0:] \
"<awk '{t[$1] = $2; print $1 - 1, 100 * t[2]/$2}' bio/weak.multi.24" u 1:2:xtic(1) w lp pt 4 ps 3 lw 3 t "draws per core: 24"
