set term pngcairo monochrome
set log x
set output "strong.multi.png"
set xlabel "logical cores for workers"
set ylabel "efficiency, %"
set key center left
set ytics 20
set xtics rotate by -45

plot [:][0:110] \
"<awk '{n[NR] = $1 - 1; t[NR] = $2; print n[NR], 100 * t[1]/t[NR] * n[1]/n[NR]}' bio/strong.multi.192" \
u 1:2:xtic(1) w lp pt 6 ps 3 lw 3 t "draws: 192"
