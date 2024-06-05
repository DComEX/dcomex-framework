set term pngcairo
set output "g.png"
set key left
set xlabel "time, days"
set ylabel "tumor volume, mm3"
set xtics 5
plot "<./c 0.125/*" w l lw 2 lc rgb "#aaaaaa" t "", \
     "0.125/1.74e-06,5.00e+00,1.26e+04,txt" t "mu = 5.00e+00" w l lw 2 lc rgb "#ff0000", \
     "0.125/1.74e-06,1.08e+01,1.26e+04,txt" t "mu = 1.08e+01" w l lw 2 lc rgb "#00ff00", \
     "0.125/1.74e-06,2.32e+01,1.26e+04,txt" t "mu = 2.32e+01" w l lw 2 lc rgb "#0000ff", \
     "0.125/1.74e-06,5.00e+01,1.26e+04,txt" t "mu = 5.00e+01" w l lw 2 lc rgb "#000000", \
