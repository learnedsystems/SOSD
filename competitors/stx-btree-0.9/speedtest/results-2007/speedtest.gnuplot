#!/usr/bin/env gnuplot

set style line 1 linecolor rgbcolor "#FF0000" linewidth 1.6 pointsize 0.7
set style line 2 linecolor rgbcolor "#00FF00" linewidth 1.6 pointsize 0.7
set style line 3 linecolor rgbcolor "#0000FF" linewidth 1.6 pointsize 0.7
set style line 4 linecolor rgbcolor "#FF00FF" linewidth 1.6 pointsize 0.7
set style line 5 linecolor rgbcolor "#00FFFF" linewidth 1.6 pointsize 0.7
set style line 6 linecolor rgbcolor "#808080" linewidth 1.6 pointsize 0.7
set style line 7 linecolor rgbcolor "#D0D020" linewidth 1.6 pointsize 0.7
set style line 8 linecolor rgbcolor "#FF4C00" linewidth 1.6 pointsize 0.7
set style line 9 linecolor rgbcolor "#000000" linewidth 1.6 pointsize 0.7
set style increment user

set terminal pdf size 6, 4
set output 'speedtest.pdf'

set label "Pentium 4 3.2 GHz" at screen 0.04, screen 0.04

### 1st Plot

set title "Speed Test Multiset - Absolute Time - Insertion Only (125-8000 Items)"
set key top left
set logscale x
set xrange [100:10000]
set xlabel "Inserts"
set ylabel "Seconds"
set format x "%.0f"

plot "speed-insert.txt" using 1:2 title "std::multiset" with linespoints, \
     "speed-insert.txt" using 1:3 title " __gnu_cxx::hash_multiset" with linespoints, \
     "speed-insert.txt" using 1:4 title "stx::btree_multiset<4>" with linespoints,  \
     "speed-insert.txt" using 1:32 title "stx::btree_multiset<32>" with linespoints,  \
     "speed-insert.txt" using 1:64 title "stx::btree_multiset<64>" with linespoints, \
     "speed-insert.txt" using 1:128 title "stx::btree_multiset<128>" with linespoints, \
     "speed-insert.txt" using 1:200 title "stx::btree_multiset<200>" with linespoints

### 2nd Plot

set title "Speed Test Multiset - Absolute Time - Insertion Only (16000-4096000 Items)"

set xrange [10000:5000000]

replot

### 3rd Plot

set title "Speed Test Multiset - Normalized Time - Insertion Only (125-4096000 Items)"
set key top left
set logscale x
set xrange [100:5000000]
set xlabel "Inserts"
set ylabel "Microseconds / Insert"
set format x "%.0f"

plot "speed-insert.txt" using 1:($2 / $1) * 1000000 title "std::multiset" with linespoints, \
     "speed-insert.txt" using 1:($3 / $1) * 1000000 title " __gnu_cxx::hash_multiset" with linespoints, \
     "speed-insert.txt" using 1:($4 / $1) * 1000000 title "stx::btree_multiset<4>" with linespoints,  \
     "speed-insert.txt" using 1:($32 / $1) * 1000000 title "stx::btree_multiset<32>" with linespoints,  \
     "speed-insert.txt" using 1:($64 / $1) * 1000000 title "stx::btree_multiset<64>" with linespoints, \
     "speed-insert.txt" using 1:($128 / $1) * 1000000 title "stx::btree_multiset<128>" with linespoints, \
     "speed-insert.txt" using 1:($200 / $1) * 1000000 title "stx::btree_multiset<200>" with linespoints

### 4th Plot

set title "Speed Test - Finding the Best Slot Size - Insertion Only - Plotted by Leaf/Inner Slots in B+ Tree"

set key top right
set autoscale x
set xlabel "Leaf/Inner Slots"
set ylabel "Seconds"
unset logscale x
unset logscale y

plot "speed-insert.trt" using ($0 + 4):14 every ::2 title "1024000 Inserts" with lines, \
     "speed-insert.trt" using ($0 + 4):15 every ::2 title "2048000 Inserts" with lines, \
     "speed-insert.trt" using ($0 + 4):16 every ::2 title "4096000 Inserts" with lines

### Now Measuring a Sequence of Insert/Find/Erase Operations

### 1st Plot

set title "Speed Test Multiset - Insert/Find/Erase (125-8000 Items)"
set key top left
set logscale x
set xrange [100:10000]
set xlabel "Data Pairs"
set ylabel "Seconds"
set format x "%.0f"

plot "speed-all.txt" using 1:2 title "std::multiset" with linespoints, \
     "speed-all.txt" using 1:3 title " __gnu_cxx::hash_multiset" with linespoints, \
     "speed-all.txt" using 1:4 title "stx::btree_multiset<4>" with linespoints,  \
     "speed-all.txt" using 1:32 title "stx::btree_multiset<32>" with linespoints,  \
     "speed-all.txt" using 1:64 title "stx::btree_multiset<64>" with linespoints, \
     "speed-all.txt" using 1:128 title "stx::btree_multiset<128>" with linespoints, \
     "speed-all.txt" using 1:200 title "stx::btree_multiset<200>" with linespoints

### 2nd Plot

set title "Speed Test Multiset - Insert/Find/Erase (16000-4096000 Items)"

set xrange [10000:5000000]

replot

### 3rd Plot

set title "Speed Test Multiset - Normalized Time - Insert/Find/Erase (125-4096000 Items)"
set key top left
set logscale x
set xrange [100:5000000]
set xlabel "Items"
set ylabel "Microseconds / Item"
set format x "%.0f"

plot "speed-all.txt" using 1:($2 / $1) * 1000000 title "std::multiset" with linespoints, \
     "speed-all.txt" using 1:($3 / $1) * 1000000 title " __gnu_cxx::hash_multiset" with linespoints, \
     "speed-all.txt" using 1:($4 / $1) * 1000000 title "stx::btree_multiset<4>" with linespoints,  \
     "speed-all.txt" using 1:($32 / $1) * 1000000 title "stx::btree_multiset<32>" with linespoints,  \
     "speed-all.txt" using 1:($64 / $1) * 1000000 title "stx::btree_multiset<64>" with linespoints, \
     "speed-all.txt" using 1:($128 / $1) * 1000000 title "stx::btree_multiset<128>" with linespoints, \
     "speed-all.txt" using 1:($200 / $1) * 1000000 title "stx::btree_multiset<200>" with linespoints

### 4th Plot

set title "Speed Test - Finding the Best Slot Size - Insert/Find/Erase - Plotted by Leaf/Inner Slots in B+ Tree"

set key top right
set autoscale x
set xlabel "Leaf/Inner Slots"
set ylabel "Seconds"
unset logscale x
unset logscale y

plot "speed-all.trt" using ($0 + 4):14 every ::2 title "1024000 Data Pairs" with lines, \
     "speed-all.trt" using ($0 + 4):15 every ::2 title "2048000 Data Pairs" with lines, \
     "speed-all.trt" using ($0 + 4):16 every ::2 title "4096000 Data Pairs" with lines


### Now Measuring only Find Operations

### 1st Plot

set title "Speed Test Multiset - Find Only (125-8000 Items)"
set key top left
set logscale x
set xrange [100:10000]
set xlabel "Data Pairs"
set ylabel "Seconds"
set format x "%.0f"

plot "speed-find.txt" using 1:2 title "std::multiset" with linespoints, \
     "speed-find.txt" using 1:3 title " __gnu_cxx::hash_multiset" with linespoints, \
     "speed-find.txt" using 1:4 title "stx::btree_multiset<4>" with linespoints,  \
     "speed-find.txt" using 1:32 title "stx::btree_multiset<32>" with linespoints,  \
     "speed-find.txt" using 1:64 title "stx::btree_multiset<64>" with linespoints, \
     "speed-find.txt" using 1:128 title "stx::btree_multiset<128>" with linespoints, \
     "speed-find.txt" using 1:200 title "stx::btree_multiset<200>" with linespoints

### 2nd Plot

set title "Speed Test Multiset - Find Only (16000-4096000 Items)"

set xrange [10000:5000000]

replot

### 3rd Plot

set title "Speed Test Multiset - Normalized Time - Find Only (125-4096000 Items)"
set key top left
set logscale x
set xrange [100:5000000]
set xlabel "Items"
set ylabel "Microseconds / Item"
set format x "%.0f"

plot "speed-find.txt" using 1:($2 / $1) * 1000000 title "std::multiset" with linespoints, \
     "speed-find.txt" using 1:($3 / $1) * 1000000 title " __gnu_cxx::hash_multiset" with linespoints, \
     "speed-find.txt" using 1:($4 / $1) * 1000000 title "stx::btree_multiset<4>" with linespoints,  \
     "speed-find.txt" using 1:($32 / $1) * 1000000 title "stx::btree_multiset<32>" with linespoints,  \
     "speed-find.txt" using 1:($64 / $1) * 1000000 title "stx::btree_multiset<64>" with linespoints, \
     "speed-find.txt" using 1:($128 / $1) * 1000000 title "stx::btree_multiset<128>" with linespoints, \
     "speed-find.txt" using 1:($200 / $1) * 1000000 title "stx::btree_multiset<200>" with linespoints

### 4th Plot

set title "Speed Test - Finding the Best Slot Size - Find Only - Plotted by Leaf/Inner Slots in B+ Tree"

set key top right
set autoscale x
set xlabel "Leaf/Inner Slots"
set ylabel "Seconds"
unset logscale x
unset logscale y

plot "speed-find.trt" using ($0 + 4):14 every ::2 title "1024000 Data Pairs" with lines, \
     "speed-find.trt" using ($0 + 4):15 every ::2 title "2048000 Data Pairs" with lines, \
     "speed-find.trt" using ($0 + 4):16 every ::2 title "4096000 Data Pairs" with lines
