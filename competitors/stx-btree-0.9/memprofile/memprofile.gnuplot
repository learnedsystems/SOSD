#!/usr/bin/env gnuplot

set style line 1 linecolor rgbcolor "#FF0000" linewidth 1.6 pointsize 0.7
set style line 2 linecolor rgbcolor "#00C000" linewidth 1.6 pointsize 0.7
set style line 3 linecolor rgbcolor "#0000FF" linewidth 1.6 pointsize 0.7
set style line 4 linecolor rgbcolor "#E000E0" linewidth 1.6 pointsize 0.7
set style line 5 linecolor rgbcolor "#00C0FF" linewidth 1.6 pointsize 0.7
set style line 6 linecolor rgbcolor "#FFC000" linewidth 1.6 pointsize 0.7
set style increment user

set terminal pdf size 5, 3.5
set output 'memprofile.pdf'

### Measuring a Sequence of Insert Operations

### 1st Plot

set title "Memory Usage Profile - Insertion of 8192000 Integer Pairs"
set key top left
set xlabel "Program Execution Time [s]"
set ylabel "Memory Usage [MiB]"

plot "memprofile-stdmap.txt" using 1:($2 / 1024/1024) title "std::multimap" with lines, \
     "memprofile-hashmap.txt" using 1:($2 / 1024/1024) title "__gnu_cxx::hash_multimap" with lines, \
     "memprofile-unorderedmap.txt" using 1:($2 / 1024/1024) title "std::tr1::unordered_multimap" with lines, \
     "memprofile-btreemap.txt" using 1:($2 / 1024/1024) title "stx::btree_multimap" with lines, \
     "memprofile-vector.txt" using 1:($2 / 1024/1024) title "std::vector" with lines, \
     "memprofile-deque.txt" using 1:($2 / 1024/1024) title "std::deque" with lines