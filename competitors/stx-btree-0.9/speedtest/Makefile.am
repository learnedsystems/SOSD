# B+ Tree Speed Tests

if BUILD_SPEEDTEST

noinst_PROGRAMS = speedtest speedtest-tune

endif

speedtest_SOURCES = speedtest.cc

speedtest_tune_SOURCES = speedtest-tune.cc

AM_CXXFLAGS = -W -Wall -Wold-style-cast -DNDEBUG -I$(top_srcdir)/include

EXTRA_DIST = \
	speedtest.gnuplot \
	transpose.awk \
	results-2007/* \
	results-2011/* \
	results-2013/* \
	results-2013-delta/*

plots:
	USER=tb ./speedtest.gnuplot
	pdftoppm speedtest.pdf speedtest-plot
	for f in *.ppm; do convert $$f $${f%%.ppm}.png; rm $$f; done

transpose:
	for f in speed-*.txt; do ./transpose.awk $$f > $${f%%.txt}.trt; done
