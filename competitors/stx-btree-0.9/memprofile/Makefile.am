# B+ Tree Memory Usage Profile

if BUILD_SPEEDTEST

noinst_PROGRAMS = memprofile

endif

memprofile_SOURCES = main.cc malloc_count.c malloc_count.h memprofile.h

AM_CXXFLAGS = -W -Wall -Wold-style-cast -I$(top_srcdir)/include
LIBS = -ldl

EXTRA_DIST = \
	memprofile.gnuplot \
	results-2013/*
