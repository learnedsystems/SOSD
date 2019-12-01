# B+ Tree Test Cases

noinst_PROGRAMS = testsuite

TESTS = testsuite

testsuite_SOURCES = tpunit.cc tpunit.h

testsuite_SOURCES += InstantiationTest.cc
testsuite_SOURCES += SimpleTest.cc
testsuite_SOURCES += LargeTest.cc
testsuite_SOURCES += BoundTest.cc
testsuite_SOURCES += IteratorTest.cc
testsuite_SOURCES += StructureTest.cc
testsuite_SOURCES += DumpRestoreTest.cc
testsuite_SOURCES += RelationTest.cc
testsuite_SOURCES += BulkLoadTest.cc

AM_CXXFLAGS = -W -Wall -Wold-style-cast -Wshadow -DBTREE_DEBUG -I$(top_srcdir)/include
