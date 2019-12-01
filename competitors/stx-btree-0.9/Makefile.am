# Makefile.am

SUBDIRS = . include testsuite speedtest memprofile wxbtreedemo

AM_CFLAGS = -W -Wall -I$(srcdir)/include
AM_CXXFLAGS = -W -Wall -Wold-style-cast -DBTREE_DEBUG -I$(srcdir)/include

EXTRA_DIST = Doxyfile \
	COPYING.GPLv3 \
	doxygen-html/*

if GCOV

clean-local:
	find -name "*.da" -o -name "*.gcov" -o -name "*.gcda" -o -name "*.gcno" | xargs rm || true

run-gcov: clean-gcov
	mkdir -p coverage/
	lcov --directory . --zerocounters
	-$(MAKE) check
	lcov --directory . --capture --output-file coverage/all.info
	lcov -e coverage/all.info "/home/*" -o coverage/testsuite.info
	genhtml -o coverage --num-spaces 8 coverage/testsuite.info --title "STX B+ Tree Testsuite" --prefix `pwd` --legend

clean-gcov:
	find -name "*.gcda" | xargs rm || true

endif
