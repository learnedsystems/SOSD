# wxbtreedemo/Makefile.am

if HAVE_WXWIDGETS

bin_PROGRAMS = wxBTreeDemo

wxBTreeDemo_SOURCES = WMain.cpp WMain.h \
	WMain_wxg.cpp WMain_wxg.h \
	WTreeDrawing.h WTreeDrawing.cpp

AM_CFLAGS = -I/usr/include
AM_CXXFLAGS = -I/usr/include @WX_CXXFLAGS@ -I$(top_srcdir)/include -DBTREE_DEBUG
LDADD = @WX_LIBS@

if GOT_RESCOMP
resources.o: resources.rc
	$(RESCOMP) -I$(srcdir) $(srcdir)/resources.rc resources.o

LDADD += resources.o
endif

EXTRA_DIST = progicon.ico progicon.xpm progicon.xcf \
	resources.rc wxBTreeDemo.wxg manifest.xml

endif
