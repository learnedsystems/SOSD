/*
 * STX B+ Tree Speed Test Program v0.9
 * Copyright (C) 2008-2013 Timo Bingmann
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <string>
#include <stdlib.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <iomanip>

#include <set>
#include <ext/hash_set>
#include <tr1/unordered_set>
#include <stx/btree_multiset.h>

#include <map>
#include <ext/hash_map>
#include <tr1/unordered_map>
#include <stx/btree_multimap.h>

#include <assert.h>

// *** Settings

/// starting number of items to insert
const unsigned int minitems = 125;

/// maximum number of items to insert
const unsigned int maxitems = 1024000 * 64;

const int randseed = 34234235;

/// b+ tree slot range to test
const int min_nodeslots = 4;
const int max_nodeslots = 256;

/// Time is measured using gettimeofday()
inline double timestamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 0.000001;
}

/// Traits used for the speed tests, BTREE_DEBUG is not defined.
template <int _innerslots, int _leafslots>
struct btree_traits_speed : stx::btree_default_set_traits<unsigned int>
{
    static const bool   selfverify = false;
    static const bool   debug = false;

    static const int    leafslots = _innerslots;
    static const int    innerslots = _leafslots;

    static const size_t binsearch_threshold = 256*1024*1024; // never
};

// --------------------------------------------------------------------------------

/// Test a generic set type with insertions
template <typename SetType>
struct Test_Set_Insert
{
    Test_Set_Insert(unsigned int) {}

    inline void run(unsigned int items)
    {
        SetType set;

        srand(randseed);
        for(unsigned int i = 0; i < items; i++)
            set.insert( rand() );

        assert( set.size() == items );
    }
};

/// Test a generic set type with insert, find and delete sequences
template <typename SetType>
struct Test_Set_InsertFindDelete
{
    Test_Set_InsertFindDelete(unsigned int) {}

    inline void run(unsigned int items)
    {
        SetType set;

        srand(randseed);
        for(unsigned int i = 0; i < items; i++)
            set.insert( rand() );

        assert( set.size() == items );

        srand(randseed);
        for(unsigned int i = 0; i < items; i++)
            set.find(rand());

        srand(randseed);
        for(unsigned int i = 0; i < items; i++)
            set.erase( set.find(rand()) );

        assert( set.empty() );
    }
};

/// Test a generic set type with insert, find and delete sequences
template <typename SetType>
struct Test_Set_Find
{
    SetType set;

    Test_Set_Find(unsigned int items)
    {
        srand(randseed);
        for(unsigned int i = 0; i < items; i++)
            set.insert( rand() );

        assert( set.size() == items );
    }

    void run(unsigned int items)
    {
        srand(randseed);
        for(unsigned int i = 0; i < items; i++)
            set.find(rand());
    }
};

/// Construct different set types for a generic test class
template < template<typename SetType> class TestClass >
struct TestFactory_Set
{
    /// Test the multiset red-black tree from STL
    typedef TestClass< std::multiset<unsigned int> > StdSet;

    /// Test the multiset hash from gcc's STL extensions
    typedef TestClass< __gnu_cxx::hash_multiset<unsigned int> > HashSet;

    /// Test the unordered_set from STL TR1
    typedef TestClass< std::tr1::unordered_multiset<unsigned int> > UnorderedSet;

    /// Test the B+ tree with a specific leaf/inner slots
    template <int Slots>
    struct BtreeSet
        : TestClass< stx::btree_multiset<unsigned int, std::less<unsigned int>,
                                         struct btree_traits_speed<Slots, Slots> > >
    {
        BtreeSet(unsigned int n)
            : TestClass< stx::btree_multiset<unsigned int, std::less<unsigned int>,
                                             struct btree_traits_speed<Slots, Slots> > >(n) {}
    };

    /// Run tests on all set types
    void call_testrunner(std::ostream& os, unsigned int items);
};

// --------------------------------------------------------------------------------

/// Test a generic map type with insertions
template <typename MapType>
struct Test_Map_Insert
{
    Test_Map_Insert(unsigned int) {}

    inline void run(unsigned int items)
    {
        MapType map;

        srand(randseed);
        for(unsigned int i = 0; i < items; i++) {
            unsigned int r = rand();
            map.insert( std::make_pair(r,r) );
        }

        assert( map.size() == items );
    }
};

/// Test a generic map type with insert, find and delete sequences
template <typename MapType>
struct Test_Map_InsertFindDelete
{
    Test_Map_InsertFindDelete(unsigned int) {}

    inline void run(unsigned int items)
    {
        MapType map;

        srand(randseed);
        for(unsigned int i = 0; i < items; i++) {
            unsigned int r = rand();
            map.insert( std::make_pair(r,r) );
        }

        assert( map.size() == items );

        srand(randseed);
        for(unsigned int i = 0; i < items; i++)
            map.find(rand());

        srand(randseed);
        for(unsigned int i = 0; i < items; i++)
            map.erase( map.find(rand()) );

        assert( map.empty() );
    }
};

/// Test a generic map type with insert, find and delete sequences
template <typename MapType>
struct Test_Map_Find
{
    MapType map;

    Test_Map_Find(unsigned int items)
    {
        srand(randseed);
        for(unsigned int i = 0; i < items; i++) {
            unsigned int r = rand();
            map.insert( std::make_pair(r,r) );
        }

        assert( map.size() == items );
    }

    void run(unsigned int items)
    {
        srand(randseed);
        for(unsigned int i = 0; i < items; i++)
            map.find(rand());
    }
};

/// Construct different map types for a generic test class
template < template<typename MapType> class TestClass >
struct TestFactory_Map
{
    /// Test the multimap red-black tree from STL
    typedef TestClass< std::multimap<unsigned int, unsigned int> > StdMap;

    /// Test the multimap hash from gcc's STL extensions
    typedef TestClass< __gnu_cxx::hash_multimap<unsigned int, unsigned int> > HashMap;

    /// Test the unordered_map from STL TR1
    typedef TestClass< std::tr1::unordered_multimap<unsigned int, unsigned int> > UnorderedMap;

    /// Test the B+ tree with a specific leaf/inner slots
    template <int Slots>
    struct BtreeMap
        : TestClass< stx::btree_multimap<unsigned int, unsigned int, std::less<unsigned int>,
                                         struct btree_traits_speed<Slots, Slots> > >
    {
        BtreeMap(unsigned int n)
            : TestClass< stx::btree_multimap<unsigned int, unsigned int, std::less<unsigned int>,
                                             struct btree_traits_speed<Slots, Slots> > >(n) {}
    };

    /// Run tests on all map types
    void call_testrunner(std::ostream& os, unsigned int items);
};

// --------------------------------------------------------------------------------

unsigned int repeatuntil;

/// Repeat (short) tests until enough time elapsed and divide by the runs.
template <typename TestClass>
void testrunner_loop(std::ostream& os, unsigned int items)
{
    unsigned int runs = 0;
    double ts1, ts2;

    do
    {
        runs = 0;	// count repetition of timed tests

        {
            TestClass test(items);	// initialize test structures

            ts1 = timestamp();

            for(unsigned int totaltests = 0; totaltests <= repeatuntil; totaltests += items)
            {
                test.run(items);	// run timed test procedure
                ++runs;
            }

            ts2 = timestamp();
        }

        std::cerr << "Insert " << items << " repeat " << (repeatuntil / items)
                  << " time " << (ts2 - ts1) << "\n";

        // discard and repeat if test took less than one second.
        if ((ts2 - ts1) < 1.0) repeatuntil *= 2;
    }
    while ((ts2 - ts1) < 1.0);

    os << std::fixed << std::setprecision(10) << ((ts2 - ts1) / runs) << " " << std::flush;
}

// Template magic to emulate a for_each slots. These templates will roll-out
// btree instantiations for each of the Low-High leaf/inner slot numbers.
template< template<int Slots> class functional, int Low, int High>
struct btree_range
{
    inline void operator()(std::ostream& os, unsigned int items)
    {
        testrunner_loop< functional<Low> >(os, items);
        btree_range<functional, Low+2, High>()(os, items);
    }
};

template< template<int Slots> class functional, int Low>
struct btree_range<functional, Low, Low>
{
    inline void operator()(std::ostream& os, unsigned int items)
    {
        testrunner_loop< functional<Low> >(os, items);
    }
};

template < template<typename Type> class TestClass >
void TestFactory_Set<TestClass>::call_testrunner(std::ostream& os, unsigned int items)
{
    os << items << " " << std::flush;

    testrunner_loop<StdSet>(os, items);
    testrunner_loop<HashSet>(os, items);
    testrunner_loop<UnorderedSet>(os, items);

#if 1
    btree_range<BtreeSet, min_nodeslots, max_nodeslots>()(os, items);
#else
    // just pick a few node sizes for quicker tests
    testrunner_loop< BtreeSet<4> >(os, items);
    for (int i = 6; i < 8; i+=2) os << "0 ";
    testrunner_loop< BtreeSet<8> >(os, items);
    for (int i = 10; i < 16; i+=2) os << "0 ";
    testrunner_loop< BtreeSet<16> >(os, items);
    for (int i = 18; i < 32; i+=2) os << "0 ";
    testrunner_loop< BtreeSet<32> >(os, items);
    for (int i = 34; i < 64; i+=2) os << "0 ";
    testrunner_loop< BtreeSet<64> >(os, items);
    for (int i = 66; i < 128; i+=2) os << "0 ";
    testrunner_loop< BtreeSet<128> >(os, items);
    for (int i = 130; i < 256; i+=2) os << "0 ";
    testrunner_loop< BtreeSet<256> >(os, items);
#endif

    os << "\n" << std::flush;
}

template < template<typename Type> class TestClass >
void TestFactory_Map<TestClass>::call_testrunner(std::ostream& os, unsigned int items)
{
    os << items << " " << std::flush;

    testrunner_loop<StdMap>(os, items);
    testrunner_loop<HashMap>(os, items);
    testrunner_loop<UnorderedMap>(os, items);

#if 1
    btree_range<BtreeMap, min_nodeslots, max_nodeslots>()(os, items);
#else
    // just pick a few node sizes for quicker tests
    testrunner_loop< BtreeMap<4> >(os, items);
    for (int i = 6; i < 8; i+=2) os << "0 ";
    testrunner_loop< BtreeMap<8> >(os, items);
    for (int i = 10; i < 16; i+=2) os << "0 ";
    testrunner_loop< BtreeMap<16> >(os, items);
    for (int i = 18; i < 32; i+=2) os << "0 ";
    testrunner_loop< BtreeMap<32> >(os, items);
    for (int i = 34; i < 64; i+=2) os << "0 ";
    testrunner_loop< BtreeMap<64> >(os, items);
    for (int i = 66; i < 128; i+=2) os << "0 ";
    testrunner_loop< BtreeMap<128> >(os, items);
    for (int i = 130; i < 256; i+=2) os << "0 ";
    testrunner_loop< BtreeMap<256> >(os, items);
#endif

    os << "\n" << std::flush;
}

/// Speed test them!
int main()
{
    { // Set - speed test only insertion

        std::ofstream os("speed-set-insert.txt");

        repeatuntil = minitems;

        for(unsigned int items = minitems; items <= maxitems; items *= 2)
        {
            std::cerr << "set: insert " << items << "\n";
            TestFactory_Set<Test_Set_Insert>().call_testrunner(os, items);
        }
    }

    { // Set - speed test insert, find and delete

        std::ofstream os("speed-set-all.txt");

        repeatuntil = minitems;

        for(unsigned int items = minitems; items <= maxitems; items *= 2)
        {
            std::cerr << "set: insert, find, delete " << items << "\n";
            TestFactory_Set<Test_Set_InsertFindDelete>().call_testrunner(os, items);
        }
    }

    { // Set - speed test find only

        std::ofstream os("speed-set-find.txt");

        repeatuntil = minitems;

        for(unsigned int items = minitems; items <= maxitems; items *= 2)
        {
            std::cerr << "set: find " << items << "\n";
            TestFactory_Set<Test_Set_Find>().call_testrunner(os, items);
        }
    }

    { // Map - speed test only insertion

        std::ofstream os("speed-map-insert.txt");

        repeatuntil = minitems;

        for(unsigned int items = minitems; items <= maxitems; items *= 2)
        {
            std::cerr << "map: insert " << items << "\n";
            TestFactory_Map<Test_Map_Insert>().call_testrunner(os, items);
        }
    }

    { // Map - speed test insert, find and delete

        std::ofstream os("speed-map-all.txt");

        repeatuntil = minitems;

        for(unsigned int items = minitems; items <= maxitems; items *= 2)
        {
            std::cerr << "map: insert, find, delete " << items << "\n";
            TestFactory_Map<Test_Map_InsertFindDelete>().call_testrunner(os, items);
        }
    }

    { // Map - speed test find only

        std::ofstream os("speed-map-find.txt");

        repeatuntil = minitems;

        for(unsigned int items = minitems; items <= maxitems; items *= 2)
        {
            std::cerr << "map: find " << items << "\n";
            TestFactory_Map<Test_Map_Find>().call_testrunner(os, items);
        }
    }

    return 0;
}
