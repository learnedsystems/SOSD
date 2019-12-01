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

#include <stx/btree_multiset.h>

#include <assert.h>

// *** Settings

/// maximum number of items to insert
const unsigned int insertnum = 1024000 * 32;

const int randseed = 34234235;

/// b+ tree binsearch_threshold range to test
const int min_nodeslots = 564;
const int max_nodeslots = 564;

/// Time is measured using gettimeofday()
inline double timestamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 0.000001;
}

/// Traits used for the speed tests, BTREE_DEBUG is not defined.
template <typename KeyType, int _bs_slots>
struct btree_traits_speed : public stx::btree_default_set_traits<KeyType>
{
    static const int    binsearch_threshold = _bs_slots;
};

/// Test the B+ tree with a specific leaf/inner slots (only insert)
template <int Slots>
struct Test_BtreeSet_Insert
{
    typedef stx::btree_multiset<unsigned int, std::less<unsigned int>,
                                btree_traits_speed<unsigned int, Slots> > btree_type;

    Test_BtreeSet_Insert(unsigned int) {}

    void run(unsigned int insertnum)
    {
        btree_type bt;

        srand(randseed);
        for(unsigned int i = 0; i < insertnum; i++)
            bt.insert( rand() );

        assert( bt.size() == insertnum );
    }
};

/// Test the B+ tree with a specific leaf/inner slots (insert, find and delete)
template <int Slots>
struct Test_BtreeSet_InsertFindDelete
{
    typedef stx::btree_multiset<unsigned int, std::less<unsigned int>,
                                struct btree_traits_speed<unsigned int, Slots> > btree_type;

    Test_BtreeSet_InsertFindDelete(unsigned int) {}

    void run(unsigned int insertnum)
    {
        btree_type bt;

        srand(randseed);
        for(unsigned int i = 0; i < insertnum; i++)
            bt.insert(rand());

        assert( bt.size() == insertnum );

        srand(randseed);
        for(unsigned int i = 0; i < insertnum; i++)
            bt.exists(rand());

        srand(randseed);
        for(unsigned int i = 0; i < insertnum; i++)
            bt.erase_one(rand());

        assert(bt.empty());
    }
};

/// Test the B+ tree with a specific leaf/inner slots (find only)
template <int Slots>
struct Test_BtreeSet_Find
{
    typedef stx::btree_multiset<unsigned int, std::less<unsigned int>,
                                struct btree_traits_speed<unsigned int, Slots> > btree_type;

    btree_type bt;

    Test_BtreeSet_Find(unsigned int insertnum)
    {
        srand(randseed);
        for(unsigned int i = 0; i < insertnum; i++)
            bt.insert(rand());

        assert( bt.size() == insertnum );
    }

    void run(unsigned int insertnum)
    {
        srand(randseed);
        for(unsigned int i = 0; i < insertnum; i++)
            bt.exists(rand());
    }
};

unsigned int repeatuntil;

/// Repeat (short) tests until enough time elapsed and divide by the runs.
template <typename TestClass>
void testrunner_loop(std::ostream& os, unsigned int insertnum, unsigned int slots)
{
    unsigned int runs = 0;
    double ts1, ts2;

    do
    {
        runs = 0;	// count repetition of timed tests

        {
            TestClass test(insertnum);	// initialize test structures

            ts1 = timestamp();

            for(unsigned int totaltests = 0; totaltests <= repeatuntil; totaltests += insertnum)
            {
                test.run(insertnum);	// run timed test procedure
                ++runs;
            }

            ts2 = timestamp();
        }

        std::cerr << "insertnum=" << insertnum << " slots=" << slots << " repeat=" << (repeatuntil / insertnum) << " time=" << (ts2 - ts1) << "\n";

        // discard and repeat if test took less than one second.
        if ((ts2 - ts1) < 1.0) repeatuntil *= 2;
    }
    while ((ts2 - ts1) < 1.0);

    os << std::fixed << std::setprecision(10) << insertnum << " " << slots << " " << ((ts2 - ts1) / runs) << std::endl;
}

// Template magic to emulate a for_each slots. These templates will roll-out
// btree instantiations for each of the Low-High leaf/inner slot numbers.
template< template<int Slots> class functional, int Low, int High>
struct test_range
{
    inline void operator()(std::ostream& os, unsigned int insertnum)
    {
        testrunner_loop< functional<Low> >(os, insertnum, Low);
        test_range<functional, Low+8, High>()(os, insertnum);
    }
};

template< template<int Slots> class functional, int Low>
struct test_range<functional, Low, Low>
{
    inline void operator()(std::ostream& os, unsigned int insertnum)
    {
        testrunner_loop< functional<Low> >(os, insertnum, Low);
    }
};

/// Speed test them!
int main()
{
/*
    { // Set - speed test only insertion

        std::ofstream os("tune-set-insert.txt");

        std::cerr << "set: insert " << insertnum << "\n";
        repeatuntil = insertnum;

        test_range<Test_BtreeSet_Insert, min_nodeslots, max_nodeslots>()(os, insertnum);
    }

    { // Set - speed test insert, find and delete

        std::ofstream os("tune-set-all.txt");

        std::cerr << "set: insert, find, delete " << insertnum << "\n";
        repeatuntil = insertnum;

        test_range<Test_BtreeSet_InsertFindDelete, min_nodeslots, max_nodeslots>()(os, insertnum);
     }
*/
    { // Set - speed test find only

        std::ofstream os("tune-set-find.txt");

        std::cerr << "set: find " << insertnum << "\n";
        repeatuntil = insertnum;

        test_range<Test_BtreeSet_Find, min_nodeslots, max_nodeslots>()(os, insertnum);
    }

    return 0;
}
