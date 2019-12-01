/*
 * STX B+ Tree Test Suite v0.9
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

#include "tpunit.h"

#include <stdlib.h>
#include <time.h>

#include <stx/btree_multiset.h>
#include <set>

struct LargeTest : public tpunit::TestFixture
{
    LargeTest() : tpunit::TestFixture(
        TEST(LargeTest::test_320_mod_1000),
        TEST(LargeTest::test_320_mod_10000),
        TEST(LargeTest::test_3200_mod_10),
        TEST(LargeTest::test_3200_mod_100),
        TEST(LargeTest::test_3200_mod_1000),
        TEST(LargeTest::test_3200_mod_10000),
        TEST(LargeTest::test_32000_mod_10000),
        TEST(LargeTest::test_sequence)
        )
    {}

    template <typename KeyType>
    struct traits_nodebug : stx::btree_default_set_traits<KeyType>
    {
        static const bool       selfverify = true;
        static const bool       debug = false;

        static const int        leafslots = 8;
        static const int        innerslots = 8;
    };

    void test_multi(const unsigned int insnum, const unsigned int modulo)
    {
        typedef stx::btree_multiset<unsigned int,
            std::less<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        btree_type bt;

        typedef std::multiset<unsigned int> multiset_type;
        multiset_type set;

        // *** insert
        srand(34234235);
        for(unsigned int i = 0; i < insnum; i++)
        {
            unsigned int k = rand() % modulo;

            ASSERT( bt.size() == set.size() );
            bt.insert(k);
            set.insert(k);
            ASSERT( bt.count(k) == set.count(k) );

            ASSERT( bt.size() == set.size() );
        }

        ASSERT( bt.size() == insnum );

        // *** iterate
        btree_type::iterator bi = bt.begin();
        multiset_type::const_iterator si = set.begin();
        for(; bi != bt.end() && si != set.end(); ++bi, ++si)
        {
            ASSERT( *si == bi.key() );
        }
        ASSERT( bi == bt.end() );
        ASSERT( si == set.end() );

        // *** existance
        srand(34234235);
        for(unsigned int i = 0; i < insnum; i++)
        {
            unsigned int k = rand() % modulo;

            ASSERT( bt.exists(k) );
        }

        // *** counting
        srand(34234235);
        for(unsigned int i = 0; i < insnum; i++)
        {
            unsigned int k = rand() % modulo;

            ASSERT( bt.count(k) == set.count(k) );
        }

        // *** deletion
        srand(34234235);
        for(unsigned int i = 0; i < insnum; i++)
        {
            unsigned int k = rand() % modulo;

            if (set.find(k) != set.end())
            {
                ASSERT( bt.size() == set.size() );

                ASSERT( bt.exists(k) );
                ASSERT( bt.erase_one(k) );
                set.erase( set.find(k) );

                ASSERT( bt.size() == set.size() );
		ASSERT( std::equal(bt.begin(), bt.end(), set.begin()) );
            }
        }

        ASSERT( bt.empty() );
        ASSERT( set.empty() );
    }

    void test_320_mod_1000()
    {
        test_multi(320, 1000);
    }

    void test_320_mod_10000()
    {
        test_multi(320, 10000);
    }

    void test_3200_mod_10()
    {
        test_multi(3200, 10);
    }

    void test_3200_mod_100()
    {
        test_multi(3200, 100);
    }

    void test_3200_mod_1000()
    {
        test_multi(3200, 1000);
    }

    void test_3200_mod_10000()
    {
        test_multi(3200, 10000);
    }

    void test_32000_mod_10000()
    {
        test_multi(32000, 10000);
    }

    void test_sequence()
    {
        typedef stx::btree_multiset<unsigned int,
            std::less<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        btree_type bt;

        const unsigned int insnum = 10000;

        typedef std::multiset<unsigned int> multiset_type;
        multiset_type set;

        // *** insert
        srand(34234235);
        for(unsigned int i = 0; i < insnum; i++)
        {
            unsigned int k = i;

            ASSERT( bt.size() == set.size() );
            bt.insert(k);
            set.insert(k);
            ASSERT( bt.count(k) == set.count(k) );

            ASSERT( bt.size() == set.size() );
        }

        ASSERT( bt.size() == insnum );

        // *** iterate
        btree_type::iterator bi = bt.begin();
        multiset_type::const_iterator si = set.begin();
        for(; bi != bt.end() && si != set.end(); ++bi, ++si)
        {
            ASSERT( *si == bi.key() );
        }
        ASSERT( bi == bt.end() );
        ASSERT( si == set.end() );

        // *** existance
        srand(34234235);
        for(unsigned int i = 0; i < insnum; i++)
        {
            unsigned int k = i;

            ASSERT( bt.exists(k) );
        }

        // *** counting
        srand(34234235);
        for(unsigned int i = 0; i < insnum; i++)
        {
            unsigned int k = i;

            ASSERT( bt.count(k) == set.count(k) );
        }

        // *** deletion
        srand(34234235);
        for(unsigned int i = 0; i < insnum; i++)
        {
            unsigned int k = i;

            if (set.find(k) != set.end())
            {
                ASSERT( bt.size() == set.size() );

                ASSERT( bt.exists(k) );
                ASSERT( bt.erase_one(k) );
                set.erase( set.find(k) );

                ASSERT( bt.size() == set.size() );
		ASSERT( std::equal(bt.begin(), bt.end(), set.begin()) );
            }
        }

        ASSERT( bt.empty() );
        ASSERT( set.empty() );
    }

} __LargeTest;

