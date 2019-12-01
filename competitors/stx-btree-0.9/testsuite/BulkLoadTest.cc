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
#include <vector>

#include <stx/btree_multiset.h>
#include <stx/btree_multimap.h>
#include <stx/btree_map.h>

struct BulkLoadTest : public tpunit::TestFixture
{
    BulkLoadTest() : tpunit::TestFixture(
        TEST(BulkLoadTest::test_set),
        TEST(BulkLoadTest::test_map)
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

    void test_set_instance(size_t numkeys, unsigned int mod)
    {
        typedef stx::btree_multiset<unsigned int,
            std::less<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        std::vector<unsigned int> keys (numkeys);

        srand(34234235);
        for(unsigned int i = 0; i < numkeys; i++)
        {
            keys[i] = rand() % mod;
        }

        std::sort(keys.begin(), keys.end());

        btree_type bt;
        bt.bulk_load(keys.begin(), keys.end());

        unsigned int i = 0;
        for(btree_type::iterator it = bt.begin();
            it != bt.end(); ++it, ++i)
        {
            ASSERT( *it == keys[i]  );
        }
    }

    void test_set()
    {
        for (size_t n = 6; n < 3200; ++n)
            test_set_instance(n, 1000);

        test_set_instance(31996, 10000);
        test_set_instance(32000, 10000);
        test_set_instance(117649, 100000);
    }

    void test_map_instance(size_t numkeys, unsigned int mod)
    {
        typedef stx::btree_multimap<int, std::string,
            std::less<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        std::vector< std::pair<int,std::string> > pairs (numkeys);

        srand(34234235);
        for(unsigned int i = 0; i < numkeys; i++)
        {
            pairs[i].first = rand() % mod;
            pairs[i].second = "key";
        }

        std::sort(pairs.begin(), pairs.end());

        btree_type bt;
        bt.bulk_load(pairs.begin(), pairs.end());

        unsigned int i = 0;
        for(btree_type::iterator it = bt.begin();
            it != bt.end(); ++it, ++i)
        {
            ASSERT( *it == pairs[i]  );
        }
    }

    void test_map()
    {
        for (size_t n = 6; n < 3200; ++n)
            test_map_instance(n, 1000);

        test_map_instance(31996, 10000);
        test_map_instance(32000, 10000);
        test_map_instance(117649, 100000);
    }

} __BulkLoadTest;
