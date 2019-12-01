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

#include <stx/btree_multiset.h>

struct RelationTest : public tpunit::TestFixture
{
    RelationTest() : tpunit::TestFixture(
        TEST(RelationTest::test_relations)
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

    void test_relations()
    {
        typedef stx::btree_multiset<unsigned int,
            std::less<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        btree_type bt1, bt2;

        srand(34234236);
        for(unsigned int i = 0; i < 320; i++)
        {
            unsigned int key = rand() % 1000;

            bt1.insert(key);
            bt2.insert(key);
        }

        ASSERT( bt1 == bt2 );

        bt1.insert(499);
        bt2.insert(500);

        ASSERT( bt1 != bt2 );
        ASSERT( bt1 < bt2 );
        ASSERT( !(bt1 > bt2) );

        bt1.insert(500);
        bt2.insert(499);

        ASSERT( bt1 == bt2 );
        ASSERT( bt1 <= bt2 );

        // test assignment operator
        btree_type bt3;

        bt3 = bt1;
        ASSERT( bt1 == bt3 );
        ASSERT( bt1 >= bt3 );

        // test copy constructor
        btree_type bt4 = bt3;

        ASSERT( bt1 == bt4 );
    }

} __RelationTest;
