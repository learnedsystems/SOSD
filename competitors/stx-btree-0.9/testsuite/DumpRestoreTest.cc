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

#include <sstream>
#include <iostream>

#include <stx/btree_multiset.h>

struct DumpRestoreTest : public tpunit::TestFixture
{
    DumpRestoreTest() : tpunit::TestFixture(
        TEST(DumpRestoreTest::test_dump_restore_3200)
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

    void test_dump_restore_3200()
    {
        typedef stx::btree_multiset<unsigned int,
            std::less<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        std::string dumpstr;

        {
            btree_type bt;

            srand(34234235);
            for(unsigned int i = 0; i < 3200; i++)
            {
                bt.insert(rand() % 100);
            }

            ASSERT(bt.size() == 3200);

            std::ostringstream os;
            bt.dump(os);

            dumpstr = os.str();
        }

	// Also cannot check the length, because it depends on the rand()
	// algorithm in stdlib.
	// ASSERT( dumpstr.size() == 47772 );

        // cannot check the string with a hash function, because it contains
        // memory pointers

        { // restore the btree image
            btree_type bt2;

	    std::istringstream iss(dumpstr);
	    ASSERT( bt2.restore(iss) );

	    ASSERT( bt2.size() == 3200 );

	    srand(34234235);
	    for(unsigned int i = 0; i < 3200; i++)
	    {
		ASSERT( bt2.exists(rand() % 100) );
	    }
	}

	{ // try restore the btree image using a different instantiation

	    typedef stx::btree_multiset<long long,
		std::less<long long>, traits_nodebug<long long> > otherbtree_type;

	    otherbtree_type bt3;

	    std::istringstream iss(dumpstr);
	    ASSERT( !bt3.restore(iss) );
	}
    }

} __DumpRestoreTest;
