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
#include <stx/btree_set.h>

struct IteratorTest : public tpunit::TestFixture
{
    IteratorTest() : tpunit::TestFixture(
        TEST(IteratorTest::test_iterator1),
        TEST(IteratorTest::test_iterator2),
        TEST(IteratorTest::test_iterator3),
        TEST(IteratorTest::test_iterator4),
        TEST(IteratorTest::test_iterator5),
        TEST(IteratorTest::test_erase_iterator1)
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

    void test_iterator1()
    {
        typedef stx::btree_multiset<unsigned int,
            std::less<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        std::vector<unsigned int> vector;

        srand(34234235);
        for(unsigned int i = 0; i < 3200; i++)
        {
            vector.push_back( rand() % 1000 );
        }

        ASSERT( vector.size() == 3200 );

        // test construction and insert(iter, iter) function
        btree_type bt(vector.begin(), vector.end());

        ASSERT( bt.size() == 3200 );

        // copy for later use
        btree_type bt2 = bt;

        // empty out the first bt
        srand(34234235);
        for(unsigned int i = 0; i < 3200; i++)
        {
            ASSERT(bt.size() == 3200 - i);
            ASSERT( bt.erase_one(rand() % 1000) );
            ASSERT(bt.size() == 3200 - i - 1);
        }

        ASSERT( bt.empty() );

        // copy btree values back to a vector

        std::vector<unsigned int> vector2;
        vector2.assign( bt2.begin(), bt2.end() );

        // afer sorting the vector, the two must be the same
        std::sort(vector.begin(), vector.end());

        ASSERT( vector == vector2 );

        // test reverse iterator
        vector2.clear();
        vector2.assign( bt2.rbegin(), bt2.rend() );

        std::reverse(vector.begin(), vector.end());

        btree_type::reverse_iterator ri = bt2.rbegin();
        for(unsigned int i = 0; i < vector2.size(); ++i)
        {
            ASSERT( vector[i] == vector2[i] );
            ASSERT( vector[i] == *ri );

            ri++;
        }

        ASSERT( ri == bt2.rend() );
    }

    void test_iterator2()
    {
        typedef stx::btree_multimap<unsigned int, unsigned int,
            std::less<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        std::vector< btree_type::value_type > vector;

        srand(34234235);
        for(unsigned int i = 0; i < 3200; i++)
        {
            vector.push_back( btree_type::value_type(rand() % 1000, 0) );
        }

        ASSERT( vector.size() == 3200 );

        // test construction and insert(iter, iter) function
        btree_type bt(vector.begin(), vector.end());

        ASSERT( bt.size() == 3200 );

        // copy for later use
        btree_type bt2 = bt;

        // empty out the first bt
        srand(34234235);
        for(unsigned int i = 0; i < 3200; i++)
        {
            ASSERT(bt.size() == 3200 - i);
            ASSERT( bt.erase_one(rand() % 1000) );
            ASSERT(bt.size() == 3200 - i - 1);
        }

        ASSERT( bt.empty() );

        // copy btree values back to a vector

        std::vector< btree_type::value_type > vector2;
        vector2.assign( bt2.begin(), bt2.end() );

        // afer sorting the vector, the two must be the same
        std::sort(vector.begin(), vector.end());

        ASSERT( vector == vector2 );

        // test reverse iterator
        vector2.clear();
        vector2.assign( bt2.rbegin(), bt2.rend() );

        std::reverse(vector.begin(), vector.end());

        btree_type::reverse_iterator ri = bt2.rbegin();
        for(unsigned int i = 0; i < vector2.size(); ++i, ++ri)
        {
            ASSERT( vector[i].first == vector2[i].first );
            ASSERT( vector[i].first == ri->first );
            ASSERT( vector[i].second == ri->second );
        }

        ASSERT( ri == bt2.rend() );
    }

    void test_iterator3()
    {
        typedef stx::btree_map<unsigned int, unsigned int,
            std::less<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        btree_type map;

        unsigned int maxnum = 1000;

        for(unsigned int i = 0; i < maxnum; ++i)
        {
            map.insert( std::make_pair(i, i*3) );
        }

        { // test iterator prefix++
            unsigned int nownum = 0;

            for(btree_type::iterator i = map.begin();
                i != map.end(); ++i)
            {
                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );

                nownum++;
            }

            ASSERT(nownum == maxnum);
        }

        { // test iterator prefix--
            unsigned int nownum = maxnum;

            btree_type::iterator i;
            for(i = --map.end(); i != map.begin(); --i)
            {
                nownum--;

                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );
            }

            nownum--;

            ASSERT( nownum == i->first );
            ASSERT( nownum * 3 == i->second );

            ASSERT(nownum == 0);
        }

        { // test const_iterator prefix++
            unsigned int nownum = 0;

            for(btree_type::const_iterator i = map.begin();
                i != map.end(); ++i)
            {
                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );

                nownum++;
            }

            ASSERT(nownum == maxnum);
        }

        { // test const_iterator prefix--
            unsigned int nownum = maxnum;

            btree_type::const_iterator i;
            for(i = --map.end(); i != map.begin(); --i)
            {
                nownum--;

                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );
            }

            nownum--;

            ASSERT( nownum == i->first );
            ASSERT( nownum * 3 == i->second );

            ASSERT(nownum == 0);
        }

        { // test reverse_iterator prefix++
            unsigned int nownum = maxnum;

            for(btree_type::reverse_iterator i = map.rbegin();
                i != map.rend(); ++i)
            {
                nownum--;

                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );
            }

            ASSERT(nownum == 0);
        }

        { // test reverse_iterator prefix--
            unsigned int nownum = 0;

            btree_type::reverse_iterator i;
            for(i = --map.rend(); i != map.rbegin(); --i)
            {
                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );

                nownum++;
            }

            ASSERT( nownum == i->first );
            ASSERT( nownum * 3 == i->second );

            nownum++;

            ASSERT(nownum == maxnum);
        }

        { // test const_reverse_iterator prefix++
            unsigned int nownum = maxnum;

            for(btree_type::const_reverse_iterator i = map.rbegin();
                i != map.rend(); ++i)
            {
                nownum--;

                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );
            }

            ASSERT(nownum == 0);
        }

        { // test const_reverse_iterator prefix--
            unsigned int nownum = 0;

            btree_type::const_reverse_iterator i;
            for(i = --map.rend(); i != map.rbegin(); --i)
            {
                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );

                nownum++;
            }

            ASSERT( nownum == i->first );
            ASSERT( nownum * 3 == i->second );

            nownum++;

            ASSERT(nownum == maxnum);
        }

        // postfix

        { // test iterator postfix++
            unsigned int nownum = 0;

            for(btree_type::iterator i = map.begin();
                i != map.end(); i++)
            {
                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );

                nownum++;
            }

            ASSERT(nownum == maxnum);
        }

        { // test iterator postfix--
            unsigned int nownum = maxnum;

            btree_type::iterator i;
            for(i = --map.end(); i != map.begin(); i--)
            {
                nownum--;

                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );
            }

            nownum--;

            ASSERT( nownum == i->first );
            ASSERT( nownum * 3 == i->second );

            ASSERT(nownum == 0);
        }

        { // test const_iterator postfix++
            unsigned int nownum = 0;

            for(btree_type::const_iterator i = map.begin();
                i != map.end(); i++)
            {
                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );

                nownum++;
            }

            ASSERT(nownum == maxnum);
        }

        { // test const_iterator postfix--
            unsigned int nownum = maxnum;

            btree_type::const_iterator i;
            for(i = --map.end(); i != map.begin(); i--)
            {
                nownum--;

                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );
            }

            nownum--;

            ASSERT( nownum == i->first );
            ASSERT( nownum * 3 == i->second );

            ASSERT(nownum == 0);
        }

        { // test reverse_iterator postfix++
            unsigned int nownum = maxnum;

            for(btree_type::reverse_iterator i = map.rbegin();
                i != map.rend(); i++)
            {
                nownum--;

                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );
            }

            ASSERT(nownum == 0);
        }

        { // test reverse_iterator postfix--
            unsigned int nownum = 0;

            btree_type::reverse_iterator i;
            for(i = --map.rend(); i != map.rbegin(); i--)
            {
                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );

                nownum++;
            }

            ASSERT( nownum == i->first );
            ASSERT( nownum * 3 == i->second );

            nownum++;

            ASSERT(nownum == maxnum);
        }

        { // test const_reverse_iterator postfix++
            unsigned int nownum = maxnum;

            for(btree_type::const_reverse_iterator i = map.rbegin();
                i != map.rend(); i++)
            {
                nownum--;

                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );
            }

            ASSERT(nownum == 0);
        }

        { // test const_reverse_iterator postfix--
            unsigned int nownum = 0;

            btree_type::const_reverse_iterator i;
            for(i = --map.rend(); i != map.rbegin(); i--)
            {
                ASSERT( nownum == i->first );
                ASSERT( nownum * 3 == i->second );

                nownum++;
            }

            ASSERT( nownum == i->first );
            ASSERT( nownum * 3 == i->second );

            nownum++;

            ASSERT(nownum == maxnum);
        }
    }

    void test_iterator4()
    {
        typedef stx::btree_set<unsigned int,
            std::less<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        btree_type set;

        unsigned int maxnum = 1000;

        for(unsigned int i = 0; i < maxnum; ++i)
        {
            set.insert(i);
        }

        { // test iterator prefix++
            unsigned int nownum = 0;

            for(btree_type::iterator i = set.begin();
                i != set.end(); ++i)
            {
                ASSERT( nownum == *i );
                nownum++;
            }

            ASSERT(nownum == maxnum);
        }

        { // test iterator prefix--
            unsigned int nownum = maxnum;

            btree_type::iterator i;
            for(i = --set.end(); i != set.begin(); --i)
            {
                ASSERT( --nownum == *i );
            }

            ASSERT( --nownum == *i );

            ASSERT(nownum == 0);
        }

        { // test const_iterator prefix++
            unsigned int nownum = 0;

            for(btree_type::const_iterator i = set.begin();
                i != set.end(); ++i)
            {
                ASSERT( nownum++ == *i );
            }

            ASSERT(nownum == maxnum);
        }

        { // test const_iterator prefix--
            unsigned int nownum = maxnum;

            btree_type::const_iterator i;
            for(i = --set.end(); i != set.begin(); --i)
            {
                ASSERT( --nownum == *i );
            }

            ASSERT( --nownum == *i );

            ASSERT(nownum == 0);
        }

        { // test reverse_iterator prefix++
            unsigned int nownum = maxnum;

            for(btree_type::reverse_iterator i = set.rbegin();
                i != set.rend(); ++i)
            {
                ASSERT( --nownum == *i );
            }

            ASSERT(nownum == 0);
        }

        { // test reverse_iterator prefix--
            unsigned int nownum = 0;

            btree_type::reverse_iterator i;
            for(i = --set.rend(); i != set.rbegin(); --i)
            {
                ASSERT( nownum++ == *i );
            }

            ASSERT( nownum++ == *i );

            ASSERT(nownum == maxnum);
        }

        { // test const_reverse_iterator prefix++
            unsigned int nownum = maxnum;

            for(btree_type::const_reverse_iterator i = set.rbegin();
                i != set.rend(); ++i)
            {
                ASSERT( --nownum == *i );
            }

            ASSERT(nownum == 0);
        }

        { // test const_reverse_iterator prefix--
            unsigned int nownum = 0;

            btree_type::const_reverse_iterator i;
            for(i = --set.rend(); i != set.rbegin(); --i)
            {
                ASSERT( nownum++ == *i );
            }

            ASSERT( nownum++ == *i );

            ASSERT(nownum == maxnum);
        }

        // postfix

        { // test iterator postfix++
            unsigned int nownum = 0;

            for(btree_type::iterator i = set.begin();
                i != set.end(); i++)
            {
                ASSERT( nownum++ == *i );
            }

            ASSERT(nownum == maxnum);
        }

        { // test iterator postfix--
            unsigned int nownum = maxnum;

            btree_type::iterator i;
            for(i = --set.end(); i != set.begin(); i--)
            {

                ASSERT( --nownum == *i );
            }

            ASSERT( --nownum == *i );

            ASSERT(nownum == 0);
        }

        { // test const_iterator postfix++
            unsigned int nownum = 0;

            for(btree_type::const_iterator i = set.begin();
                i != set.end(); i++)
            {
                ASSERT( nownum++ == *i );
            }

            ASSERT(nownum == maxnum);
        }

        { // test const_iterator postfix--
            unsigned int nownum = maxnum;

            btree_type::const_iterator i;
            for(i = --set.end(); i != set.begin(); i--)
            {
                ASSERT( --nownum == *i );
            }

            ASSERT( --nownum == *i );

            ASSERT(nownum == 0);
        }

        { // test reverse_iterator postfix++
            unsigned int nownum = maxnum;

            for(btree_type::reverse_iterator i = set.rbegin();
                i != set.rend(); i++)
            {
                ASSERT( --nownum == *i );
            }

            ASSERT(nownum == 0);
        }

        { // test reverse_iterator postfix--
            unsigned int nownum = 0;

            btree_type::reverse_iterator i;
            for(i = --set.rend(); i != set.rbegin(); i--)
            {
                ASSERT( nownum++ == *i );
            }

            ASSERT( nownum++ == *i );

            ASSERT(nownum == maxnum);
        }

        { // test const_reverse_iterator postfix++
            unsigned int nownum = maxnum;

            for(btree_type::const_reverse_iterator i = set.rbegin();
                i != set.rend(); i++)
            {
                ASSERT( --nownum == *i );
            }

            ASSERT(nownum == 0);
        }

        { // test const_reverse_iterator postfix--
            unsigned int nownum = 0;

            btree_type::const_reverse_iterator i;
            for(i = --set.rend(); i != set.rbegin(); i--)
            {
                ASSERT( nownum++ == *i );
            }

            ASSERT( nownum++ == *i );

            ASSERT(nownum == maxnum);
        }
    }

    void test_iterator5()
    {
        typedef stx::btree_set<unsigned int,
            std::less<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        btree_type set;

        unsigned int maxnum = 100;

        for(unsigned int i = 0; i < maxnum; ++i)
        {
            set.insert(i);
        }

        {
            btree_type::iterator it;

            it = set.begin();
            it--;
            ASSERT( it == set.begin() );

            it = set.begin();
            --it;
            ASSERT( it == set.begin() );

            it = set.end();
            it++;
            ASSERT( it == set.end() );

            it = set.end();
            ++it;
            ASSERT( it == set.end() );
        }

        {
            btree_type::const_iterator it;

            it = set.begin();
            it--;
            ASSERT( it == set.begin() );

            it = set.begin();
            --it;
            ASSERT( it == set.begin() );

            it = set.end();
            it++;
            ASSERT( it == set.end() );

            it = set.end();
            ++it;
            ASSERT( it == set.end() );
        }

        {
            btree_type::reverse_iterator it;

            it = set.rbegin();
            it--;
            ASSERT( it == set.rbegin() );

            it = set.rbegin();
            --it;
            ASSERT( it == set.rbegin() );

            it = set.rend();
            it++;
            ASSERT( it == set.rend() );

            it = set.rend();
            ++it;
            ASSERT( it == set.rend() );
        }

        {
            btree_type::const_reverse_iterator it;

            it = set.rbegin();
            it--;
            ASSERT( it == set.rbegin() );

            it = set.rbegin();
            --it;
            ASSERT( it == set.rbegin() );

            it = set.rend();
            it++;
            ASSERT( it == set.rend() );

            it = set.rend();
            ++it;
            ASSERT( it == set.rend() );
        }
    }

    void test_erase_iterator1()
    {
        typedef stx::btree_multimap<int, int,
	    std::less<int>, traits_nodebug<unsigned int> > btree_type;

        btree_type map;

	const int size1 = 32; 
	const int size2 = 256; 

	for (int i = 0; i < size1; ++i)
	{
	    for (int j = 0; j < size2; ++j)
	    {
		map.insert2(i,j);
	    }
	}

	ASSERT( map.size() == size1 * size2 );

	// erase in reverse order. that should be the worst case for
	// erase_iter()

	for (int i = size1-1; i >= 0; --i)
	{
	    for (int j = size2-1; j >= 0; --j)
	    {
		// find iterator
		btree_type::iterator it = map.find(i);
		
		while (it != map.end() && it.key() == i && it.data() != j)
		    ++it;

		ASSERT( it.key() == i );
		ASSERT( it.data() == j );

		unsigned int mapsize = map.size();
		map.erase(it);
		ASSERT( map.size() == mapsize - 1 );
	    }
	}

	ASSERT( map.size() == 0 );
    }

} __IteratorTest;
