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
#include <inttypes.h>

#include <stx/btree_multiset.h>
#include <stx/btree_multimap.h>
#include <stx/btree_map.h>

template <int Slots>
struct SimpleTest : public tpunit::TestFixture
{
    SimpleTest() : tpunit::TestFixture(
        TEST(SimpleTest::test_empty),
        TEST(SimpleTest::test_set_insert_erase_3200),
        TEST(SimpleTest::test_set_insert_erase_3200_descending),
        TEST(SimpleTest::test_map_insert_erase_3200),
        TEST(SimpleTest::test_map_insert_erase_3200_descending),
        TEST(SimpleTest::test2_map_insert_erase_strings),
        TEST(SimpleTest::test_set_100000_uint64),
        TEST(SimpleTest::test_multiset_100000_uint32)
        )
    {}

    template <typename KeyType>
    struct traits_nodebug : stx::btree_default_set_traits<KeyType>
    {
        static const bool       selfverify = true;
        static const bool       debug = false;

        static const int        leafslots = Slots;
        static const int        innerslots = Slots;
    };

    void test_empty()
    {
        typedef stx::btree_multiset<unsigned int,
            std::less<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        btree_type bt, bt2;
        bt.verify();

        ASSERT( bt.erase(42) == false );

        ASSERT( bt == bt2 );
    }

    void test_set_insert_erase_3200()
    {
        typedef stx::btree_multiset<unsigned int,
            std::less<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        btree_type bt;
        bt.verify();

        srand(34234235);
        for(unsigned int i = 0; i < 3200; i++)
        {
            ASSERT(bt.size() == i);
            bt.insert(rand() % 100);
            ASSERT(bt.size() == i + 1);
        }

        srand(34234235);
        for(unsigned int i = 0; i < 3200; i++)
        {
            ASSERT(bt.size() == 3200 - i);
            ASSERT( bt.erase_one(rand() % 100) );
            ASSERT(bt.size() == 3200 - i - 1);
        }

        ASSERT( bt.empty() );
    }

    void test_set_insert_erase_3200_descending()
    {
        typedef stx::btree_multiset<unsigned int,
            std::greater<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        btree_type bt;

        srand(34234235);
        for(unsigned int i = 0; i < 3200; i++)
        {
            ASSERT(bt.size() == i);
            bt.insert(rand() % 100);
            ASSERT(bt.size() == i + 1);
        }

        srand(34234235);
        for(unsigned int i = 0; i < 3200; i++)
        {
            ASSERT(bt.size() == 3200 - i);
            ASSERT( bt.erase_one(rand() % 100) );
            ASSERT(bt.size() == 3200 - i - 1);
        }

        ASSERT( bt.empty() );
    }

    void test_map_insert_erase_3200()
    {
        typedef stx::btree_multimap<unsigned int, std::string,
            std::less<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        btree_type bt;

        srand(34234235);
        for(unsigned int i = 0; i < 3200; i++)
        {
            ASSERT(bt.size() == i);
            bt.insert2(rand() % 100, "101");
            ASSERT(bt.size() == i + 1);
        }

        srand(34234235);
        for(unsigned int i = 0; i < 3200; i++)
        {
            ASSERT(bt.size() == 3200 - i);
            ASSERT( bt.erase_one(rand() % 100) );
            ASSERT(bt.size() == 3200 - i - 1);
        }

        ASSERT( bt.empty() );
        bt.verify();
    }

    void test_map_insert_erase_3200_descending()
    {
        typedef stx::btree_multimap<unsigned int, std::string,
            std::greater<unsigned int>, traits_nodebug<unsigned int> > btree_type;

        btree_type bt;

        srand(34234235);
        for(unsigned int i = 0; i < 3200; i++)
        {
            ASSERT(bt.size() == i);
            bt.insert2(rand() % 100, "101");
            ASSERT(bt.size() == i + 1);
        }

        srand(34234235);
        for(unsigned int i = 0; i < 3200; i++)
        {
            ASSERT(bt.size() == 3200 - i);
            ASSERT( bt.erase_one(rand() % 100) );
            ASSERT(bt.size() == 3200 - i - 1);
        }

        ASSERT( bt.empty() );
        bt.verify();
    }

    void test2_map_insert_erase_strings()
    {
        typedef stx::btree_multimap<std::string, unsigned int,
            std::less<std::string>, traits_nodebug<std::string> > btree_type;

        std::string letters = "abcdefghijklmnopqrstuvwxyz";

        btree_type bt;

        for(unsigned int a = 0; a < letters.size(); ++a)
        {
            for(unsigned int b = 0; b < letters.size(); ++b)
            {
                bt.insert2(std::string(1, letters[a]) + letters[b],
                           a * letters.size() + b);
            }
        }

        for(unsigned int b = 0; b < letters.size(); ++b)
        {
            for(unsigned int a = 0; a < letters.size(); ++a)
            {
                std::string key = std::string(1, letters[a]) + letters[b];

                ASSERT( bt.find(key)->second == a * letters.size() + b );
                ASSERT( bt.erase_one(key) );
            }
        }

        ASSERT( bt.empty() );
        bt.verify();
    }

    void test_set_100000_uint64()
    {
        stx::btree_map<uint64_t, uint8_t> bt;

        for(uint64_t i = 10; i < 100000; ++i)
        {
            uint64_t key = i % 1000;

            if (bt.find(key) == bt.end())
            {
                bt.insert( std::make_pair(key, key % 100) );
            }
        }

        ASSERT( bt.size() == 1000 );
    }

    void test_multiset_100000_uint32()
    {
        stx::btree_multiset<uint32_t> bt;

        for(uint64_t i = 0; i < 100000; ++i)
        {
            uint64_t key = i % 1000;

            bt.insert(key);
        }

        ASSERT( bt.size() == 100000 );
    }
};

// test binary search on different slot sizes
struct SimpleTest<8> __SimpleTest8;
struct SimpleTest<9> __SimpleTest9;
struct SimpleTest<10> __SimpleTest10;
struct SimpleTest<11> __SimpleTest11;
struct SimpleTest<12> __SimpleTest12;
struct SimpleTest<13> __SimpleTest13;
struct SimpleTest<14> __SimpleTest14;
struct SimpleTest<15> __SimpleTest15;
struct SimpleTest<16> __SimpleTest16;
struct SimpleTest<17> __SimpleTest17;
struct SimpleTest<19> __SimpleTest19;
struct SimpleTest<20> __SimpleTest20;
struct SimpleTest<21> __SimpleTest21;
struct SimpleTest<23> __SimpleTest23;
struct SimpleTest<24> __SimpleTest24;
struct SimpleTest<32> __SimpleTest32;
struct SimpleTest<48> __SimpleTest48;
struct SimpleTest<63> __SimpleTest63;
struct SimpleTest<64> __SimpleTest64;
struct SimpleTest<65> __SimpleTest65;
struct SimpleTest<101> __SimpleTest101;
struct SimpleTest<203> __SimpleTest203;

