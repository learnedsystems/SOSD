/*
 * STX B+ Tree Memory Profiling Program v0.9
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
#include <sys/wait.h>
#include <assert.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <iomanip>

#include <map>
#include <ext/hash_map>
#include <tr1/unordered_map>
#include <stx/btree_multimap.h>

#include <vector>
#include <deque>

#include "memprofile.h"

// *** Settings

/// maximum number of items to insert
const unsigned int insertnum = 1024000 * 8;

const int randseed = 34234235;

/// Time is measured using gettimeofday()
inline double timestamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 0.000001;
}

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

    /// Test the B+ tree with a auto-detected leaf/inner slots
    typedef TestClass< stx::btree_multimap<unsigned int, unsigned int,
                                           std::less<unsigned int> > > BtreeMap;
};

// --------------------------------------------------------------------------------

/// Test a generic array type with insertions
template <typename ArrayType>
struct Test_Array_Insert
{
    Test_Array_Insert(unsigned int) {}

    inline void run(unsigned int items)
    {
        ArrayType array;

        srand(randseed);
        for(unsigned int i = 0; i < items; i++) {
            unsigned int r = rand();
            array.push_back( std::make_pair(r,r) );
        }

        assert( array.size() == items );
    }
};

/// Construct different array types for a generic test class
template < template<typename ArrayType> class TestClass >
struct TestFactory_Array
{
    /// Test the vector from STL
    typedef TestClass< std::vector< std::pair<unsigned int, unsigned int> > > StdVector;

    /// Test the deque from STL
    typedef TestClass< std::deque< std::pair<unsigned int, unsigned int> > > StdDeque;
};

// --------------------------------------------------------------------------------

template <typename TestClass>
void write_memprofile(const char* filename)
{
    // the memory profile test seriously messes up malloc(), so that massive
    // house-keeping is necessary after the structure is freed. Use just fork
    // instead.
    pid_t pid = fork();
    
    if (pid == 0)
    {
        std::cout << "Writing memory profile " << filename << std::endl;
        {
            MemProfile mp(filename, 0.1, 16*1024);
            TestClass test(insertnum);	// initialize test structures

            double ts1 = timestamp();
            test.run(insertnum);	// run timed test procedure
            double ts2 = timestamp();
            std::cout << "done, time=" << (ts2-ts1) << std::endl;
        }
        exit(0);
    }

    int status;
    wait(&status);
}

/// Create memory usage profiles
int main()
{
    typedef TestFactory_Map<Test_Map_Insert> testmap_type;

    write_memprofile< testmap_type::StdMap >("memprofile-stdmap.txt");
    write_memprofile< testmap_type::HashMap >("memprofile-hashmap.txt");
    write_memprofile< testmap_type::UnorderedMap >("memprofile-unorderedmap.txt");
    write_memprofile< testmap_type::BtreeMap >("memprofile-btreemap.txt");

    typedef TestFactory_Array<Test_Array_Insert> testarray_type;

    write_memprofile< testarray_type::StdVector >("memprofile-vector.txt");
    write_memprofile< testarray_type::StdDeque >("memprofile-deque.txt");

    return 0;
}
