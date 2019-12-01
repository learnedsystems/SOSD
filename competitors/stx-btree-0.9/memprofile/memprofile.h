/******************************************************************************
 * memprofile.h
 *
 * Class to write the datafile for a memory profile plot using malloc_count.
 *
 ******************************************************************************
 * Copyright (C) 2013 Timo Bingmann <tb@panthema.net>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifndef _MEM_PROFILE_H_
#define _MEM_PROFILE_H_

#include <stdio.h>
#include <sys/time.h>

#include "malloc_count.h"

/**
 * MemProfile is a class which hooks into malloc_count's callback and writes a
 * heap usage profile at run-time.
 *
 * A usual application will have many heap allocations and deallocations,
 * therefore these must be aggregated to create a useful plot. This is the main
 * purposes of MemProfile. However, the "resolution" of discrete aggregation
 * intervals must be configured manually, as they highly depend on the profiled
 * application.
 */
class MemProfile
{
protected:

    /// output time resolution
    double      m_time_resolution;
    /// output memory resolution
    size_t      m_size_resolution;

    /// function marker for multi-output
    const char* m_funcname;
    /// output file
    FILE*       m_file;

    /// start of current memprofile
    double      m_base_ts;
    /// start memory usage of current memprofile
    size_t      m_base_mem;
    /// start stack pointer of memprofile
    char*       m_stack_base;

    /// timestamp of previous log output
    double      m_prev_ts;
    /// memory usage of previous log output
    size_t      m_prev_mem;
    /// maximum memory usage to previous log output
    size_t      m_max;

protected:

    /// template function missing in cmath, absolute difference
    template <typename Type>
    static inline Type absdiff(const Type& a, const Type& b)
    {
        return (a < b) ? (b - a) : (a - b);
    }

    /// time is measured using gettimeofday() or omp_get_wtime()
    static inline double timestamp()
    {
#ifdef _OPENMP
        return omp_get_wtime();
#else
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + tv.tv_usec / 1e6;
#endif
    }

    /// output a data pair (ts,mem) to log file
    inline void output(double ts, unsigned long long mem)
    {
        if (m_funcname) { // more verbose output format
            fprintf(m_file, "func=%s ts=%g mem=%llu\n",
                    m_funcname, ts - m_base_ts, mem);
        }
        else { // simple gnuplot output
            fprintf(m_file, "%g %llu\n",
                    ts - m_base_ts, mem);
        }
    }

    /// callback invoked by malloc_count when heap usage changes.
    inline void callback(size_t memcurr)
    {
        size_t mem = (memcurr > m_base_mem) ? (memcurr - m_base_mem) : 0;

        if (reinterpret_cast<char*>(&mem) < m_stack_base) // add stack usage
            mem += m_stack_base - reinterpret_cast<char*>(&mem);

        double ts = timestamp();
        if (m_max < mem) m_max = mem; // keep max usage to last output

        // check to output a pair
        if (ts - m_prev_ts > m_time_resolution ||
            absdiff(mem, m_prev_mem) > m_size_resolution )
        {
            output(ts, m_max);
            m_max = 0;
            m_prev_ts = ts;
            m_prev_mem = mem;
        }
    }

    /// static callback for malloc_count, forwards to class method.
    static void static_callback(void* cookie, size_t memcurr)
    {
        return static_cast<MemProfile*>(cookie)->callback(memcurr);
    }

public:

    /** Constructor for MemProfile.
     * @param filepath          file to write memprofile log entries to.
     * @param time_resolution   resolution when a log entry is always written.
     * @param size_resolution   resolution when a log entry is always written.
     * @param funcname          enables multi-function output, appends to file.
     */
    MemProfile(const char* filepath,
               double time_resolution = 0.1, size_t size_resolution = 1024,
               const char* funcname = NULL)
        : m_time_resolution( time_resolution ),
          m_size_resolution( size_resolution ),
          m_funcname( funcname ),
          m_base_ts( timestamp() ),
          m_base_mem( malloc_count_current() ),
          m_prev_ts( m_base_ts ),
          m_prev_mem( m_base_mem ),
          m_max( 0 )
    {
        char stack;
        m_stack_base = &stack;
        m_file = fopen(filepath, funcname ? "a" : "w");
        malloc_count_set_callback(MemProfile::static_callback, this);
    }

    /// Destructor flushes currently aggregated values and closes the file.
    ~MemProfile()
    {
        m_prev_ts = 0; // force flush
        m_prev_mem = 0;
        callback( malloc_count_current() );
        malloc_count_set_callback(NULL, NULL);
        fclose(m_file);
    }
};

#endif // _MEM_PROFILE_H_
