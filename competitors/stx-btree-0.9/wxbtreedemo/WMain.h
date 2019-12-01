/*
 * STX B+ Tree Demo Program v0.9
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

#ifndef _WMain_H_
#define _WMain_H_

#include "WMain_wxg.h"

#include "WTreeDrawing.h"

class BTreeBundle;

// Very difficult definition of the template friend drawing functions to
// include in the B+ tree classes.
#define BTREE_FRIENDS   \
    friend class ::BTreeBundle;                                         \
    template<class BTreeType> friend wxSize WTreeDrawing::BTreeOp_Draw::draw_tree(BTreeType &); \
    template<class BTreeType> friend wxSize WTreeDrawing::BTreeOp_Draw::draw_node(int, int, const class BTreeType::btree_impl::node*); \

#include <stx/btree_map.h>
#include <stx/btree_multimap.h>
#include <string>

/// The demo allows many different B+ trees to be used. All these must be
/// instantiated and correctly switched using this tree bundling class.
class BTreeBundle
{
public:

    /// Traits structure for the enclosed B+ tree instances.
    template <int Slots, typename Type>
    struct btree_traits_nodebug : stx::btree_default_map_traits<Type,Type>
    {
        static const bool       selfverify = true;
        static const bool       debug = false;

        static const int        leafslots = Slots;
        static const int        innerslots = Slots;
    };

    // *** Many many instantiations of the B+ tree classes

    stx::btree_map<int, int, std::less<int>, btree_traits_nodebug<4,int> >          btmap_int_4_slots;
    stx::btree_map<int, int, std::less<int>, btree_traits_nodebug<5,int> >          btmap_int_5_slots;
    stx::btree_map<int, int, std::less<int>, btree_traits_nodebug<6,int> >          btmap_int_6_slots;
    stx::btree_map<int, int, std::less<int>, btree_traits_nodebug<7,int> >          btmap_int_7_slots;
    stx::btree_map<int, int, std::less<int>, btree_traits_nodebug<8,int> >          btmap_int_8_slots;
    stx::btree_map<int, int, std::less<int>, btree_traits_nodebug<9,int> >          btmap_int_9_slots;
    stx::btree_map<int, int, std::less<int>, btree_traits_nodebug<10,int> >         btmap_int_10_slots;
    stx::btree_map<int, int, std::less<int>, btree_traits_nodebug<11,int> >         btmap_int_11_slots;
    stx::btree_map<int, int, std::less<int>, btree_traits_nodebug<12,int> >         btmap_int_12_slots;
    stx::btree_map<int, int, std::less<int>, btree_traits_nodebug<13,int> >         btmap_int_13_slots;
    stx::btree_map<int, int, std::less<int>, btree_traits_nodebug<14,int> >         btmap_int_14_slots;
    stx::btree_map<int, int, std::less<int>, btree_traits_nodebug<15,int> >         btmap_int_15_slots;
    stx::btree_map<int, int, std::less<int>, btree_traits_nodebug<16,int> >         btmap_int_16_slots;
    stx::btree_map<int, int, std::less<int>, btree_traits_nodebug<32,int> >         btmap_int_32_slots;

    stx::btree_map<wxString, wxString, std::less<wxString>, btree_traits_nodebug<4,wxString> >           btmap_string_4_slots;
    stx::btree_map<wxString, wxString, std::less<wxString>, btree_traits_nodebug<5,wxString> >           btmap_string_5_slots;
    stx::btree_map<wxString, wxString, std::less<wxString>, btree_traits_nodebug<6,wxString> >           btmap_string_6_slots;
    stx::btree_map<wxString, wxString, std::less<wxString>, btree_traits_nodebug<7,wxString> >           btmap_string_7_slots;
    stx::btree_map<wxString, wxString, std::less<wxString>, btree_traits_nodebug<8,wxString> >           btmap_string_8_slots;
    stx::btree_map<wxString, wxString, std::less<wxString>, btree_traits_nodebug<9,wxString> >           btmap_string_9_slots;
    stx::btree_map<wxString, wxString, std::less<wxString>, btree_traits_nodebug<10,wxString> >          btmap_string_10_slots;
    stx::btree_map<wxString, wxString, std::less<wxString>, btree_traits_nodebug<11,wxString> >          btmap_string_11_slots;
    stx::btree_map<wxString, wxString, std::less<wxString>, btree_traits_nodebug<12,wxString> >          btmap_string_12_slots;
    stx::btree_map<wxString, wxString, std::less<wxString>, btree_traits_nodebug<13,wxString> >          btmap_string_13_slots;
    stx::btree_map<wxString, wxString, std::less<wxString>, btree_traits_nodebug<14,wxString> >          btmap_string_14_slots;
    stx::btree_map<wxString, wxString, std::less<wxString>, btree_traits_nodebug<15,wxString> >          btmap_string_15_slots;
    stx::btree_map<wxString, wxString, std::less<wxString>, btree_traits_nodebug<16,wxString> >          btmap_string_16_slots;
    stx::btree_map<wxString, wxString, std::less<wxString>, btree_traits_nodebug<32,wxString> >          btmap_string_32_slots;

    stx::btree_multimap<int, int, std::less<int>, btree_traits_nodebug<4,int> >             btmultimap_int_4_slots;
    stx::btree_multimap<int, int, std::less<int>, btree_traits_nodebug<5,int> >             btmultimap_int_5_slots;
    stx::btree_multimap<int, int, std::less<int>, btree_traits_nodebug<6,int> >             btmultimap_int_6_slots;
    stx::btree_multimap<int, int, std::less<int>, btree_traits_nodebug<7,int> >             btmultimap_int_7_slots;
    stx::btree_multimap<int, int, std::less<int>, btree_traits_nodebug<8,int> >             btmultimap_int_8_slots;
    stx::btree_multimap<int, int, std::less<int>, btree_traits_nodebug<9,int> >             btmultimap_int_9_slots;
    stx::btree_multimap<int, int, std::less<int>, btree_traits_nodebug<10,int> >            btmultimap_int_10_slots;
    stx::btree_multimap<int, int, std::less<int>, btree_traits_nodebug<11,int> >            btmultimap_int_11_slots;
    stx::btree_multimap<int, int, std::less<int>, btree_traits_nodebug<12,int> >            btmultimap_int_12_slots;
    stx::btree_multimap<int, int, std::less<int>, btree_traits_nodebug<13,int> >            btmultimap_int_13_slots;
    stx::btree_multimap<int, int, std::less<int>, btree_traits_nodebug<14,int> >            btmultimap_int_14_slots;
    stx::btree_multimap<int, int, std::less<int>, btree_traits_nodebug<15,int> >            btmultimap_int_15_slots;
    stx::btree_multimap<int, int, std::less<int>, btree_traits_nodebug<16,int> >            btmultimap_int_16_slots;
    stx::btree_multimap<int, int, std::less<int>, btree_traits_nodebug<32,int> >            btmultimap_int_32_slots;

    stx::btree_multimap<wxString, wxString, std::less<wxString>, btree_traits_nodebug<4,wxString> >              btmultimap_string_4_slots;
    stx::btree_multimap<wxString, wxString, std::less<wxString>, btree_traits_nodebug<5,wxString> >              btmultimap_string_5_slots;
    stx::btree_multimap<wxString, wxString, std::less<wxString>, btree_traits_nodebug<6,wxString> >              btmultimap_string_6_slots;
    stx::btree_multimap<wxString, wxString, std::less<wxString>, btree_traits_nodebug<7,wxString> >              btmultimap_string_7_slots;
    stx::btree_multimap<wxString, wxString, std::less<wxString>, btree_traits_nodebug<8,wxString> >              btmultimap_string_8_slots;
    stx::btree_multimap<wxString, wxString, std::less<wxString>, btree_traits_nodebug<9,wxString> >              btmultimap_string_9_slots;
    stx::btree_multimap<wxString, wxString, std::less<wxString>, btree_traits_nodebug<10,wxString> >             btmultimap_string_10_slots;
    stx::btree_multimap<wxString, wxString, std::less<wxString>, btree_traits_nodebug<11,wxString> >             btmultimap_string_11_slots;
    stx::btree_multimap<wxString, wxString, std::less<wxString>, btree_traits_nodebug<12,wxString> >             btmultimap_string_12_slots;
    stx::btree_multimap<wxString, wxString, std::less<wxString>, btree_traits_nodebug<13,wxString> >             btmultimap_string_13_slots;
    stx::btree_multimap<wxString, wxString, std::less<wxString>, btree_traits_nodebug<14,wxString> >             btmultimap_string_14_slots;
    stx::btree_multimap<wxString, wxString, std::less<wxString>, btree_traits_nodebug<15,wxString> >             btmultimap_string_15_slots;
    stx::btree_multimap<wxString, wxString, std::less<wxString>, btree_traits_nodebug<16,wxString> >             btmultimap_string_16_slots;
    stx::btree_multimap<wxString, wxString, std::less<wxString>, btree_traits_nodebug<32,wxString> >             btmultimap_string_32_slots;

    /// Selects the active tree: type == 0 -> integer, type == 1 -> string.
    int         selected_type;

    /// Selects the active tree: slots is the number of slots.
    int         selected_slots;

    /// Selects the active tree: map or mulitmap.
    bool        selected_multimap;

    /// Test if this is a integer tree
    inline bool isIntegerType() const
    {
        return (selected_type == 0);
    }

    /// Test if this is a string tree
    inline bool isStringType() const
    {
        return (selected_type == 1);
    }

    /// Test if the selected tree allows duplicates
    inline bool isMultimap() const
    {
        return selected_multimap;
    }

    template <class Operation>
    typename Operation::result_type     run(Operation operation)
    {
        if (isIntegerType() && !isMultimap())
        {
            if (selected_slots == 4) {
                return operation.opInteger(btmap_int_4_slots);
            }
            else if (selected_slots == 5) {
                return operation.opInteger(btmap_int_5_slots);
            }
            else if (selected_slots == 6) {
                return operation.opInteger(btmap_int_6_slots);
            }
            else if (selected_slots == 7) {
                return operation.opInteger(btmap_int_7_slots);
            }
            else if (selected_slots == 8) {
                return operation.opInteger(btmap_int_8_slots);
            }
            else if (selected_slots == 9) {
                return operation.opInteger(btmap_int_9_slots);
            }
            else if (selected_slots == 10) {
                return operation.opInteger(btmap_int_10_slots);
            }
            else if (selected_slots == 11) {
                return operation.opInteger(btmap_int_11_slots);
            }
            else if (selected_slots == 12) {
                return operation.opInteger(btmap_int_12_slots);
            }
            else if (selected_slots == 13) {
                return operation.opInteger(btmap_int_13_slots);
            }
            else if (selected_slots == 14) {
                return operation.opInteger(btmap_int_14_slots);
            }
            else if (selected_slots == 15) {
                return operation.opInteger(btmap_int_15_slots);
            }
            else if (selected_slots == 16) {
                return operation.opInteger(btmap_int_16_slots);
            }
            else if (selected_slots == 32) {
                return operation.opInteger(btmap_int_32_slots);
            }
        }
        else if (isStringType() && !isMultimap())
        {
            if (selected_slots == 4) {
                return operation.opString(btmap_string_4_slots);
            }
            else if (selected_slots == 5) {
                return operation.opString(btmap_string_5_slots);
            }
            else if (selected_slots == 6) {
                return operation.opString(btmap_string_6_slots);
            }
            else if (selected_slots == 7) {
                return operation.opString(btmap_string_7_slots);
            }
            else if (selected_slots == 8) {
                return operation.opString(btmap_string_8_slots);
            }
            else if (selected_slots == 9) {
                return operation.opString(btmap_string_9_slots);
            }
            else if (selected_slots == 10) {
                return operation.opString(btmap_string_10_slots);
            }
            else if (selected_slots == 11) {
                return operation.opString(btmap_string_11_slots);
            }
            else if (selected_slots == 12) {
                return operation.opString(btmap_string_12_slots);
            }
            else if (selected_slots == 13) {
                return operation.opString(btmap_string_13_slots);
            }
            else if (selected_slots == 14) {
                return operation.opString(btmap_string_14_slots);
            }
            else if (selected_slots == 15) {
                return operation.opString(btmap_string_15_slots);
            }
            else if (selected_slots == 16) {
                return operation.opString(btmap_string_16_slots);
            }
            else if (selected_slots == 32) {
                return operation.opString(btmap_string_32_slots);
            }
        }
        else if (isIntegerType() && isMultimap())
        {
            if (selected_slots == 4) {
                return operation.opIntegerMulti(btmultimap_int_4_slots);
            }
            else if (selected_slots == 5) {
                return operation.opIntegerMulti(btmultimap_int_5_slots);
            }
            else if (selected_slots == 6) {
                return operation.opIntegerMulti(btmultimap_int_6_slots);
            }
            else if (selected_slots == 7) {
                return operation.opIntegerMulti(btmultimap_int_7_slots);
            }
            else if (selected_slots == 8) {
                return operation.opIntegerMulti(btmultimap_int_8_slots);
            }
            else if (selected_slots == 9) {
                return operation.opIntegerMulti(btmultimap_int_9_slots);
            }
            else if (selected_slots == 10) {
                return operation.opIntegerMulti(btmultimap_int_10_slots);
            }
            else if (selected_slots == 11) {
                return operation.opIntegerMulti(btmultimap_int_11_slots);
            }
            else if (selected_slots == 12) {
                return operation.opIntegerMulti(btmultimap_int_12_slots);
            }
            else if (selected_slots == 13) {
                return operation.opIntegerMulti(btmultimap_int_13_slots);
            }
            else if (selected_slots == 14) {
                return operation.opIntegerMulti(btmultimap_int_14_slots);
            }
            else if (selected_slots == 15) {
                return operation.opIntegerMulti(btmultimap_int_15_slots);
            }
            else if (selected_slots == 16) {
                return operation.opIntegerMulti(btmultimap_int_16_slots);
            }
            else if (selected_slots == 32) {
                return operation.opIntegerMulti(btmultimap_int_32_slots);
            }
        }
        else if (isStringType() && isMultimap())
        {
            if (selected_slots == 4) {
                return operation.opStringMulti(btmultimap_string_4_slots);
            }
            else if (selected_slots == 5) {
                return operation.opStringMulti(btmultimap_string_5_slots);
            }
            else if (selected_slots == 6) {
                return operation.opStringMulti(btmultimap_string_6_slots);
            }
            else if (selected_slots == 7) {
                return operation.opStringMulti(btmultimap_string_7_slots);
            }
            else if (selected_slots == 8) {
                return operation.opStringMulti(btmultimap_string_8_slots);
            }
            else if (selected_slots == 9) {
                return operation.opStringMulti(btmultimap_string_9_slots);
            }
            else if (selected_slots == 10) {
                return operation.opStringMulti(btmultimap_string_10_slots);
            }
            else if (selected_slots == 11) {
                return operation.opStringMulti(btmultimap_string_11_slots);
            }
            else if (selected_slots == 12) {
                return operation.opStringMulti(btmultimap_string_12_slots);
            }
            else if (selected_slots == 13) {
                return operation.opStringMulti(btmultimap_string_13_slots);
            }
            else if (selected_slots == 14) {
                return operation.opStringMulti(btmultimap_string_14_slots);
            }
            else if (selected_slots == 15) {
                return operation.opStringMulti(btmultimap_string_15_slots);
            }
            else if (selected_slots == 16) {
                return operation.opStringMulti(btmultimap_string_16_slots);
            }
            else if (selected_slots == 32) {
                return operation.opStringMulti(btmultimap_string_32_slots);
            }
        }

        throw(wxT("Program Error: could not find selected B+ tree"));
    }

    // *** Marked Node Slots

    /// node pointer of the first mark
    const void* mark1_node;
    /// slot number of the first mark
    int         mark1_slot;

    /// node pointer of the second mark
    const void* mark2_node;
    /// slot number of the second mark
    int         mark2_slot;

    /// Clear both marks
    inline void clearMarks()
    {
        mark1_node = 0;
        mark1_slot = 0;
        mark2_node = 0;
        mark2_slot = 0;
    }

    /// Set the first mark, clear the second
    template <class BTreeIter>
    inline void setMark1(const BTreeIter &iter)
    {
        mark1_node = iter.currnode;
        mark1_slot = iter.currslot;
        mark2_node = 0;
        mark2_slot = 0;
    }

    /// Set the second mark
    template <class BTreeIter>
    inline void setMark2(const BTreeIter &iter)
    {
        mark2_node = iter.currnode;
        mark2_slot = iter.currslot;
    }

    /// Compare to the first mark
    inline bool isMark1(const void* node, int slot) const
    {
        return (mark1_node == node) && (mark1_slot == slot);
    }

    /// Compare to the second mark
    inline bool isMark2(const void* node, int slot) const
    {
        return (mark2_node == node) && (mark2_slot == slot);
    }
};

/** Main Window class */
class WMain : public WMain_wxg
{
public:
    WMain();

    class BTreeBundle   treebundle;

    /// Refresh view(s) of the B+ tree after it changes
    void        UpdateViews();

    // *** Choices to selected the activated B+ tree instance

    void        OnChoiceDataType(wxCommandEvent &ce);
    void        OnChoiceNodeSlots(wxCommandEvent &ce);
    void        OnCheckboxDuplicates(wxCommandEvent &ce);

    // *** Operation buttons to change the tree's contents

    void        OnButtonInsert(wxCommandEvent &ce);
    void        OnButtonErase(wxCommandEvent &ce);
    void        OnButtonInsertRandom(wxCommandEvent &ce);
    void        OnButtonFindKey(wxCommandEvent &ce);
    void        OnButtonEqualRange(wxCommandEvent &ce);
    void        OnButtonClear(wxCommandEvent &ce);
    void        OnButtonLoadFile(wxCommandEvent &ce);

    void        OnMenuInsertRandom(wxCommandEvent &ce);

    DECLARE_EVENT_TABLE();
};

#endif // _WMain_H_
