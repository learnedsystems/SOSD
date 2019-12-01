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

#include <map>
#include <vector>

#include "WMain.h"
#include "WTreeDrawing.h"

#include <wx/filedlg.h>
#include <wx/textfile.h>
#include <wx/tokenzr.h>

WMain::WMain()
    : WMain_wxg(NULL, -1, wxT("STX B+ Tree Demo " VERSION " - http://panthema.net/"))
{
    {
        #include "progicon.xpm"
        SetIcon(wxIcon(progicon));
    }

    window_TreeDrawing->SetWMain(this);
    SetTitle(wxT("STX B+ Tree Demo " VERSION " - http://panthema.net/"));

    treebundle.selected_type = 0;
    treebundle.selected_slots = 4;
    treebundle.clearMarks();
}

void WMain::UpdateViews()
{
    window_TreeDrawing->Refresh();
}

// *** Insert Operation

struct BTreeOp_Insert
{
    BTreeBundle&        treebundle;
    wxString            inputkey, inputdata;

    typedef     wxString        result_type;

    template <class BTreeType>
    inline result_type opInteger(BTreeType &bt) const
    {
        long key, data;

        if (!inputkey.ToLong(&key)) {
            return wxT("Could not interpret key string as integer.");
        }
        if (!inputdata.ToLong(&data)) {
            return wxT("Could not interpret data string as integer.");
        }

        std::pair<typename BTreeType::iterator, bool> inres = bt.insert2(key, data);
        if (!inres.second)
            return wxT("Insert returned false: key already exists.");
        else
        {
            treebundle.setMark1(inres.first);
            return wxT("Insert succeeded.");
        }
    }

    template <class BTreeType>
    inline result_type opIntegerMulti(BTreeType &bt) const
    {
        long key, data;

        if (!inputkey.ToLong(&key)) {
            return wxT("Could not interpret key string as integer.");
        }
        if (!inputdata.ToLong(&data)) {
            return wxT("Could not interpret data string as integer.");
        }

        typename BTreeType::iterator iter = bt.insert2(key, data);

        treebundle.setMark1(iter);
        return wxT("Insert succeeded.");
    }

    template <class BTreeType>
    inline result_type opString(BTreeType &bt) const
    {
        std::pair<typename BTreeType::iterator, bool> inres = bt.insert2(inputkey, inputdata);

        if (!inres.second)
            return wxT("Insert returned false: key already exists.");
        else
        {
            treebundle.setMark1(inres.first);
            return wxT("Insert succeeded.");
        }
    }

    template <class BTreeType>
    inline result_type opStringMulti(BTreeType &bt) const
    {
        typename BTreeType::iterator iter = bt.insert2(inputkey, inputdata);

        treebundle.setMark1(iter);
        return wxT("Insert succeeded.");
    }
};

void WMain::OnButtonInsert(wxCommandEvent &)
{
    BTreeOp_Insert op = { treebundle };
    op.inputkey = textctrl_Key->GetValue();
    op.inputdata = textctrl_Data->GetValue();

    wxString result = treebundle.run(op);
    textctrl_OpResult->SetValue(result);

    UpdateViews();
}

// *** Erase Operation

struct BTreeOp_Erase
{
    BTreeBundle&        treebundle;
    wxString            inputkey;

    typedef     wxString        result_type;

    template <class BTreeType>
    inline result_type opInteger(BTreeType &bt) const
    {
        long key;

        if (!inputkey.ToLong(&key)) {
            return wxT("Could not interpret key string as integer.");
        }

        if (!bt.erase(key))
            return wxT("Erase returned false: key does not exist.");
        else
        {
            treebundle.clearMarks();
            return wxT("Erase succeeded.");
        }
    }

    template <class BTreeType>
    inline result_type opIntegerMulti(BTreeType &bt) const
    {
        return opInteger(bt);
    }

    template <class BTreeType>
    inline result_type opString(BTreeType &bt) const
    {
        if ( !bt.erase(inputkey) )
            return wxT("Erase returned false: key does not exist.");
        else
        {
            treebundle.clearMarks();
            return wxT("Erase succeeded.");
        }
    }

    template <class BTreeType>
    inline result_type opStringMulti(BTreeType &bt) const
    {
        return opString(bt);
    }
};

void WMain::OnButtonErase(wxCommandEvent &)
{
    BTreeOp_Erase op = { treebundle };
    op.inputkey = textctrl_Key->GetValue();

    wxString result = treebundle.run(op);
    textctrl_OpResult->SetValue(result);

    UpdateViews();
}

// *** Insert Random Operation

void WMain::OnButtonInsertRandom(wxCommandEvent &)
{
    wxMenu* menu = new wxMenu;

    menu->Append(500,   wxT("Insert 10 Random Integer Pairs"));
    menu->Append(501,   wxT("Insert 20 Random Integer Pairs"));
    menu->Append(502,   wxT("Insert 50 Random Integer Pairs"));
    menu->Append(503,   wxT("Insert 100 Random Integer Pairs"));
    menu->Append(504,   wxT("Insert 200 Random Integer Pairs"));

    if (treebundle.isStringType())
    {
        menu->AppendSeparator();
        menu->Append(510,       wxT("Insert 10 Random 1 Letter String Pairs"));
        menu->Append(511,       wxT("Insert 10 Random 2 Letter String Pairs"));
        menu->Append(512,       wxT("Insert 25 Random 2 Letter String Pairs"));
        menu->Append(513,       wxT("Insert 50 Random 2 Letter String Pairs"));
        menu->Append(514,       wxT("Insert 10 Random 3 Letter String Pairs"));
        menu->Append(515,       wxT("Insert 25 Random 3 Letter String Pairs"));
        menu->Append(516,       wxT("Insert 50 Random 3 Letter String Pairs"));
        menu->Append(517,       wxT("Insert 100 Random 3 Letter String Pairs"));
        menu->Append(518,       wxT("Insert 200 Random 3 Letter String Pairs"));
    }

    PopupMenu(menu);
}

struct BTreeOp_InsertRandomInteger
{
    int         num;

    typedef     wxString        result_type;

    template <class BTreeType>
    inline result_type opInteger(BTreeType &bt) const
    {
        int count = 0;
        for(unsigned int i = 0; i < num; i++)
        {
            if (bt.insert2(rand() % 1000, rand() % 1000).second)
                count++;
        }

        return wxString::Format(wxT("Inserted %d random integers."), count);
    }

    template <class BTreeType>
    inline result_type opIntegerMulti(BTreeType &bt) const
    {
        for(unsigned int i = 0; i < num; i++)
        {
            bt.insert2(rand() % 1000, rand() % 1000);
        }

        return wxString::Format(wxT("Inserted %d random integers."), num);
    }

    template <class BTreeType>
    inline result_type opString(BTreeType &bt) const
    {
        int count = 0;
        for(unsigned int i = 0; i < num; i++)
        {
            wxString key, val;
            key << rand() % 1000;
            val << rand() % 1000;

            if (bt.insert2(key, val).second)
                count++;
        }

        return wxString::Format(wxT("Inserted %d random integer strings."), count);
    }

    template <class BTreeType>
    inline result_type opStringMulti(BTreeType &bt) const
    {
        for(unsigned int i = 0; i < num; i++)
        {
            wxString key, val;
            key << rand() % 1000;
            val << rand() % 1000;

            bt.insert2(key, val);
        }

        return wxString::Format(wxT("Inserted %d random integer strings."), num);
    }
};

struct BTreeOp_InsertRandomString
{
    int         len;
    int         num;

    typedef     wxString        result_type;

    template <class BTreeType>
    inline result_type opInteger(BTreeType &bt) const
    {
        return wxT("Cannot insert strings into integer tree");
    }

    template <class BTreeType>
    inline result_type opIntegerMulti(BTreeType &bt) const
    {
        return opInteger(bt);
    }

    template <class BTreeType>
    inline result_type opString(BTreeType &bt) const
    {
        static const char letters[27] = "abcdefghijklmnopqrstuvwxyz";

        int count = 0;
        for(unsigned int i = 0; i < num; ++i)
        {
            wxString key, val;
            for(unsigned int l = 0; l < len; ++l)
            {
                key += letters[rand() % 26];
                val += letters[rand() % 26];
            }

            if (bt.insert2(key, val).second)
                count++;
        }

        return wxString::Format(wxT("Inserted %d random strings."), count);
    }

    template <class BTreeType>
    inline result_type opStringMulti(BTreeType &bt) const
    {
        static const char letters[27] = "abcdefghijklmnopqrstuvwxyz";

        for(unsigned int i = 0; i < num; ++i)
        {
            wxString key, val;
            for(unsigned int l = 0; l < len; ++l)
            {
                key += letters[rand() % 26];
                val += letters[rand() % 26];
            }

            bt.insert2(key, val);
        }

        return wxString::Format(wxT("Inserted %d random strings."), num);
    }
};

void WMain::OnMenuInsertRandom(wxCommandEvent &ce)
{
    srand(time(NULL));
    if (ce.GetId() == 500)
    {
        BTreeOp_InsertRandomInteger op = { 10 };
        textctrl_OpResult->SetValue( treebundle.run(op) );
    }
    else if (ce.GetId() == 501)
    {
        BTreeOp_InsertRandomInteger op = { 20 };
        textctrl_OpResult->SetValue( treebundle.run(op) );
    }
    else if (ce.GetId() == 502)
    {
        BTreeOp_InsertRandomInteger op = { 50 };
        textctrl_OpResult->SetValue( treebundle.run(op) );
    }
    else if (ce.GetId() == 503)
    {
        BTreeOp_InsertRandomInteger op = { 100 };
        textctrl_OpResult->SetValue( treebundle.run(op) );
    }
    else if (ce.GetId() == 504)
    {
        BTreeOp_InsertRandomInteger op = { 200 };
        textctrl_OpResult->SetValue( treebundle.run(op) );
    }

    else if (ce.GetId() == 510)
    {
        BTreeOp_InsertRandomString op = { 1, 10 };
        textctrl_OpResult->SetValue( treebundle.run(op) );
    }
    else if (ce.GetId() == 511)
    {
        BTreeOp_InsertRandomString op = { 2, 10 };
        textctrl_OpResult->SetValue( treebundle.run(op) );
    }
    else if (ce.GetId() == 512)
    {
        BTreeOp_InsertRandomString op = { 2, 25 };
        textctrl_OpResult->SetValue( treebundle.run(op) );
    }
    else if (ce.GetId() == 513)
    {
        BTreeOp_InsertRandomString op = { 2, 50 };
        textctrl_OpResult->SetValue( treebundle.run(op) );
    }
    else if (ce.GetId() == 514)
    {
        BTreeOp_InsertRandomString op = { 3, 10 };
        textctrl_OpResult->SetValue( treebundle.run(op) );
    }
    else if (ce.GetId() == 515)
    {
        BTreeOp_InsertRandomString op = { 3, 25 };
        textctrl_OpResult->SetValue( treebundle.run(op) );
    }
    else if (ce.GetId() == 516)
    {
        BTreeOp_InsertRandomString op = { 3, 50 };
        textctrl_OpResult->SetValue( treebundle.run(op) );
    }
    else if (ce.GetId() == 517)
    {
        BTreeOp_InsertRandomString op = { 3, 100 };
        textctrl_OpResult->SetValue( treebundle.run(op) );
    }
    else if (ce.GetId() == 518)
    {
        BTreeOp_InsertRandomString op = { 3, 200 };
        textctrl_OpResult->SetValue( treebundle.run(op) );
    }

    UpdateViews();
}

// *** Find Key Operation

struct BTreeOp_FindKey
{
    BTreeBundle&        treebundle;
    wxString            inputkey;

    typedef     wxString        result_type;

    template <class BTreeType>
    inline result_type opInteger(BTreeType &bt) const
    {
        long key;
        if (!inputkey.ToLong(&key)) {
            return wxT("Could not interpret key string as integer.");
        }

        typename BTreeType::const_iterator bti = bt.find(key);

        if (bti == bt.end())
        {
            treebundle.clearMarks();
            return wxT("Find Key failed: key does not exist.");
        }
        else
        {
            treebundle.setMark1(bti);
            return wxT("Find Key succeeded.");
        }
    }

    template <class BTreeType>
    inline result_type opIntegerMulti(BTreeType &bt) const
    {
        return opInteger(bt);
    }

    template <class BTreeType>
    inline result_type opString(BTreeType &bt) const
    {
        typename BTreeType::const_iterator bti = bt.find(inputkey);

        if (bti == bt.end())
        {
            treebundle.clearMarks();
            return wxT("Find Key failed: key does not exist.");
        }
        else
        {
            treebundle.setMark1(bti);
            return wxT("Find Key succeeded.");
        }
    }

    template <class BTreeType>
    inline result_type opStringMulti(BTreeType &bt) const
    {
        return opString(bt);
    }
};

void WMain::OnButtonFindKey(wxCommandEvent &)
{
    BTreeOp_FindKey op = { treebundle, textctrl_Key->GetValue() };

    wxString result = treebundle.run(op);
    textctrl_OpResult->SetValue(result);

    UpdateViews();
}

// *** Equal Range Operation

struct BTreeOp_EqualRange
{
    BTreeBundle&        treebundle;
    wxString            inputkey;

    typedef     wxString        result_type;

    template <class BTreeType>
    inline result_type opInteger(BTreeType &bt) const
    {
        long key;
        if (!inputkey.ToLong(&key)) {
            return wxT("Could not interpret key string as integer.");
        }

        std::pair< typename BTreeType::const_iterator,  typename BTreeType::const_iterator >
            btip = bt.equal_range(key);

        if (btip.first == bt.end() && btip.second == bt.end())
        {
            treebundle.clearMarks();
            return wxT("Equal Range failed: key is beyond the maximum of the tree.");
        }
        else
        {
            treebundle.setMark1(btip.first);
            treebundle.setMark2(btip.second);
            return wxT("Equal Range succeeded.");
        }
    }

    template <class BTreeType>
    inline result_type opIntegerMulti(BTreeType &bt) const
    {
        return opInteger(bt);
    }

    template <class BTreeType>
    inline result_type opString(BTreeType &bt) const
    {
        std::pair< typename BTreeType::const_iterator,  typename BTreeType::const_iterator >
            btip = bt.equal_range(inputkey);

        if (btip.first == bt.end() && btip.second == bt.end())
        {
            treebundle.clearMarks();
            return wxT("Equal Range failed: key is beyond the maximum of the tree.");
        }
        else
        {
            treebundle.setMark1(btip.first);
            treebundle.setMark2(btip.second);
            return wxT("Equal Range succeeded.");
        }
    }

    template <class BTreeType>
    inline result_type opStringMulti(BTreeType &bt) const
    {
        return opString(bt);
    }
};

void WMain::OnButtonEqualRange(wxCommandEvent &)
{
    BTreeOp_EqualRange op = { treebundle, textctrl_Key->GetValue() };

    wxString result = treebundle.run(op);
    textctrl_OpResult->SetValue(result);

    UpdateViews();
}

// *** Clear Tree Operation

struct BTreeOp_Clear
{
    typedef     wxString        result_type;

    template <class BTreeType>
    inline result_type opInteger(BTreeType &bt) const
    {
        bt.clear();
        return wxT("Tree cleared.");
    }

    template <class BTreeType>
    inline result_type opIntegerMulti(BTreeType &bt) const
    {
        return opInteger(bt);
    }

    template <class BTreeType>
    inline result_type opString(BTreeType &bt) const
    {
        bt.clear();
        return wxT("Tree cleared.");
    }

    template <class BTreeType>
    inline result_type opStringMulti(BTreeType &bt) const
    {
        return opString(bt);
    }
};

void WMain::OnButtonClear(wxCommandEvent &)
{
    BTreeOp_Clear op;
    wxString result = treebundle.run(op);
    textctrl_OpResult->SetValue(result);

    UpdateViews();
}

// *** Load File Operation

struct BTreeOp_LoadFile
{
    std::vector< std::pair<wxString,wxString> > &invector;

    typedef     int     result_type;

    template <class BTreeType>
    inline result_type opInteger(BTreeType &bt) const
    {
        unsigned int btsizebefore = bt.size();

        for (std::vector< std::pair<wxString,wxString> >::const_iterator ci = invector.begin();
             ci != invector.end(); ++ci)
        {
            long key, val;
            if (ci->first.ToLong(&key) && ci->second.ToLong(&val))
            {
                bt.insert2(key, val);
            }
        }

        return bt.size() - btsizebefore;
    }

    template <class BTreeType>
    inline result_type opIntegerMulti(BTreeType &bt) const
    {
        return opInteger(bt);
    }

    template <class BTreeType>
    inline result_type opString(BTreeType &bt) const
    {
        unsigned int btsizebefore = bt.size();

        for (std::vector< std::pair<wxString,wxString> >::const_iterator ci = invector.begin();
             ci != invector.end(); ++ci)
        {
            bt.insert(*ci);
        }

        return bt.size() - btsizebefore;
    }

    template <class BTreeType>
    inline result_type opStringMulti(BTreeType &bt) const
    {
        return opString(bt);
    }
};

void WMain::OnButtonLoadFile(wxCommandEvent &)
{
    wxFileDialog filedlg(this,
                         wxT("Select text file containing key/data pairs."),
                         wxT(""), wxT(""),
                         wxT("Text files (*.txt)|*.txt|CSV files (*.csv)|*.csv|All files (*.*)|*.*"),
#if wxCHECK_VERSION(2,8,0)
                         wxFD_OPEN | wxFD_FILE_MUST_EXIST);
#else
                         wxOPEN | wxFILE_MUST_EXIST);
#endif

    if (filedlg.ShowModal() != wxID_OK) return;

    wxTextFile text(filedlg.GetPath());
    if (!text.Open()) return;

    std::vector< std::pair<wxString,wxString> > datavector;

    for (wxString line = text.GetFirstLine(); !text.Eof(); line = text.GetNextLine())
    {
        wxStringTokenizer linetok(line, wxT(" \t:;,"));

        wxString word1 = linetok.GetNextToken();
        wxString word2 = linetok.GetNextToken();

        if (word1.IsEmpty()) continue;

        datavector.push_back( std::pair<wxString,wxString>(word1, word2) );
    }

    BTreeOp_LoadFile op = { datavector };
    int insize = treebundle.run(op);

    textctrl_OpResult->SetValue(wxString::Format(wxT("Loaded %lu data items into new tree"), insize));

    UpdateViews();
}

// *** Tree Activation Operation

struct BTreeOp_GetVector
{
    typedef     std::pair<wxString,wxString> stringpair_type;

    std::vector< std::pair<wxString,wxString> > &outvector;

    typedef     void    result_type;

    template <class BTreeType>
    inline result_type opInteger(BTreeType &bt) const
    {
        for(typename BTreeType::const_iterator ci = bt.begin(); ci != bt.end(); ++ci)
        {
            wxString key, val;
            key << ci->first;
            val << ci->second;

            outvector.push_back( stringpair_type(key, val) );
        }
        bt.clear();
    }

    template <class BTreeType>
    inline result_type opIntegerMulti(BTreeType &bt) const
    {
        return opInteger(bt);
    }

    template <class BTreeType>
    inline result_type opString(BTreeType &bt) const
    {
        for(typename BTreeType::const_iterator ci = bt.begin(); ci != bt.end(); ++ci)
        {
            outvector.push_back(*ci);
        }
        bt.clear();
    }

    template <class BTreeType>
    inline result_type opStringMulti(BTreeType &bt) const
    {
        return opString(bt);
    }
};

struct BTreeOp_PutVector
{
    std::vector< std::pair<wxString,wxString> > &invector;

    typedef     int     result_type;

    template <class BTreeType>
    inline result_type opInteger(BTreeType &bt) const
    {
        bt.clear();
        for (std::vector< std::pair<wxString,wxString> >::const_iterator ci = invector.begin();
             ci != invector.end(); ++ci)
        {
            long key, val;
            if (ci->first.ToLong(&key) && ci->second.ToLong(&val))
            {
                bt.insert2(key, val);
            }
        }

        return bt.size();
    }

    template <class BTreeType>
    inline result_type opIntegerMulti(BTreeType &bt) const
    {
        return opInteger(bt);
    }

    template <class BTreeType>
    inline result_type opString(BTreeType &bt) const
    {
        bt.clear();
        for (std::vector< std::pair<wxString,wxString> >::const_iterator ci = invector.begin();
             ci != invector.end(); ++ci)
        {
            bt.insert(*ci);
        }

        return bt.size();
    }

    template <class BTreeType>
    inline result_type opStringMulti(BTreeType &bt) const
    {
        return opString(bt);
    }
};

void WMain::OnChoiceDataType(wxCommandEvent &)
{
    int seltype = choice_DataType->GetSelection();
    if (seltype >= 2) return;

    if (treebundle.selected_type == seltype) return;

    std::vector< std::pair<wxString,wxString> > datavector;

    BTreeOp_GetVector op1 = { datavector };
    treebundle.run(op1);

    treebundle.selected_type = seltype;

    BTreeOp_PutVector op2 = { datavector };
    int btsize = treebundle.run(op2);

    textctrl_OpResult->SetValue(wxString::Format(wxT("Moved %lu data items into new tree"), btsize));

    UpdateViews();
}

void WMain::OnChoiceNodeSlots(wxCommandEvent &)
{
    static const int choice_NodeSlots_choices[] = {
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32
    };

    int selslot = choice_NodeSlots->GetSelection();
    if (selslot >= sizeof(choice_NodeSlots_choices) / sizeof(choice_NodeSlots_choices[0])) return;

    selslot = choice_NodeSlots_choices[selslot];

    if (treebundle.selected_slots == selslot) return;

    std::vector< std::pair<wxString,wxString> > datavector;

    BTreeOp_GetVector op1 = { datavector };
    treebundle.run(op1);

    treebundle.selected_slots = selslot;

    BTreeOp_PutVector op2 = { datavector };
    int btsize = treebundle.run(op2);

    textctrl_OpResult->SetValue(wxString::Format(wxT("Moved %lu data items into new tree"), btsize));

    UpdateViews();
}

void WMain::OnCheckboxDuplicates(wxCommandEvent &)
{
    bool seldup = checkbox_Duplicates->GetValue();

    if (treebundle.selected_multimap == seldup) return;

    std::vector< std::pair<wxString,wxString> > datavector;

    BTreeOp_GetVector op1 = { datavector };
    treebundle.run(op1);

    treebundle.selected_multimap = seldup;

    BTreeOp_PutVector op2 = { datavector };
    int btsize = treebundle.run(op2);

    textctrl_OpResult->SetValue(wxString::Format(wxT("Moved %lu data items into new tree"), btsize));

    UpdateViews();
}

BEGIN_EVENT_TABLE(WMain, wxFrame)

    EVT_CHOICE  (ID_CHOICE_DATATYPE,            WMain::OnChoiceDataType)
    EVT_CHOICE  (ID_CHOICE_NODESLOTS,           WMain::OnChoiceNodeSlots)
    EVT_CHECKBOX(ID_CHECKBOX_DUPLICATES,        WMain::OnCheckboxDuplicates)

    EVT_MENU_RANGE (500, 520,                   WMain::OnMenuInsertRandom)

    EVT_BUTTON  (ID_BUTTON_INSERT,              WMain::OnButtonInsert)
    EVT_BUTTON  (ID_BUTTON_ERASE,               WMain::OnButtonErase)
    EVT_BUTTON  (ID_BUTTON_INSERTRANDOM,        WMain::OnButtonInsertRandom)
    EVT_BUTTON  (ID_BUTTON_FINDKEY,             WMain::OnButtonFindKey)
    EVT_BUTTON  (ID_BUTTON_EQUALRANGE,          WMain::OnButtonEqualRange)
    EVT_BUTTON  (ID_BUTTON_CLEAR,               WMain::OnButtonClear)
    EVT_BUTTON  (ID_BUTTON_LOADFILE,            WMain::OnButtonLoadFile)

END_EVENT_TABLE();

// *** Main Application

class AppBTreeDemo : public wxApp
{
public:
    bool                OnInit();
};

IMPLEMENT_APP(AppBTreeDemo)

bool AppBTreeDemo::OnInit()
{
    wxImage::AddHandler(new wxXPMHandler());

    WMain* wm = new WMain();
    SetTopWindow(wm);
    SetExitOnFrameDelete(true);
    wm->Show();

    return true;
}
