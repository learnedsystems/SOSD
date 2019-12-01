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

#include "WTreeDrawing.h"
#include "WMain.h"

#include <vector>

WTreeDrawing::WTreeDrawing(wxWindow *parent, int id)
    : wxScrolledWindow(parent, id),
      wmain(NULL)
{
    SetWindowStyle(wxWANTS_CHARS);
    SetSize(300, 300);

    scalefactor = 1.0;
    hasfocus = false;
}

void WTreeDrawing::SetWMain(WMain *wm)
{
    wmain = wm;
}

void WTreeDrawing::OnPaint(wxPaintEvent &)
{
    wxPaintDC dc(this);

    DoPrepareDC(dc);
    dc.SetUserScale(scalefactor, scalefactor);

    DrawBTree(dc);
}

void WTreeDrawing::OnSize(wxSizeEvent &se)
{
    Refresh();
}

void WTreeDrawing::OnSetFocus(wxFocusEvent &fe)
{
    hasfocus = true;
    Refresh();
}

void WTreeDrawing::OnKillFocus(wxFocusEvent &fe)
{
    hasfocus = false;
    Refresh();
}

void WTreeDrawing::OnMouseWheel(wxMouseEvent &me)
{
    scalefactor += 0.05 * me.GetWheelRotation() / me.GetWheelDelta();
    scalefactor = std::max(0.1, scalefactor);
    Refresh();
}

template <class BTreeType>
wxSize WTreeDrawing::BTreeOp_Draw::draw_node(int offsetx, int offsety, const class BTreeType::btree_impl::node* node)
{
    typedef class BTreeType::btree_impl btree_impl;

    static const wxColor colorMark1 = wxColor(128, 179, 255);
    static const wxColor colorMark2 = wxColor(128, 255, 128);
    static const wxColor colorUnfocused = *wxWHITE;
    static const wxColor colorFocused = wxColor(255, 255, 253);

    const int textpadding = 3;
    const int nodepadding = 10;

    if (node->isleafnode())
    {
        const typename btree_impl::leaf_node *leafnode = static_cast<const typename btree_impl::leaf_node*>(node);

        int textx = 0, texty = 0;
        int maxh = 0;
        for (unsigned int slot = 0; slot < leafnode->slotuse; ++slot)
        {
            int textkeyw, textkeyh;
            int textvalw, textvalh;

            wxString textkey;
            textkey << leafnode->slotkey[slot];
            dc.GetTextExtent(textkey, &textkeyw, &textkeyh);

            wxString textval;
            textval << leafnode->slotdata[slot];
            dc.GetTextExtent(textval, &textvalw, &textvalh);

            int maxw = std::max(textkeyw, textvalw);
            int addkeyw = (maxw - textkeyw) / 2;
            int addvalw = (maxw - textvalw) / 2;

            if (offsetx >= 0)
            {
                if (tb.isMark1(leafnode, slot))
                {
                    dc.SetBrush(colorMark1);
                }
                else if (tb.isMark2(leafnode, slot))
                {
                    dc.SetBrush(colorMark2);
                }
                else
                {
                    if (w.hasfocus) {
                        dc.SetBrush(colorFocused);
                    }
                    else {
                        dc.SetBrush(colorUnfocused);
                    }
                }

                dc.DrawRectangle(offsetx + textx, offsety + texty,
                                 maxw + 2*textpadding, textkeyh + 2*textpadding);

                dc.DrawText(textkey,
                            offsetx + textx + textpadding + addkeyw,
                            offsety + texty + textpadding);

                dc.DrawRectangle(offsetx + textx, offsety + texty + textkeyh + 2*textpadding - 1,
                                 maxw + 2*textpadding, textvalh + 2*textpadding);

                dc.DrawText(textval,
                            offsetx + textx + textpadding + addvalw,
                            offsety + texty + textkeyh + 2*textpadding + textpadding - 1);
            }

            textx += maxw + 2*textpadding - 1;
            maxh = std::max(maxh, textkeyh + 2*textpadding + textvalh + 2*textpadding - 1);
        }

        return wxSize(textx, maxh);
    }
    else
    {
        const typename btree_impl::inner_node *innernode = static_cast<const typename btree_impl::inner_node*>(node);

        const int childnum = (innernode->slotuse + 1);
        // find the maximium width and height of all children
        int childmaxw = 0, childmaxh = 0;

        std::vector<int> childw;
        for (unsigned int slot = 0; slot <= innernode->slotuse; ++slot)
        {
            wxSize cs = draw_node<BTreeType>(-1, -1, innernode->childid[slot]);
            childmaxw = std::max(childmaxw, cs.GetWidth());
            childmaxh = std::max(childmaxh, cs.GetHeight());
            childw.push_back(cs.GetWidth());
        }

        int textx = 0, texty = 0;
        int childx = 0, childy = 60;
        int maxh = 0;

        // calculate width of box.
        int allchildw = (childnum + 1) * (childmaxw / 2 + nodepadding) - 2*nodepadding;

        if (childnum % 2 == 0)
        {
            allchildw += (childmaxw / 2 + nodepadding);
        }

        // debug rectangle for total size of children dc.DrawRectangle(childx, childy-5, allchildw, 2);

        // calc width of node's keys box
        int keyboxh = 0;
        for (unsigned int slot = 0; slot < innernode->slotuse; ++slot)
        {
            int textkeyw, textkeyh;

            wxString textkey;
            textkey << innernode->slotkey[slot];
            dc.GetTextExtent(textkey, &textkeyw, &textkeyh);

            textx += textkeyw + 2*textpadding - 1;
            keyboxh = std::max(keyboxh, textkeyh + 2*textpadding - 1);
        }

        textx = std::max(0, allchildw - textx) / 2;
        childy = keyboxh + nodepadding*4;

        for (unsigned int slot = 0; slot < innernode->slotuse; ++slot)
        {
            int textkeyw, textkeyh;

            wxString textkey;
            textkey << innernode->slotkey[slot];
            dc.GetTextExtent(textkey, &textkeyw, &textkeyh);

            if (offsetx >= 0)
            {
                if (w.hasfocus) {
                    dc.SetBrush(colorFocused);
                }
                else {
                    dc.SetBrush(colorUnfocused);
                }

                dc.DrawRectangle(offsetx + textx, offsety + texty,
                                 textkeyw + 2*textpadding, textkeyh + 2*textpadding);
                dc.DrawText(textkey,
                            offsetx + textx + textpadding,
                            offsety + texty + textpadding);

                // draw child
                if (innernode->level == 1)
                {
                    // draw leaf node with border to see free slots
                    dc.DrawRectangle(offsetx + childx, offsety + childy,
                                     childmaxw, childmaxh);

                    draw_node<BTreeType>(offsetx + childx, offsety + childy, innernode->childid[slot]);
                }
                else
                {
                    // draw centered inner node
                    draw_node<BTreeType>(offsetx + childx + std::max(0, (childmaxw - childw[slot]) / 2),
                                         offsety + childy,
                                         innernode->childid[slot]);
                }

                // calculate spline from key anchor to middle of child's box
                wxPoint splinept[4];
                splinept[0] = wxPoint(offsetx + textx, offsety + texty + keyboxh);
                splinept[1] = wxPoint(offsetx + textx, offsety + texty + keyboxh + 20);

                splinept[2] = wxPoint(offsetx + childx + childmaxw / 2, offsety + childy - 20);
                splinept[3] = wxPoint(offsetx + childx + childmaxw / 2, offsety + childy);

                dc.DrawSpline(4, splinept);
            }

            // advance text position
            textx += textkeyw + 2*textpadding - 1;

            // advance child position
            childx += childmaxw / 2 + nodepadding;

            if ((slot+1) * 2 == childnum)
            {
                childx += childmaxw / 2 + nodepadding;
            }
            else if (slot < innernode->slotuse / 2)
            {
                childy += childmaxh + nodepadding;
            }
            else
                childy -= childmaxh + nodepadding;

            maxh = std::max(maxh, childy + childmaxh);
        }

        if (offsetx >= 0)
        {
            // draw child
            if (innernode->level == 1)
            {
                // draw leaf node with border to see free slots
                if (w.hasfocus) {
                    dc.SetBrush(colorFocused);
                }
                else {
                    dc.SetBrush(colorUnfocused);
                }

                dc.DrawRectangle(offsetx + childx, offsety + childy,
                                 childmaxw, childmaxh);

                draw_node<BTreeType>(offsetx + childx, offsety + childy, innernode->childid[innernode->slotuse]);
            }
            else
            {
                // draw centered inner node
                draw_node<BTreeType>(offsetx + childx + std::max(0, (childmaxw - childw[innernode->slotuse]) / 2),
                                     offsety + childy,
                                     innernode->childid[innernode->slotuse]);
            }

            // calculate spline from key anchor to middle of child's box
            wxPoint splinept[4];
            splinept[0] = wxPoint(offsetx + textx, offsety + texty + keyboxh);
            splinept[1] = wxPoint(offsetx + textx, offsety + texty + keyboxh + 20);

            splinept[2] = wxPoint(offsetx + childx + childmaxw / 2, offsety + childy - 20);
            splinept[3] = wxPoint(offsetx + childx + childmaxw / 2, offsety + childy);

            dc.DrawSpline(4, splinept);
        }

        return wxSize(allchildw, maxh);
    }
}

template <class BTreeType>
wxSize WTreeDrawing::BTreeOp_Draw::draw_tree(BTreeType &bt)
{
    dc.SetPen(*wxBLACK_PEN);

    if (bt.tree.m_root)
    {
        // draw tree data items in different font sizes depending on the depth
        // of the tree
        if (bt.tree.m_root->level <= 1)
        {
            dc.SetFont(wxFont(14, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        }
        else if (bt.tree.m_root->level <= 2)
        {
            dc.SetFont(wxFont(12, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        }
        else if (bt.tree.m_root->level <= 3)
        {
            dc.SetFont(wxFont(10, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        }
        else {
            dc.SetFont(wxFont(8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        }

        const int offsety = 4;

        // calculate width of the drawn tree
        wxSize ts = draw_node<BTreeType>(-1, -1, bt.tree.m_root);

        if (ts.GetWidth() < w.GetSize().GetWidth())
        {
            // center small trees on the current view area
            ts = draw_node<BTreeType>((w.GetSize().GetWidth() - ts.GetWidth()) / 2, offsety, bt.tree.m_root);
        }
        else
        {
            ts = draw_node<BTreeType>(0, offsety, bt.tree.m_root);
        }

        if (ts != w.oldTreeSize || w.scalefactor != w.oldscalefactor)
        {
            // set scroll bar extents
            int scrx, scry;
            w.GetViewStart(&scrx, &scry);
            w.SetScrollbars(10, 10,
                            int(ts.GetWidth() / 10 * w.scalefactor),
                            int(ts.GetHeight() / 10 * w.scalefactor), scrx, scry);
            w.oldTreeSize = ts;
            w.oldscalefactor = w.scalefactor;
        }
    }
    else
    {
        w.SetScrollbars(10, 10, 0, 0);
    }
}

template <class BTreeType>
wxSize WTreeDrawing::BTreeOp_Draw::opInteger(BTreeType &bt)
{
    draw_tree(bt);
}

template <class BTreeType>
wxSize WTreeDrawing::BTreeOp_Draw::opIntegerMulti(BTreeType &bt)
{
    draw_tree(bt);
}

template <class BTreeType>
wxSize WTreeDrawing::BTreeOp_Draw::opString(BTreeType &bt)
{
    draw_tree(bt);
}

template <class BTreeType>
wxSize WTreeDrawing::BTreeOp_Draw::opStringMulti(BTreeType &bt)
{
    draw_tree(bt);
}

void WTreeDrawing::DrawBTree(wxDC &dc)
{
    if (!wmain) return;

    BTreeOp_Draw drawop(*this, dc, wmain->treebundle);
    wmain->treebundle.run(drawop);
}

BEGIN_EVENT_TABLE(WTreeDrawing, wxScrolledWindow)

    EVT_PAINT           (WTreeDrawing::OnPaint)
    EVT_SIZE            (WTreeDrawing::OnSize)
    EVT_MOUSEWHEEL      (WTreeDrawing::OnMouseWheel)

    EVT_SET_FOCUS       (WTreeDrawing::OnSetFocus)
    EVT_KILL_FOCUS      (WTreeDrawing::OnKillFocus)

END_EVENT_TABLE()
