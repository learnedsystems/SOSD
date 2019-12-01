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

#ifndef _WTreeDrawing_H_
#define _WTreeDrawing_H_

#include <wx/wx.h>

/** The Custom wxScrolledWindow Canvas on which the B+ tree is drawng. It
 * supports zooming via mouse wheel and scrolling from wxScrolledWindow. */
class WTreeDrawing : public wxScrolledWindow
{
public:
    WTreeDrawing(wxWindow *parent, int id);

    /// Used to determine when to update the scroll bars.
    wxSize              oldTreeSize;
    double              oldscalefactor;

    /// Zoom factor changed by the mouse wheel.
    double              scalefactor;

    /// Set if this windows has focus and draw a faint frame around it.
    bool                hasfocus;

    void                OnPaint(wxPaintEvent &pe);
    void                OnSize(wxSizeEvent &se);
    void                OnMouseWheel(wxMouseEvent &me);
    void                OnSetFocus(wxFocusEvent &fe);
    void                OnKillFocus(wxFocusEvent &fe);

    void                DrawBTree(wxDC &dc);

    /// Tree operation to draw the nodes on this canvas.
    struct BTreeOp_Draw
    {
        BTreeOp_Draw(WTreeDrawing &_w, wxDC &_dc, const class BTreeBundle &_tb)
            : w(_w), dc(_dc), tb(_tb)
        {
        }

        WTreeDrawing &w;
        wxDC &dc;
        const BTreeBundle &tb;

        typedef wxSize  result_type;

        template <class BTreeType>
        wxSize draw_node(int offsetx, int offsety, const class BTreeType::btree_impl::node* node);

        template <class BTreeType>
        wxSize draw_tree(BTreeType &bt);

        template <class BTreeType>
        wxSize opInteger(BTreeType &bt);

        template <class BTreeType>
        wxSize opIntegerMulti(BTreeType &bt);

        template <class BTreeType>
        wxSize opString(BTreeType &bt);

        template <class BTreeType>
        wxSize opStringMulti(BTreeType &bt);
    };

    class WMain*        wmain;
    void                SetWMain(class WMain *wm);

    DECLARE_EVENT_TABLE();
};

#endif // _WTreeDrawing_H_
