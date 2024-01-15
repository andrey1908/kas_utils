#include "kas_utils/cgal_utils.h"

#include <CGAL/HalfedgeDS_decorator.h>

#include <iostream>

namespace kas_utils {

Polyhedron make_parallelepiped(
        const Point& p, const Vector& v1, const Vector& v2, const Vector& v3,
        bool triangle_faces /* false */) {
    typedef CGAL::HalfedgeDS_decorator<Polyhedron::HDS> HalfedgeDS_decorator;
    typedef Polyhedron::Halfedge_handle Halfedge_handle;

    Polyhedron parallelepiped;
    if (triangle_faces) {
        parallelepiped.reserve(8, 36, 12);
    } else {
        parallelepiped.reserve(8, 24, 6);
    }

    HalfedgeDS_decorator decorator(parallelepiped.hds());
    Halfedge_handle h1, h2, h3, h4;
    Halfedge_handle g1, g2;
    Halfedge_handle f1, f2;
    Halfedge_handle e1, e2;

    h1 = decorator.create_loop();
    h2 = parallelepiped.split_edge(h1);
    h3 = parallelepiped.split_edge(h2);
    h4 = parallelepiped.split_edge(h3);

    g1 = parallelepiped.split_edge(h1);
    g2 = parallelepiped.split_edge(g1);
    h1 = parallelepiped.split_facet(h2, h1);

    f1 = parallelepiped.split_edge(h3);
    f2 = parallelepiped.split_edge(f1);
    h3 = parallelepiped.split_facet(h4, h3);

    e1 = parallelepiped.split_facet(f1->opposite(), g1->next()->opposite());
    e2 = parallelepiped.split_facet(f1->next()->opposite(), g1->opposite());

    if (triangle_faces) {
        parallelepiped.split_facet(h1, h3);
        parallelepiped.split_facet(g1, g2->prev());
        parallelepiped.split_facet(f2, f1->next());
        parallelepiped.split_facet(e1, e1->next()->next());
        parallelepiped.split_facet(e2, e2->next()->next());
        parallelepiped.split_facet(e1->opposite(), e1->opposite()->next()->next());
    }

    h1->vertex()->point() = p;
    h2->vertex()->point() = p + v1;
    h3->vertex()->point() = p + v1 + v2;
    h4->vertex()->point() = p + v2;
    g1->vertex()->point() = p + v3;
    g2->vertex()->point() = p + v1 + v3;
    f1->vertex()->point() = p + v1 + v2 + v3;
    f2->vertex()->point() = p + v2 + v3;

    return parallelepiped;
}

}
