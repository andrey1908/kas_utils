#pragma once

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Point_3.h>
#include <CGAL/Vector_3.h>
#include <CGAL/Polyhedron_3.h>

namespace kas_utils {

typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3 Point;
typedef K::Vector_3 Vector;
typedef CGAL::Polyhedron_3<K> Polyhedron;

Polyhedron make_parallelepiped(
    const Point& p, const Vector& v1, const Vector& v2, const Vector& v3,
    bool triangle_faces = false);

}
