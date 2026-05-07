"""
Unit Tests for `pointTriangleDistance` Function

This module contains a series of tests to validate the correctness and robustness
of the `pointTriangleDistance` function, which computes the shortest Euclidean
distance from a point to a triangle in 3D space.

Tested Function:
----------------
pointTriangleDistance(P, TRI)
    - Computes the minimum distance from a point `P` to a given triangle `TRI`.

Test Methodology:
-----------------
The tests cover multiple edge cases to ensure the function performs correctly
in different geometric scenarios. The following test cases are included:

1️⃣ **Point above the triangle**
   - The point is projected onto the plane and falls inside the triangle.
   - Expected output: Perpendicular distance.

2️⃣ **Point inside the triangle**
   - The point is directly within the triangle’s boundary.
   - Expected output: Distance = 0.

3️⃣ **Point on a triangle vertex**
   - The point coincides with one of the triangle's vertices.
   - Expected output: Distance = 0.

4️⃣ **Point on a triangle edge**
   - The point lies exactly on one of the triangle's edges.
   - Expected output: Distance = 0.

5️⃣ **Point far away from the triangle**
   - The point is located at a significant distance from the triangle.
   - Expected output: The shortest distance to the closest vertex or edge.

6️⃣ **Point below the triangle**
   - The point is projected onto the triangle’s plane but falls outside.
   - Expected output: Shortest distance to an edge or vertex.

7️⃣ **Degenerate triangle (all vertices are the same)**
   - A collapsed triangle where all three vertices coincide.
   - Expected output: Distance to the single vertex.

8️⃣ **Triangle rotated vertically, point above**
   - The triangle is not parallel to the XY plane.
   - Expected output: The perpendicular distance to the closest point on the plane.

9️⃣ **Point near but outside the triangle**
   - The point is close to a vertex but not inside the triangle.
   - Expected output: Shortest distance to a vertex.

🔟 **Point on the same plane but outside the triangle**
   - The point is on the same plane as the triangle but outside its bounds.
   - Expected output: Distance to the closest edge or projected point.

1️⃣1️⃣ **Collinear vertices (degenerate but distinct)**
   -The triangle is reduced to a straight line segment.
   - Expected output: Distance to the closest segment.

1️⃣2️⃣ Triangle in an arbitrary plane (not aligned with axes)
    - The triangle is rotated in 3D space.
    - Expected output: perpendicular distance to the closest triangle feature.

1️⃣3️⃣ Point exactly on an edge but not at a vertex
    - The point lies on an edge but not at a vertex.
    - Expected output: Distance = 0.

1️⃣4️⃣ Extreme large values for numerical stability
    - The point and triangle coordinates are very large.
    - Expected output: Distance to the closest vertex or edge with floating point issue.

Execution:
----------
Run the unit test using Python's built-in unittest framework:

        python -m unittest discover tests

Expected Behavior:
------------------
- All tests should pass without failures.
- The function should handle degenerate cases gracefully.
- Numerical stability is ensured by checking results within a small tolerance (`1e-5`).

Example:
--------
    P = np.array([0.5, 0.5, 1.0])
    TRI = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    pointTriangleDistance(P, TRI)
    1.0
--------
"""

import unittest
import numpy as np
from facebenchmark.performance_reporters.error_computation.distance_computers import pointTriangleDistance


class TestPointTriangleDistance(unittest.TestCase):

    def setUp(self):
        # Defines a set of test cases for different scenarios.
        self.test_cases = [
            # 🔹 Case 1: Point above the triangle
            {"P": np.array([0.5, 0.5, 1.0]),
             "TRI": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
             "expected": 1.0},

            # 🔹 Case 2: Point inside the triangle
            {"P": np.array([0.3, 0.3, 0.0]),
             "TRI": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
             "expected": 0.0},

            # 🔹 Case 3: Point coinciding with a vertex
            {"P": np.array([0, 0, 0]),
             "TRI": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
             "expected": 0.0},

            # 🔹 Case 4: Point on a triangle edge
            {"P": np.array([0.5, 0.0, 0.0]),
             "TRI": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
             "expected": 0.0},

            # 🔹 Case 5: Point far away from the triangle
            {"P": np.array([10, 10, 10]),
             "TRI": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
             "expected": np.sqrt(280.5)},  # result of all the squares of the coordinates

            # 🔹 Case 6: Point below the triangle
            {"P": np.array([0.5, 0.5, -1.0]),
             "TRI": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
             "expected": 1.0},

            # 🔹 Case 7: Degenerate triangle (all vertices identical)
            {"P": np.array([1, 1, 1]),
             "TRI": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
             "expected": np.linalg.norm([1, 1, 1])},  # Distance from the origin

            # 🔹 Case 8: Vertical triangle, point above
            {"P": np.array([0.5, 0.5, 2]),
             "TRI": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 2]]),
             "expected": np.sqrt(1 / 3)},  # results of all the squares of the coordinates

            # 🔹 Case 9: Point very close to a vertex
            {"P": np.array([0.99, 0.01, 0.0]),
             "TRI": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
             "expected": 0.0},

            # 🔹 Case 10: Point on the same plane but outside the triangle
            {"P": np.array([5, 5, 0]),
             "TRI": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
             "expected": np.linalg.norm([5, 5, 0] - np.array([0.5, 0.5, 0]))},  # Closest point projection

            # 🔹 Case 11: Collinear vertices (degenerate but distinct)
            {"P": np.array([1, 2, 0]),
             "TRI": np.array([[0, 0, 0],
                              [1, 1, 0],
                              [2, 2, 0]]),
             "expected": np.sqrt(2)/2},

            # 🔹 Case 12: Triangle in an arbitrary plane (not aligned with axes)
            {"P": np.array([1, 1, 1]),
             "TRI": np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]),
             "expected": 2 / np.sqrt(3)},

            # 🔹 Case 13: Point exactly on the edge (non-vertex)
            {"P": np.array([0.5, 0.5, 0]),
             "TRI": np.array([[0, 0, 0],
                              [1, 0, 0],
                              [0, 1, 0]]),
             "expected": 0.0},

            # 🔹 Case 14: Extreme large values for numerical stability
            {"P": np.array([1e9, 1e9, 1e9]),
             "TRI": np.array([[0, 0, 0],
                              [1, 0, 0],
                              [0, 1, 0]]),
             "expected": np.linalg.norm(np.array([1e9, 1e9, 1e9]) - np.array([0.5, 0.5, 0]))},
            # We subtract (0.5, 0.5, 0) because it represents the closest point inside the triangle,
            # around which the projection should occur.
        ]

    def test_point_triangle_distance(self):
        # Runs each test case and checks if the computed distance matches the expected result.
        for i, test in enumerate(self.test_cases):
            with self.subTest(i=i):

                P, TRI, expected = test["P"], test["TRI"], test["expected"]
                result = pointTriangleDistance(P, TRI)
                print(f"\n🔍 **Test {i + 1}**")
                print(f"📌 Point: {P}")
                print(f"📌 Triangle:\n{TRI}")
                print(f"✅ Computed Distance: {result:.6f}")
                print(f"🎯 Expected Distance: {expected:.6f}")
                print("✅ Passed" if np.isclose(result, expected, atol=1e-5) else "❌ Failed", "\n")

                self.assertAlmostEqual(result, expected, places=5)

            if not np.isclose(result, expected, atol=1e-5):
                print(f"🔍 DEBUG: Test {i + 1} failed.")
                print(f"🔍 Expected distance: {expected}")
                print(f"🔍 Computed distance: {result}")
                print(f"🔍 Difference: {abs(result - expected)}")


if __name__ == "__main__":
    unittest.main()
