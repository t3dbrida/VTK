// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#ifndef vtkMarchingCellsContourCases_h
#define vtkMarchingCellsContourCases_h

#include "vtkCommonDataModelModule.h"

#include <cstdint> // For uint8_t

/**
 * @class vtkMarchingCellsContourCases
 * @brief Lookup tables for marching cells contouring.
 *
 * vtkMarchingCellsContourCases provides the edge intersection tables used
 * by marching cells algorithms to generate contours (isosurfaces/isolines)
 * for each supported cell type.
 *
 * For each cell type, a case index is computed by treating each vertex as a
 * bit in a bitmask: bit i is set if vertex i's scalar value is greater than
 * or equal to the contour value. This case index is then used to look up a
 * list of edges that the contour surface intersects.
 *
 * Each case entry is an array of edge indices terminated by -1. For surface
 * cells (3D), edges are grouped into triangles (3 edges per triangle). For
 * line cells (2D), edges are grouped into line segments (2 edges per segment).
 * Edge indices refer to the cell type's own edge numbering convention, as
 * returned by GetCellEdges().
 *
 * The number of cases per cell type is 2^N where N is the number of vertices:
 * - Line:        2^2  =   4 cases
 * - Triangle:    2^3  =   8 cases
 * - Pixel/Quad:  2^4  =  16 cases
 * - Tetra:       2^4  =  16 cases
 * - Voxel/Hex:   2^8  = 256 cases
 * - Wedge:       2^6  =  64 cases
 * - Pyramid:     2^5  =  32 cases
 *
 * @sa vtkMarchingCellsClipCases
 */
VTK_ABI_NAMESPACE_BEGIN
class VTKCOMMONDATAMODEL_EXPORT vtkMarchingCellsContourCases
{
public:
  ///@{
  /**
   * Case tables for a line cell (VTK_LINE).
   * Each case entry is an array of 2 ints: one edge index followed by -1.
   * There are 4 cases (2^2 vertices).
   */
  using LineCase = int[2];
  static const LineCase* GetLineCases();
  static const LineCase& GetLineCase(uint8_t caseIndex);
  ///@}

  ///@{
  /**
   * Case tables for a triangle cell (VTK_TRIANGLE).
   * Each case entry is an array of 3 ints: edge indices grouped into line
   * segments, terminated by -1. There are 8 cases (2^3 vertices).
   */
  using TriangleCase = int[3];
  static const TriangleCase* GetTriangleCases();
  static const TriangleCase& GetTriangleCase(uint8_t caseIndex);
  ///@}

  ///@{
  /**
   * Case tables for a pixel cell (VTK_PIXEL).
   * Each case entry is an array of 5 ints: edge indices grouped into line
   * segments, terminated by -1. There are 16 cases (2^4 vertices).
   * Note: vtkPixel uses a different point ordering than vtkQuad
   * (points 2 and 3 are swapped), so this table differs from QuadCases.
   */
  using PixelCase = int[5];
  static const PixelCase* GetPixelCases();
  static const PixelCase& GetPixelCase(uint8_t caseIndex);
  ///@}

  ///@{
  /**
   * Case tables for a quad cell (VTK_QUAD).
   * Each case entry is an array of 5 ints: edge indices grouped into line
   * segments, terminated by -1. There are 16 cases (2^4 vertices).
   */
  using QuadCase = int[5];
  static const QuadCase* GetQuadCases();
  static const QuadCase& GetQuadCase(uint8_t caseIndex);
  ///@}

  ///@{
  /**
   * Case tables for a tetrahedron cell (VTK_TETRA).
   * Each case entry is an array of 7 ints: edge indices grouped into triangles
   * (3 edges per triangle), terminated by -1. There are 16 cases (2^4 vertices).
   */
  using TetraCase = int[7];
  static const TetraCase* GetTetraCases();
  static const TetraCase& GetTetraCase(uint8_t caseIndex);
  ///@}

  ///@{
  /**
   * Case tables for a voxel cell (VTK_VOXEL).
   * Each case entry is an array of 16 ints: edge indices grouped into triangles
   * (3 edges per triangle), terminated by -1. There are 256 cases (2^8 vertices).
   * Note: vtkVoxel uses a different point ordering than vtkHexahedron
   * (points 2 and 3 are swapped, as are points 6 and 7), so this table
   * differs from HexahedronCases.
   */
  using VoxelCase = int[16];
  static const VoxelCase* GetVoxelCases();
  static const VoxelCase& GetVoxelCase(uint8_t caseIndex);
  ///@}

  ///@{
  /**
   * Case tables for a hexahedron cell (VTK_HEXAHEDRON).
   * Each case entry is an array of 16 ints: edge indices grouped into triangles
   * (3 edges per triangle), terminated by -1. There are 256 cases (2^8 vertices).
   */
  using HexahedronCase = int[16];
  static const HexahedronCase* GetHexahedronCases();
  static const HexahedronCase& GetHexahedronCase(uint8_t caseIndex);
  ///@}

  ///@{
  /**
   * Case tables for a hexahedron cell with polygon output (VTK_HEXAHEDRON).
   * Each case entry is an array of 17 ints: edge indices grouped into polygons,
   * terminated by -1. There are 256 cases (2^8 vertices).
   */
  using HexahedronWithPolygonCase = int[17];
  static const HexahedronWithPolygonCase* GetHexahedronWithPolygonCases();
  static const HexahedronWithPolygonCase& GetHexahedronWithPolygonCase(uint8_t caseIndex);
  ///@}

  ///@{
  /**
   * Case tables for a wedge cell (VTK_WEDGE).
   * Each case entry is an array of 13 ints: edge indices grouped into triangles
   * (3 edges per triangle), terminated by -1. There are 64 cases (2^6 vertices).
   */
  using WedgeCase = int[13];
  static const WedgeCase* GetWedgeCases();
  static const WedgeCase& GetWedgeCase(uint8_t caseIndex);
  ///@}

  ///@{
  /**
   * Case tables for a pyramid cell (VTK_PYRAMID).
   * Each case entry is an array of 13 ints: edge indices grouped into triangles
   * (3 edges per triangle), terminated by -1. There are 32 cases (2^5 vertices).
   */
  using PyramidCase = int[13];
  static const PyramidCase* GetPyramidCases();
  static const PyramidCase& GetPyramidCase(uint8_t caseIndex);
  ///@}

  /**
   * Generic interface to retrieve a contour case entry for any supported cell type.
   * @param cellType A VTK cell type constant (e.g. VTK_HEXAHEDRON, VTK_WEDGE).
   * @param caseIndex The case index computed from the vertex inside/outside bitmask.
   * @return A pointer to the first edge index in the case entry, terminated by -1,
   *         or a pointer to -1 if the cell type is not supported.
   */
  using CellCase = const int*;
  static CellCase GetCellCase(int cellType, uint8_t caseIndex);

  /**
   * Returns the edge definitions for the given cell type as an array of
   * (point index pairs), where each pair defines one edge of the cell.
   * The array is terminated by a { -1, -1 } sentinel.
   * Returns nullptr if the cell type is not supported.
   */
  using Edge = int[2];
  using EdgeArray = const Edge*;
  static EdgeArray GetCellEdges(int cellType);
};
VTK_ABI_NAMESPACE_END

#endif // vtkMarchingCellsContourCases_h
