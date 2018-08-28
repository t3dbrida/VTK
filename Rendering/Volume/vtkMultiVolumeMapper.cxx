/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkMultiVolumeMapper.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkMultiVolumeMapper.h"

#include "vtkDataSet.h"
#include "vtkExecutive.h"
#include "vtkGarbageCollector.h"
#include "vtkImageData.h"
#include "vtkInformation.h"

//----------------------------------------------------------------------------
// Construct a vtkMultiVolumeMapper with empty scalar input and clipping off.
vtkMultiVolumeMapper::vtkMultiVolumeMapper()
{
  int i;

  this->BlendMode = vtkMultiVolumeMapper::COMPOSITE_BLEND;
  this->AverageIPScalarRange[0] = VTK_DOUBLE_MIN;
  this->AverageIPScalarRange[1] = VTK_DOUBLE_MAX;

  this->Cropping = 0;
  for ( i = 0; i < 3; i++ )
  {
    this->CroppingRegionPlanes[2 * i    ]      = 0;
    this->CroppingRegionPlanes[2 * i + 1]      = 1;
    this->VoxelCroppingRegionPlanes[2 * i]     = 0;
    this->VoxelCroppingRegionPlanes[2 * i + 1] = 1;
  }
  this->CroppingRegionFlags = VTK_CROP_SUBVOLUME;
}

//----------------------------------------------------------------------------
vtkMultiVolumeMapper::~vtkMultiVolumeMapper()
{
}

//----------------------------------------------------------------------------
// JB
// this is called in fixed point volume ray cast mapper so we technically don't need to implement this
void vtkMultiVolumeMapper::ConvertCroppingRegionPlanesToVoxels()
{
  double *spacing = this->GetInput()->GetSpacing();
  for (int i = 1; i < this->GetNumberOfInputConnections(0); ++i)
  {
	  double *tmp = this->GetInput(i)->GetSpacing();
	  if (tmp[0] > spacing[0])
		  spacing[0] = tmp[0];
	  if (tmp[1] > spacing[1])
		  spacing[1] = tmp[1];
	  if (tmp[2] > spacing[2])
		  spacing[2] = tmp[2];
  }

  int dimensions[3];
  this->GetInput()->GetDimensions(dimensions);
  for (int i = 1; i < this->GetNumberOfInputConnections(0); ++i)
  {
	  int tmp[3];
	  this->GetInput(i)->GetDimensions(tmp);
	  if (tmp[0] > dimensions[0])
		  dimensions[0] = tmp[0];
	  if (tmp[1] > dimensions[1])
		  dimensions[1] = tmp[1];
	  if (tmp[2] > dimensions[2])
		  dimensions[2] = tmp[2];
  }

  double origin[3];
  //const double *bds = this->GetInput()->GetBounds(); // XXX ???
  const double *bds = this->GetBounds();
  origin[0] = bds[0];
  origin[1] = bds[2];
  origin[2] = bds[4];

  for ( int i = 0; i < 6; i++ )
  {
    this->VoxelCroppingRegionPlanes[i] =
      (this->CroppingRegionPlanes[i] - origin[i / 2]) / spacing[i / 2];

    this->VoxelCroppingRegionPlanes[i] =
      ( this->VoxelCroppingRegionPlanes[i] < 0 ) ?
      ( 0 ) : ( this->VoxelCroppingRegionPlanes[i] );

    this->VoxelCroppingRegionPlanes[i] =
      ( this->VoxelCroppingRegionPlanes[i] > dimensions[i / 2] - 1 ) ?
      ( dimensions[i / 2] - 1 ) : ( this->VoxelCroppingRegionPlanes[i] );
  }
}

//----------------------------------------------------------------------------
void vtkMultiVolumeMapper::SetInputData( int i, vtkDataSet *genericInput )
{
  vtkImageData *input =
    vtkImageData::SafeDownCast( genericInput );

  if ( input )
  {
    this->SetInputData( i, input );
  }
  else
  {
    vtkErrorMacro("The SetInput method of this mapper requires vtkImageData as input");
  }
}

//----------------------------------------------------------------------------
void vtkMultiVolumeMapper::SetInputData( int i, vtkImageData *input )
{
  this->AddInputDataInternal(0, input);
}

//----------------------------------------------------------------------------
vtkImageData *vtkMultiVolumeMapper::GetInput( int i )
{
  if (this->GetNumberOfInputConnections(0) <= i)
  {
    return 0;
  }
  return vtkImageData::SafeDownCast(
    this->GetExecutive()->GetInputData(0, i));
}

//----------------------------------------------------------------------------
// Print the vtkMultiVolumeMapper
void vtkMultiVolumeMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Cropping: " << (this->Cropping ? "On\n" : "Off\n");

  os << indent << "Cropping Region Planes: " << endl
     << indent << "  In X: " << this->CroppingRegionPlanes[0]
     << " to " << this->CroppingRegionPlanes[1] << endl
     << indent << "  In Y: " << this->CroppingRegionPlanes[2]
     << " to " << this->CroppingRegionPlanes[3] << endl
     << indent << "  In Z: " << this->CroppingRegionPlanes[4]
     << " to " << this->CroppingRegionPlanes[5] << endl;

  os << indent << "Cropping Region Flags: "
     << this->CroppingRegionFlags << endl;

  os << indent << "BlendMode: " << this->BlendMode << endl;

  // Don't print this->VoxelCroppingRegionPlanes
}

//----------------------------------------------------------------------------
int vtkMultiVolumeMapper::FillInputPortInformation(int port, vtkInformation* info)
{
	if (port == 0) {
		if (!this->Superclass::FillInputPortInformation(port, info))
			return 0;
		info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
		return 1;
	}
	return 0;
}

//----------------------------------------------------------------------------
double vtkMultiVolumeMapper::SpacingAdjustedSampleDistance(double inputSpacing[3],
                                                           int inputExtent[6])
{
  // compute 1/2 the average spacing
  double dist =
    (inputSpacing[0] + inputSpacing[1] + inputSpacing[2])/6.0;
  double avgNumVoxels =
    pow(static_cast<double>((inputExtent[1] - inputExtent[0]) *
                            (inputExtent[3] - inputExtent[2]) *
                            (inputExtent[5] - inputExtent[4])),
        static_cast<double>(0.333));

  if (avgNumVoxels < 100)
  {
    dist *= 0.01 + (1 - 0.01) * avgNumVoxels / 100;
  }

  return dist;
}