/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkAbstractMultiVolumeMapper.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkAbstractMultiVolumeMapper.h"

#include "vtkDataSet.h"
#include "vtkExecutive.h"
#include "vtkInformation.h"
#include "vtkMath.h"


//----------------------------------------------------------------------------
// Construct a vtkAbstractMultiVolumeMapper
vtkAbstractMultiVolumeMapper::vtkAbstractMultiVolumeMapper()
{
  vtkMath::UninitializeBounds(this->Bounds);
  this->Center[0] = this->Center[1] = this->Center[2] = 0.0;

  this->ScalarMode = VTK_SCALAR_MODE_DEFAULT;

  this->ArrayName = new char[1];
  this->ArrayName[0] = '\0';
  this->ArrayId = -1;
  this->ArrayAccessMode = VTK_GET_ARRAY_BY_ID;
}

//----------------------------------------------------------------------------
vtkAbstractMultiVolumeMapper::~vtkAbstractMultiVolumeMapper()
{
  delete[] this->ArrayName;
}

//----------------------------------------------------------------------------
// Get the bounds for the input of this mapper as
// (Xmin,Xmax,Ymin,Ymax,Zmin,Zmax).
double *vtkAbstractMultiVolumeMapper::GetBounds()
{
  if ( ! this->GetDataSetInput(0) )
    vtkMath::UninitializeBounds(this->Bounds);
  else
  {
    this->Update();
	this->GetDataSetInput(0)->GetBounds(this->Bounds);
	for (int i = 1; i < this->GetNumberOfInputConnections(0); ++i)
	{
		double bounds[6] = {0.};
		this->GetDataSetInput(i)->GetBounds(bounds);

		if (bounds[0] < this->Bounds[0])
			this->Bounds[0] = bounds[0];
		if (bounds[1] > this->Bounds[1])
			this->Bounds[1] = bounds[1];
		if (bounds[2] < this->Bounds[2])
			this->Bounds[2] = bounds[2];
		if (bounds[3] > this->Bounds[3])
			this->Bounds[3] = bounds[3];
		if (bounds[4] < this->Bounds[4])
			this->Bounds[4] = bounds[4];
		if (bounds[5] > this->Bounds[5])
			this->Bounds[5] = bounds[5];
	}
  }

  return this->Bounds;
}

//----------------------------------------------------------------------------
vtkDataObject *vtkAbstractMultiVolumeMapper::GetDataObjectInput(int i)
{
  if (this->GetNumberOfInputConnections(0) <= i)
  {
    return nullptr;
  }
  return this->GetInputDataObject(0, i);
}

//----------------------------------------------------------------------------
vtkDataSet *vtkAbstractMultiVolumeMapper::GetDataSetInput(int i)
{
  if (this->GetNumberOfInputConnections(0) <= i)
  {
    return nullptr;
  }
  return vtkDataSet::SafeDownCast(this->GetInputDataObject(0, i));
}

//----------------------------------------------------------------------------
int vtkAbstractMultiVolumeMapper::FillInputPortInformation(int port,
                                                           vtkInformation* info)
{
	if (port == 0) {
		info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
		info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1); // allow multiple datasets
		return 1;
	}
	return 0;
}

//----------------------------------------------------------------------------
void vtkAbstractMultiVolumeMapper::SelectScalarArray(int arrayNum)
{
  if ((this->ArrayId == arrayNum) &&
	  (this->ArrayAccessMode == VTK_GET_ARRAY_BY_ID) )
    return;
  this->Modified();

  this->ArrayId = arrayNum;
  this->ArrayAccessMode = VTK_GET_ARRAY_BY_ID;
}

//----------------------------------------------------------------------------
void vtkAbstractMultiVolumeMapper::SelectScalarArray(const char *arrayName)
{
  if (!arrayName || ((strcmp(this->ArrayName, arrayName) == 0) &&
      (this->ArrayAccessMode == VTK_GET_ARRAY_BY_NAME)))
    return;
  this->Modified();

  delete[] this->ArrayName;
  this->ArrayName = new char[strlen(arrayName) + 1];
  strcpy(this->ArrayName, arrayName);
  this->ArrayAccessMode = VTK_GET_ARRAY_BY_NAME;
}

//----------------------------------------------------------------------------
// Return the method for obtaining scalar data.
const char *vtkAbstractMultiVolumeMapper::GetScalarModeAsString(void)
{
  if ( this->ScalarMode == VTK_SCALAR_MODE_USE_CELL_DATA )
  {
    return "UseCellData";
  }
  else if ( this->ScalarMode == VTK_SCALAR_MODE_USE_POINT_DATA )
  {
    return "UsePointData";
  }
  else if ( this->ScalarMode == VTK_SCALAR_MODE_USE_POINT_FIELD_DATA )
  {
    return "UsePointFieldData";
  }
  else if ( this->ScalarMode == VTK_SCALAR_MODE_USE_CELL_FIELD_DATA )
  {
    return "UseCellFieldData";
  }
  else
  {
    return "Default";
  }
}

//----------------------------------------------------------------------------
// Print the vtkAbstractMultiVolumeMapper
void vtkAbstractMultiVolumeMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
  os << indent << "ScalarMode: " << this->GetScalarModeAsString() << endl;

  if ( this->ScalarMode == VTK_SCALAR_MODE_USE_POINT_FIELD_DATA ||
       this->ScalarMode == VTK_SCALAR_MODE_USE_CELL_FIELD_DATA )
  {
    if (this->ArrayAccessMode == VTK_GET_ARRAY_BY_ID)
    {
      os << indent << "ArrayId: " << this->ArrayId << endl;
    }
    else
    {
      os << indent << "ArrayName: " << this->ArrayName << endl;
    }
  }
}