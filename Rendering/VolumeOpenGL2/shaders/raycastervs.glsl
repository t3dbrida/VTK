//VTK::System::Dec

/*=========================================================================

  Program:   Visualization Toolkit
  Module:    raycastervs.glsl

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/// Needed to enable inverse function
#extension GL_ARB_gpu_shader5 : enable

//////////////////////////////////////////////////////////////////////////////
///
/// Uniforms, attributes, and globals
///
//////////////////////////////////////////////////////////////////////////////
//VTK::CustomUniforms::Dec

//VTK::Base::Dec

//VTK::Termination::Dec

//VTK::Cropping::Dec

//VTK::Shading::Dec

struct VolumeParameters
{
  vec4 boundsMin,
       boundsMax,
       boxMaskOrigin,
       boxMaskAxisX,
       boxMaskAxisY,
       boxMaskAxisZ,
       cylinderMaskCenter,
       cylinderMaskAxis_cylinderMaskRadius,
       volumeScale,
       volumeBias;

  ivec4 noOfComponents_maskIndex_regionIndex_transfer2dIndex;

  mat4 volumeMatrix,
       inverseVolumeMatrix,
       textureDatasetMatrix,
       inverseTextureDatasetMatrix,
       textureToEye,
       cellToPoint;

  vec4 texMin,
	   texMax,
       cellStep,
       cellSpacing,
       transfer2dRegion,
       scalarsRange_gradMagMax_sampling;

  uvec4 volumeVisibility;
};

layout (std430, binding = 0) buffer VP
{
    VolumeParameters data[];
} volumeParameters;


//////////////////////////////////////////////////////////////////////////////
///
/// Inputs
///
//////////////////////////////////////////////////////////////////////////////
in vec3 in_vertexPos;

//////////////////////////////////////////////////////////////////////////////
///
/// Outputs
///
//////////////////////////////////////////////////////////////////////////////
/// 3D texture coordinates for texture lookup in the fragment shader
out vec3 ip_textureCoords;
out vec3 ip_vertexPos;

void main()
{
  /// Get clipspace position
  //VTK::ComputeClipPos::Impl

  /// Compute texture coordinates
  //VTK::ComputeTextureCoords::Impl

  /// Copy incoming vertex position for the fragment shader
  ip_vertexPos = in_vertexPos;
}
