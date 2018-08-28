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

//VTK::Base::Dec

//VTK::Termination::Dec

//VTK::Cropping::Dec

//VTK::Shading::Dec

//////////////////////////////////////////////////////////////////////////////
///
/// Inputs
///
//////////////////////////////////////////////////////////////////////////////
attribute vec3 in_vertexPos;


void main()
{
	/// Get clipspace position
	//VTK::ComputeClipPos::Impl

	vec4 posWorld = in_volumeMatrix[in_instanceID] * vec4(in_vertexPos, 1.);

	gl_Position = in_projectionMatrix * in_modelViewMatrix * posWorld;

	if (posWorld.w != 1.) {
		posWorld.xyz /= posWorld.w;
		posWorld.w = 1.;
	}

	for (int i = 0; i < __NUMBER_OF_VOLUMES__; ++i) {
		/// Compute texture coordinates
		//VTK::ComputeTextureCoords::Impl
		vec4 posAdjusted = vec4(in_vertexPos, 1.);
		if (i != in_instanceID) {
			posAdjusted = in_inverseVolumeMatrix[i] * posWorld;
			if (posAdjusted.w != 1.) {
				posAdjusted.xyz /= posAdjusted.w;
				posAdjusted.w = 1.;
			}
		}

		vec3 uvx = sign(in_cellSpacing[i]) * (posAdjusted.xyz - in_volumeExtentsMin[i]) /
				   (in_volumeExtentsMax[i] - in_volumeExtentsMin[i]);

		if (in_cellFlag[i])
		{
			ip_textureCoords[i] = uvx;
			ip_inverseTextureDataAdjusted[i] = in_inverseTextureDatasetMatrix[i];
		}
		else
		{
			// Transform cell tex-coordinates to point tex-coordinates
			ip_textureCoords[i] = (in_cellToPoint[i] * vec4(uvx, 1.)).xyz;
			ip_inverseTextureDataAdjusted[i] = in_cellToPoint[i] * in_inverseTextureDatasetMatrix[i];
		}

		ip_vertexPos[i] = posAdjusted.xyz;
	}
}