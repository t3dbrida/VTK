//VTK::System::Dec

/*=========================================================================

  Program:   Visualization Toolkit
  Module:    raycasterfs.glsl

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

struct vec2_4 {
	vec2 data[4];
};

struct vec3_4 {
	vec3 data[4];
};

struct float_4 {
	float data[4];
};

//////////////////////////////////////////////////////////////////////////////
///
/// Inputs
///
//////////////////////////////////////////////////////////////////////////////

/// 3D texture coordinates form vertex shader
varying vec3 ip_textureCoords[__NUMBER_OF_VOLUMES__];
varying vec3 ip_vertexPos[__NUMBER_OF_VOLUMES__];
//varying vec3 ip_vertexPosWorld;

//////////////////////////////////////////////////////////////////////////////
///
/// Outputs
///
//////////////////////////////////////////////////////////////////////////////

vec4 g_fragColor = vec4(0.0);

//////////////////////////////////////////////////////////////////////////////
///
/// Uniforms, attributes, and globals
///
//////////////////////////////////////////////////////////////////////////////
vec3 g_dataPos[__NUMBER_OF_VOLUMES__];
vec3 g_dirStep[__NUMBER_OF_VOLUMES__];
float g_currentT;
float g_terminatePointMax;
vec3 g_xvec[__NUMBER_OF_VOLUMES__];
vec3 g_yvec[__NUMBER_OF_VOLUMES__];
vec3 g_zvec[__NUMBER_OF_VOLUMES__];
const float g_opacityThreshold = 1. - 1. / 255.;

uniform int in_instanceID;
uniform vec4 in_volume_scale[__NUMBER_OF_VOLUMES__];
uniform vec4 in_volume_bias[__NUMBER_OF_VOLUMES__];

//VTK::Output::Dec

//VTK::Base::Dec

//VTK::Termination::Dec

//VTK::Cropping::Dec

//VTK::Clipping::Dec

//VTK::Shading::Dec

//VTK::BinaryMask::Dec

//VTK::CompositeMask::Dec

//VTK::ComputeOpacity::Dec

//VTK::ComputeGradient::Dec

//VTK::ComputeLighting::Dec

//VTK::ComputeColor::Dec

//VTK::ComputeRayDirection::Dec
vec3 computeRayDirection(const int i, const vec4 eyePosObj)
{
	return normalize(ip_vertexPos[i].xyz - eyePosObj.xyz);
}

//VTK::Picking::Dec

//VTK::RenderToImage::Dec

//VTK::DepthPeeling::Dec

/// We support only 8 clipping planes for now
/// The first value is the size of the data array for clipping
/// planes (origin, normal)
uniform float in_clippingPlanes[49];
uniform float in_scale;
uniform float in_bias;

//////////////////////////////////////////////////////////////////////////////
///
/// Helper functions
///
//////////////////////////////////////////////////////////////////////////////

/**
 * Transform window coordinate to NDC.
 */
vec4 WindowToNDC(const float xCoord, const float yCoord, const float zCoord)
{
	vec4 NDCCoord = vec4(0., 0., 0., 1.);

	NDCCoord.x = (xCoord - in_windowLowerLeftCorner.x) * 2. *
	             in_inverseWindowSize.x - 1.;
	NDCCoord.y = (yCoord - in_windowLowerLeftCorner.y) * 2. *
	             in_inverseWindowSize.y - 1.;
	NDCCoord.z = (2. * zCoord - (gl_DepthRange.near + gl_DepthRange.far)) /
	             gl_DepthRange.diff;

	return NDCCoord;
}

/**
 * Transform NDC coordinate to window coordinates.
 */
vec4 NDCToWindow(const float xNDC, const float yNDC, const float zNDC)
{
	vec4 WinCoord = vec4(0., 0., 0., 1.);

	WinCoord.x = (xNDC + 1.) / (2. * in_inverseWindowSize.x) +
	             in_windowLowerLeftCorner.x;
	WinCoord.y = (yNDC + 1.) / (2. * in_inverseWindowSize.y) +
	             in_windowLowerLeftCorner.y;
	WinCoord.z = (zNDC * gl_DepthRange.diff +
	             (gl_DepthRange.near + gl_DepthRange.far)) / 2.;

	return WinCoord;
}

//////////////////////////////////////////////////////////////////////////////
///
/// Ray-casting
///
//////////////////////////////////////////////////////////////////////////////

bool initializeRayCast()
{
	/// Initialize g_fragColor (output) to 0
	g_fragColor = vec4(0.);

	//VTK::Base::Init

	for (int i = 0; i < __NUMBER_OF_VOLUMES__; ++i) {
		vec4 eyePosObj;
		g_dataPos[i] = ip_textureCoords[i];
		eyePosObj = in_inverseVolumeMatrix[i] * vec4(in_cameraPos, 1.);
		if (eyePosObj.w != 0.) {
			eyePosObj.xyz /= eyePosObj.w;
			eyePosObj.w = 1.;
		}
		vec3 rayDir = computeRayDirection(i, eyePosObj);
		g_dirStep[i] = (ip_inverseTextureDataAdjusted[i] * vec4(rayDir, 0.)).xyz * in_sampleDistance[0];
		g_dataPos[i] += g_dirStep[i];

		g_lightPosObj[i] = eyePosObj;
		g_ldir[i] = normalize(g_lightPosObj[i].xyz - ip_vertexPos[i]);
		g_vdir[i] = normalize(eyePosObj.xyz - ip_vertexPos[i]);
		g_h[i] = normalize(g_ldir[i] + g_vdir[i]);
		g_xvec[i] = vec3(in_cellStep[i].x, 0., 0.);
		g_yvec[i] = vec3(0., in_cellStep[i].y, 0.);
		g_zvec[i] = vec3(0., 0., in_cellStep[i].z);
	}

	//VTK::Terminate::Init
	// color buffer or max scalar buffer have a reduced size.
	vec2 fragTexCoord = (gl_FragCoord.xy - in_windowLowerLeftCorner) *
	                    in_inverseWindowSize; 

	// Compute max number of iterations it will take before we hit
	// the termination point

	// Abscissa of the point on the depth buffer along the ray.
	// point in texture coordinates
	vec4 l_depthValue = texture2D(in_depthSampler, fragTexCoord);
	if (gl_FragCoord.z >= l_depthValue.x) {
		discard;
		return false;
	}
	vec4 fragCoord = WindowToNDC(gl_FragCoord.x, gl_FragCoord.y, l_depthValue.x);

	// From normalized device coordinates to eye coordinates.
	// in_projectionMatrix is inversed because of way VT
	// From eye coordinates to texture coordinates
	vec4 terminatePoint = ip_inverseTextureDataAdjusted[in_instanceID] *
	                      in_inverseVolumeMatrix[in_instanceID] *
	                      in_inverseModelViewMatrix *
	                      in_inverseProjectionMatrix *
	                      fragCoord;
	terminatePoint /= terminatePoint.w;
	terminatePoint.w = 1.;

	g_terminatePointMax = length(terminatePoint.xyz - g_dataPos[in_instanceID].xyz) /
	                      length(g_dirStep[in_instanceID]);
	g_currentT = 0.;

	return true;
}

/**
 * March along the ray direction sampling the volume texture.  This function
 * takes a start and end point as arguments but it is up to the specific render
 * pass implementation to use these values (e.g. vtkDualDepthPeelingPass). The
 * mapper does not use these values by default, instead it uses the number of
 * steps defined by g_terminatePointMax.
 */
vec4 castRay(const float zStart, const float zEnd)
{
	bool hit = false;
	for (bool withinBounds = true; withinBounds;) {
		withinBounds = false; // assume the traversal will end

		for (int i = 0; i < __NUMBER_OF_VOLUMES__; ++i) {
			if (all(lessThanEqual(g_dataPos[i], in_texMax[i])) &&
			    all(greaterThanEqual(g_dataPos[i], in_texMin[i]))) {
				withinBounds = true; // traversal continues
				// sample
				vec4 src = vec4(0.);
				vec4 scalar = texture3D(in_volume[i], g_dataPos[i]);
				// adjust the scalar value
				scalar = vec4(scalar.r * in_volume_scale[i].r + in_volume_bias[i].r);
				src.a = computeOpacity(i, scalar);
				if (src.a > 0.) {
					src = computeColor(i, scalar, src.a);
					src.rgb *= src.a;
					// front-to-back compositing
					g_fragColor += (1. - g_fragColor.a) * src;
					hit = true;
				}
			}

			/// Advance ray
			g_dataPos[i] += g_dirStep[i];
		}

		// Early ray termination
		// if the currently composited colour alpha is already fully saturated
		// we terminated the loop or if we have hit an obstacle in the
		// direction of they ray (using depth buffer) we terminate as well.
		if (g_fragColor.a > g_opacityThreshold ||
		    g_currentT >= g_terminatePointMax)
			break;
		++g_currentT;
	}
	
	if (hit == false)
		discard;

	return g_fragColor;
}

/**
 * Finalize specific modes and set output data.
 */
void finalizeRayCast()
{
	//VTK::Base::Exit

	//VTK::Terminate::Exit

	//VTK::Cropping::Exit

	//VTK::Clipping::Exit

	//VTK::Picking::Exit

	/// Apply color correction.
	g_fragColor.r = g_fragColor.r * in_scale + in_bias * g_fragColor.a;
	g_fragColor.g = g_fragColor.g * in_scale + in_bias * g_fragColor.a;
	g_fragColor.b = g_fragColor.b * in_scale + in_bias * g_fragColor.a;
	gl_FragData[0] = g_fragColor;

	//VTK::RenderToImage::Exit

	//VTK::DepthPass::Exit
}

//////////////////////////////////////////////////////////////////////////////
///
/// Main
///
//////////////////////////////////////////////////////////////////////////////
void main()
{
	//VTK::CallWorker::Impl
	if (initializeRayCast() == true) {
		castRay(-1.0, -1.0);
		finalizeRayCast();
	}
}