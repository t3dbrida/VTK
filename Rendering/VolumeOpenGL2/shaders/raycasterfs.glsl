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

#define FLOAT_MAX 1e20
#define FLOAT_EPS 1e-7

//////////////////////////////////////////////////////////////////////////////
///
/// Inputs
///
//////////////////////////////////////////////////////////////////////////////

/// 3D texture coordinates form vertex shader
in vec3 ip_textureCoords;
in vec3 ip_vertexPos;

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
vec3 g_rayDir;
float g_rayDirDot;
vec3 g_dataPos;
vec3 g_terminatePos;
vec3 g_dirStep;
vec4 g_srcColor;
vec4 g_eyePosObj;
bool g_exit;
bool g_skip;
float g_currentT;
float g_terminatePointMax;

//VTK::CustomUniforms::Dec

//VTK::Output::Dec

//VTK::Base::Dec

//VTK::Termination::Dec

//VTK::Cropping::Dec

//VTK::Clipping::Dec

//VTK::Shading::Dec

//VTK::BinaryMask::Dec

//VTK::RegionMask::Dec

//VTK::CompositeMask::Dec

//VTK::GradientCache::Dec

//VTK::Transfer2D::Dec

//VTK::ComputeOpacity::Dec

//VTK::ComputeGradient::Dec

//VTK::ComputeGradientOpacity1D::Dec

//VTK::ComputeLighting::Dec

//VTK::ComputeColor::Dec

//VTK::ComputeRayDirection::Dec

//VTK::Picking::Dec

//VTK::RenderToImage::Dec

//VTK::DepthPeeling::Dec

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
  vec4 NDCCoord = vec4(0.0, 0.0, 0.0, 1.0);

  NDCCoord.x = (xCoord - in_windowLowerLeftCorner.x) * 2.0 * in_inverseWindowSize.x - 1.0;
  NDCCoord.y = (yCoord - in_windowLowerLeftCorner.y) * 2.0 * in_inverseWindowSize.y - 1.0;
  NDCCoord.z = (2.0 * zCoord - (gl_DepthRange.near + gl_DepthRange.far)) / gl_DepthRange.diff;

  return NDCCoord;
}

/**
 * Transform NDC coordinate to window coordinates.
 */
vec4 NDCToWindow(const float xNDC, const float yNDC, const float zNDC)
{
  vec4 WinCoord = vec4(0.0, 0.0, 0.0, 1.0);

  WinCoord.x = (xNDC + 1.f) / (2.f * in_inverseWindowSize.x) + in_windowLowerLeftCorner.x;
  WinCoord.y = (yNDC + 1.f) / (2.f * in_inverseWindowSize.y) + in_windowLowerLeftCorner.y;
  WinCoord.z = (zNDC * gl_DepthRange.diff + (gl_DepthRange.near + gl_DepthRange.far)) / 2.f;

  return WinCoord;
}

vec2 intersectRayBox(vec3 rayOrigin, vec3 rayDir, vec3 aabbMin, vec3 aabbMax, mat4 transform)
{
  // This is actually correct, even though it appears not to handle edge cases
  // (rayDir.{x,y,z} == 0).  It works because the infinities that result from
  // dividing by zero will still behave correctly in the comparisons. Rays
  // which are parallel to an axis and outside the box will have tmin == inf
  // or tmax == -inf, while rays inside the box will have tmin and tmax
  // unchanged.

  rayOrigin = (transform * vec4(rayOrigin, 1.)).xyz;
  rayDir = (transform * vec4(rayDir, 0.)).xyz;

  vec3 rayDirInv = vec3(1.) / rayDir;

  float t1 = (aabbMin.x - rayOrigin.x) * rayDirInv.x,
        t2 = (aabbMax.x - rayOrigin.x) * rayDirInv.x,
        tMin = min(t1, t2),
        tMax = max(t1, t2);

  t1 = (aabbMin.y - rayOrigin.y) * rayDirInv.y;
  t2 = (aabbMax.y - rayOrigin.y) * rayDirInv.y;

  tMin = max(tMin, min(t1, t2));
  tMax = min(tMax, max(t1, t2));

  t1 = (aabbMin.z - rayOrigin.z) * rayDirInv.z;
  t2 = (aabbMax.z - rayOrigin.z) * rayDirInv.z;

  tMin = max(tMin, min(t1, t2));
  tMax = min(tMax, max(t1, t2));

  return vec2(tMin, tMax);
}

/*vec2 intersectRayBox(vec3 ray_origin, vec3 ray_direction, vec3 aabb_min, vec3 aabb_max, mat4 transform)
{
    // Intersection method from Real-Time Rendering and Essential Mathematics for Games
	float tMin = -FLOAT_MAX,
	      tMax = FLOAT_MAX;

	vec3 obbPos = vec3(transform[3].x, transform[3].y, transform[3].z),
		 delta = obbPos - ray_origin;

	// Test intersection with the 2 planes perpendicular to the OBB's X axis
	{
		vec3 xaxis = vec3(transform[0].x, transform[0].y, transform[0].z);
		float e = dot(xaxis, delta),
			  f = dot(ray_direction, xaxis);

		if (abs(f) > FLOAT_EPS)
		{
		    // Standard case
			float t1 = (e + aabb_min.x) / f, // Intersection with the "left" plane
			      t2 = (e + aabb_max.x) / f; // Intersection with the "right" plane
			// t1 and t2 now contain distances betwen ray origin and ray-plane intersections

			// We want t1 to represent the nearest intersection, 
			// so if it's not the case, invert t1 and t2
			if (t1 > t2)
			{
				float w = t1;
				t1 = t2;
				t2 = w; // swap t1 and t2
			}

			// tMax is the nearest "far" intersection (amongst the X,Y and Z planes pairs)
			if (t2 < tMax)
			{
				tMax = t2;
			}
			// tMin is the farthest "near" intersection (amongst the X,Y and Z planes pairs)
			if (t1 > tMin)
			{
				tMin = t1;
			}

			// If "far" is closer than "near", then there is NO intersection.
			if (tMax < tMin)
			{
				return vec2(0., 0.);
			}

		}
		else
		{
			// Rare case : the ray is almost parallel to the planes, so they don't have any "intersection"
			if (-e + aabb_min.x > 0. || -e + aabb_max.x < 0.)
			{
				return vec2(0., 0.);
			}
		}
	}


	// Test intersection with the 2 planes perpendicular to the OBB's Y axis
	// Exactly the same thing than above.
	{
		vec3 yaxis = vec3(transform[1].x, transform[1].y, transform[1].z);
		float e = dot(yaxis, delta),
			  f = dot(ray_direction, yaxis);

		if (abs(f) > FLOAT_EPS)
		{
			float t1 = (e + aabb_min.y) / f,
			      t2 = (e + aabb_max.y) / f;

			if (t1 > t2)
			{
				float w = t1;
				t1 = t2;
				t2 = w;
			}

			if (t2 < tMax)
			{
				tMax = t2;
			}
			if (t1 > tMin)
			{
				tMin = t1;
			}
			if (tMin > tMax)
			{
				return vec2(0., 0.);
			}
		}
		else
		{
			if (-e + aabb_min.y > 0. || -e + aabb_max.y < 0.)
			{
				return vec2(0., 0.);
			}
		}
	}

	// Test intersection with the 2 planes perpendicular to the OBB's Z axis
	// Exactly the same thing than above.
	{
		vec3 zaxis = vec3(transform[2].x, transform[2].y, transform[2].z);
		float e = dot(zaxis, delta),
			  f = dot(ray_direction, zaxis);

		if (abs(f) > FLOAT_EPS)
		{
			float t1 = (e + aabb_min.z) / f,
			      t2 = (e + aabb_max.z) / f;

			if (t1 > t2)
			{
				float w = t1;
				t1 = t2;
				t2 = w;
			}

			if (t2 < tMax)
			{
				tMax = t2;
			}
			if (t1 > tMin)
			{
				tMin = t1;
			}
			if (tMin > tMax)
			{
				return vec2(0., 0.);
			}
		}
		else
		{
			if (-e + aabb_min.z > 0. || -e + aabb_max.z < 0.)
			{
				return vec2(0., 0.);
			}
		}
	}

	return vec2(tMin, tMax);
}*/

/**
 * Clamps the texture coordinate vector @a pos to a new position in the set
 * { start + i * step }, where i is an integer. If @a ceiling
 * is true, the sample located further in the direction of @a step is used,
 * otherwise the sample location closer to the eye is used.
 */
vec3 ClampToSampleLocation(vec3 start, vec3 step, vec3 pos, bool ceiling)
{
  pos -= g_rayJitter[0];

  vec3 offset = pos - start;
  float stepLength = length(step);

  // Scalar projection of offset on step:
  float dist = dot(offset, step / stepLength);
  if (dist < 0.) // Don't move before the start position:
  {
    return start + g_rayJitter[0];
  }

  // Number of steps
  float steps = dist / stepLength;

  // If we're reeaaaaallly close, just round -- it's likely just numerical noise
  // and the value should be considered exact.
  if (abs(mod(steps, 1.)) > 1e-5)
  {
    if (ceiling)
    {
      steps = ceil(steps);
    }
    else
    {
      steps = floor(steps);
    }
  }
  else
  {
    steps = floor(steps + 0.5);
  }

  return start + steps * step + g_rayJitter[0];
}

//////////////////////////////////////////////////////////////////////////////
///
/// Ray-casting
///
//////////////////////////////////////////////////////////////////////////////

/**
 * Global initialization. This method should only be called once per shader
 * invocation regardless of whether castRay() is called several times (e.g.
 * vtkDualDepthPeelingPass). Any castRay() specific initialization should be
 * placed within that function.
 */
void initializeRayCast()
{
  /// Initialize g_fragColor (output) to 0
  g_fragColor = vec4(0.0);
  g_dirStep = vec3(0.0);
  g_srcColor = vec4(0.0);
  g_exit = false;

  //VTK::Base::Init

  //VTK::Terminate::Init

  //VTK::Cropping::Init

  //VTK::Clipping::Init

  //VTK::RenderToImage::Init

  //VTK::DepthPass::Init
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
  //VTK::DepthPeeling::Ray::Init

  //VTK::Clipping::Impl

  //VTK::DepthPeeling::Ray::PathCheck

  //VTK::Shading::Init

  /// For all samples along the ray
  while (!g_exit)
  {
    //VTK::Base::Impl

    //VTK::Cropping::Impl

    //VTK::BinaryMask::Impl

    //VTK::RegionMask::Impl

    //VTK::CompositeMask::Impl

    //VTK::PreComputeGradients::Impl

    //VTK::Shading::Impl

    //VTK::RenderToImage::Impl

    //VTK::DepthPass::Impl

    //VTK::Base::Advance

    //VTK::Terminate::Impl
  }

  //VTK::Shading::Exit

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
}
