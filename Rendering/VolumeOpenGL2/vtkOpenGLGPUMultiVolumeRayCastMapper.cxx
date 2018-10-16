/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkOpenGLGPUMultiVolumeRayCastMapper.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "vtkOpenGLGPUMultiVolumeRayCastMapper.h"

#include "vtkOpenGLMultiVolumeGradientOpacityTable.h"
#include "vtkOpenGLMultiVolumeOpacityTable.h"
#include "vtkOpenGLMultiVolumeRGBTable.h"
#include "vtkMultiVolumeShaderComposer.h"
#include "vtkVolumeStateRAII.h"

// Include compiled shader code
#include <multiraycasterfs.h>
#include <multiraycastervs.h>

// VTK includes
#include <vtkBoundingBox.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkClipConvexPolyData.h>
#include <vtkColorTransferFunction.h>
#include <vtkCommand.h>
#include <vtkContourFilter.h>
#include <vtkDataArray.h>
#include <vtkDensifyPolyData.h>
#include <vtkFloatArray.h>
#include <vtkOpenGLFramebufferObject.h>
#include <vtkImageData.h>
#include "vtkInformation.h"
#include <vtkLightCollection.h>
#include <vtkLight.h>
#include <vtkMath.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include "vtkOpenGLActor.h"
#include <vtkOpenGLError.h>
#include <vtkOpenGLBufferObject.h>
#include <vtkOpenGLCamera.h>
#include <vtkOpenGLRenderPass.h>
#include <vtkOpenGLRenderUtilities.h>
#include <vtkOpenGLRenderWindow.h>
#include "vtkOpenGLResourceFreeCallback.h"
#include <vtkOpenGLShaderCache.h>
#include <vtkOpenGLVertexArrayObject.h>
#include <vtkPerlinNoise.h>
#include <vtkPixelBufferObject.h>
#include <vtkPixelExtent.h>
#include <vtkPixelTransfer.h>
#include <vtkPlaneCollection.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkShader.h>
#include <vtkShaderProgram.h>
#include <vtkSmartPointer.h>
#include <vtkTessellatedBoxSource.h>
#include <vtkTextureObject.h>
#include <vtkTimerLog.h>
#include <vtkTransform.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkVolumeMask.h>
#include <vtkVolumeProperty.h>
#include <vtkMultiVolumeTexture.h>
#include <vtkWeakPointer.h>
#include <vtkHardwareSelector.h>
#include <vtkMultiVolume2.h>

#include <vtk_glew.h>

// C/C++ includes
#include <cassert>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <array>

vtkStandardNewMacro(vtkOpenGLGPUMultiVolumeRayCastMapper);

//----------------------------------------------------------------------------
class vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal
{
public:
  // Constructor
  //--------------------------------------------------------------------------
  vtkInternal(vtkOpenGLGPUMultiVolumeRayCastMapper* parent)
  {
    this->Parent = parent;
    this->LoadDepthTextureExtensionsSucceeded = false;
    this->NoiseTextureObject = NULL;
    this->DepthTextureObject = 0;
    //this->ActualSampleDistance = 1.0;
    this->Mask1RGBTable = 0;
    this->Mask2RGBTable =  0;
	this->GradientOpacityTables = {};
    this->CurrentMask = 0;
    this->WindowLowerLeft[0] = this->WindowLowerLeft[1] = 0;
    this->WindowSize[0] = this->WindowSize[1] = 0;
    this->LastDepthPassWindowSize[0] = this->LastDepthPassWindowSize[1] = 0;
    this->LastRenderToImageWindowSize[0] = 0;
    this->LastRenderToImageWindowSize[1] = 0;
    this->CurrentSelectionPass = vtkHardwareSelector::MIN_KNOWN_PASS - 1;

    this->NoiseTextureData = NULL;

    this->NumberOfLights = 0;
    this->LightComplexity = 0;

    this->MaskTextures = new vtkMapMaskTextureId;

    this->ShaderCache = 0;

    this->FBO = 0;
    this->RTTDepthBufferTextureObject = 0;
    this->RTTDepthTextureObject = 0;
    this->RTTColorTextureObject = 0;
    this->RTTDepthTextureType = -1;

    this->DPFBO = 0;
    this->DPDepthBufferTextureObject = 0;
    this->DPColorTextureObject = 0;
    this->PreserveViewport = false;
    this->PreserveGLState = false;
  }

  // Destructor
  //--------------------------------------------------------------------------
  ~vtkInternal()
  {
    delete [] this->NoiseTextureData;

    if (this->NoiseTextureObject)
    {
      this->NoiseTextureObject->Delete();
      this->NoiseTextureObject = NULL;
    }

    if (this->DepthTextureObject)
    {
      this->DepthTextureObject->Delete();
      this->DepthTextureObject = NULL;
    }

    if (this->FBO)
    {
      this->FBO->Delete();
      this->FBO = NULL;
    }

    if (this->RTTDepthBufferTextureObject)
    {
      this->RTTDepthBufferTextureObject->Delete();
      this->RTTDepthBufferTextureObject = NULL;
    }

    if (this->RTTDepthTextureObject)
    {
      this->RTTDepthTextureObject->Delete();
      this->RTTDepthTextureObject = NULL;
    }

    if (this->RTTColorTextureObject)
    {
      this->RTTColorTextureObject->Delete();
      this->RTTColorTextureObject = NULL;
    }

    if (this->ImageSampleFBO)
    {
      this->ImageSampleFBO->Delete();
      this->ImageSampleFBO = NULL;
    }

    for (auto& tex : this->ImageSampleTexture)
    {
      tex = nullptr;
    }
    this->ImageSampleTexture.clear();
    this->ImageSampleTexNames.clear();

    if (this->ImageSampleVBO)
    {
      this->ImageSampleVBO->Delete();
      this->ImageSampleVBO = NULL;
    }

    if (this->ImageSampleVAO)
    {
      this->ImageSampleVAO->Delete();
      this->ImageSampleVAO = NULL;
    }

	for (int i = 0; i < this->Parent->GetNumberOfInputConnections(0); ++i)
		this->DeleteTransferFunctions(i);

    delete this->MaskTextures;


    // Do not delete the shader programs - Let the cache clean them up.
    this->ImageSampleProg = NULL;
  }

  // Helper methods
  //--------------------------------------------------------------------------
  template<typename T>
  static void ToFloat(const T& in1, const T& in2, float (&out)[2]);
  template<typename T>
  static void ToFloat(const T& in1, const T& in2, const T& in3,
                      float (&out)[3]);
  template<typename T>
  static void ToFloat(T* in, float* out, int noOfComponents);
  template<typename T>
  static void ToFloat(T (&in)[3], float (&out)[3]);
  template<typename T>
  static void ToFloat(T (&in)[2], float (&out)[2]);
  template<typename T>
  static void ToFloat(T& in, float& out);
  template<typename T>
  static void ToFloat(T (&in)[4][2], float (&out)[4][2]);

  void Initialize(int i, vtkRenderer* ren, vtkMultiVolume2* vol,
                  int noOfComponents, int independentComponents);

  bool LoadMask(vtkRenderer* ren, vtkImageData* input,
                vtkImageData* maskInput, int textureExtent[6],
                vtkMultiVolume2* volume);

  bool LoadData(int i, vtkRenderer* ren, vtkMultiVolume2* vol, vtkVolumeProperty* volProp,
    vtkImageData* input, vtkDataArray* scalars);

  void DeleteTransferFunctions(int i);

  void ComputeBounds(int i, vtkImageData* input);

  // Update OpenGL volume information
  void UpdateVolume(int i, vtkVolumeProperty* volumeProperty);

  // Update transfer color function based on the incoming inputs
  // and number of scalar components.
  int UpdateColorTransferFunction(int in, vtkRenderer* ren,
                                  vtkMultiVolume2* vol,
                                  unsigned int component);

  // Update opacity transfer function (not gradient opacity)
  int UpdateOpacityTransferFunction(int in, vtkRenderer* ren,
                                    vtkMultiVolume2* vol,
                                    unsigned int component);

  // Update gradient opacity function
  int UpdateGradientOpacityTransferFunction(int in, vtkRenderer* ren,
                                            vtkMultiVolume2* vol,
                                            unsigned int component);

  // Update noise texture (used to reduce rendering artifacts
  // specifically banding effects)
  void CreateNoiseTexture(vtkRenderer* ren);

  // Update depth texture (used for early termination of the ray)
  void CaptureDepthTexture(vtkRenderer* ren, vtkMultiVolume2* vol);

  // Test if camera is inside the volume geometry
  bool IsCameraInside(int i, vtkRenderer* ren, vtkMultiVolume2* vol);

  // Compute transformation from cell texture-coordinates to point texture-coords
  // (CTP). Cell data maps correctly to OpenGL cells, point data does not (VTK
  // defines points at the cell corners). To set the point data in the center of the
  // OpenGL texels, a translation of 0.5 texels is applied, and the range is rescaled
  // to the point range.
  //
  // delta = TextureExtentsMax - TextureExtentsMin;
  // min   = vec3(0.5) / delta;
  // max   = (delta - vec3(0.5)) / delta;
  // range = max - min
  //
  // CTP = translation * Scale
  // CTP = range.x,        0,        0,  min.x
  //             0,  range.y,        0,  min.y
  //             0,        0,  range.z,  min.z
  //             0,        0,        0,    1.0
  void ComputeCellToPointMatrix(int i);

  // Update parameters for lighting that will be used in the shader.
  void SetLightingParameters(vtkRenderer* ren,
                             vtkShaderProgram* prog,
                             vtkMultiVolume2* vol);

  // Update the volume geometry
  void RenderVolumeGeometry(vtkRenderer* ren,
                            vtkShaderProgram* prog,
                            vtkMultiVolume2* vol);

  // Update cropping params to shader
  void SetCroppingRegions(vtkRenderer* ren, vtkShaderProgram* prog,
                          vtkMultiVolume2* vol);

  // Update clipping params to shader
  void SetClippingPlanes(vtkRenderer* ren, vtkShaderProgram* prog,
                         vtkMultiVolume2* vol);

  // Update the interval of sampling
  void UpdateSamplingDistance(int in, vtkImageData *input,
                              vtkRenderer* ren, vtkMultiVolume2* vol);

  // Check if the mapper should enter picking mode.
  void CheckPickingState(vtkRenderer* ren);

  // Look for property keys used to control the mapper's state.
  // This is necessary for some render passes which need to ensure
  // a specific OpenGL state when rendering through this mapper.
  void CheckPropertyKeys(vtkMultiVolume2* vol);

  // Configure the vtkHardwareSelector to begin a picking pass.
  void BeginPicking(vtkRenderer* ren);

  // Update the prop Id if hardware selection is enabled.
  void SetPickingId(vtkRenderer* ren);

  // Configure the vtkHardwareSelector to end a picking pass.
  void EndPicking(vtkRenderer* ren);

  // Load OpenGL extensiosn required to grab depth sampler buffer
  void LoadRequireDepthTextureExtensions(vtkRenderWindow* renWin);

  // Create GL buffers
  void CreateBufferObjects(const int i);

  // Dispose / free GL buffers
  void DeleteBufferObjects(const int i);

  // Convert vtkTextureObject to vtkImageData
  void ConvertTextureToImageData(vtkTextureObject* texture,
                                 vtkImageData* output);

  // Render to texture for final rendering
  void SetupRenderToTexture(vtkRenderer* ren);
  void ExitRenderToTexture(vtkRenderer* ren);

  // Render to texture for depth pass
  void SetupDepthPass(vtkRenderer* ren);
  void ExitDepthPass(vtkRenderer* ren);

  //@{
  /**
   * Image XY-Sampling
   * Render to an internal framebuffer with lower resolution than the currently
   * bound one (hence casting less rays and improving performance). The rendered
   * image is subsequently rendered as a texture-mapped quad (linearly
   * interpolated) to the default (or previously attached) framebuffer. If a
   * vtkOpenGLRenderPass is attached, a variable number of render targets are
   * supported (as specified by the RenderPass). The render targets are assumed
   * to be ordered from GL_COLOR_ATTACHMENT0 to GL_COLOR_ATTACHMENT$N$, where
   * $N$ is the number of targets specified (targets of the previously bound
   * framebuffer as activated through ActivateDrawBuffers(int)). Without a
   * RenderPass attached, it relies on FramebufferObject to re-activate the
   * appropriate previous DrawBuffer.
   *
   * \sa vtkOpenGLRenderPass vtkOpenGLFramebufferObject
   */
  void BeginImageSample(vtkRenderer* ren, vtkMultiVolume2* vol);
  bool InitializeImageSampleFBO(vtkRenderer* ren);
  void EndImageSample(vtkRenderer* ren);
  size_t GetNumImageSampleDrawBuffers(vtkMultiVolume2* vol);
  //@}

  void ReleaseRenderToTextureGraphicsResources(vtkWindow* win);
  void ReleaseImageSampleGraphicsResources(vtkWindow* win);
  void ReleaseDepthPassGraphicsResources(vtkWindow* win);

  // Private member variables
  //--------------------------------------------------------------------------
  vtkOpenGLGPUMultiVolumeRayCastMapper* Parent;

  bool LoadDepthTextureExtensionsSucceeded;
  std::map<int, bool> CameraWasInsideInLastUpdate;

  //GLuint CubeVBOId;
  //GLuint CubeVAOId;
  //GLuint CubeIndicesId;
  std::map<int, GLuint> CubeVBOId;
  std::map<int, GLuint> CubeVAOId;
  std::map<int, GLuint> CubeIndicesId;

  vtkTextureObject* NoiseTextureObject;
  vtkTextureObject* DepthTextureObject;

  float* NoiseTextureData;

  std::map<int, float> ActualSampleDistance;

  int LastProjectionParallel;
  int WindowLowerLeft[2];
  int WindowSize[2];
  int LastDepthPassWindowSize[2];
  int LastRenderToImageWindowSize[2];

  std::map<int, std::array<double, 6>> LoadedBounds;
  std::map<int, std::array<int, 6>> Extents;
  std::map<int, std::array<double, 3>> CellSpacing;

  int NumberOfLights;
  int LightComplexity;

  std::ostringstream ExtensionsStringStream;

  std::map<int, vtkOpenGLMultiVolumeRGBTables*> RGBTables;
  std::map<int, std::map<int, std::string>> RGBTablesMap;

  std::map<int, vtkOpenGLMultiVolumeOpacityTables*> OpacityTables;
  std::map<int, std::map<int, std::string>> OpacityTablesMap;

  std::map<int, vtkOpenGLMultiVolumeGradientOpacityTables*> GradientOpacityTables;
  std::map<int, std::map<int, std::string>> GradientOpacityTablesMap;

  vtkOpenGLMultiVolumeRGBTable* Mask1RGBTable;
  vtkOpenGLMultiVolumeRGBTable* Mask2RGBTable;

  vtkTimeStamp ShaderBuildTime;

  std::map<int, vtkSmartPointer<vtkMatrix4x4>> CellToPointMatrix;
  std::map<int, std::array<float, 4>> AdjustedTexMin;
  std::map<int, std::array<float, 4>> AdjustedTexMax;

  vtkSmartPointer<vtkMatrix4x4> GetCellToPointMatrix(int i)
  {
	  if (!this->CellToPointMatrix[i]) {
		  this->CellToPointMatrix[i] = vtkSmartPointer<vtkMatrix4x4>::New();
		  this->CellToPointMatrix[i]->Identity();
	  }

	  return this->CellToPointMatrix[i];
  }

  vtkSmartPointer<vtkPolyData> BBoxPolyData;

  vtkMapMaskTextureId* MaskTextures;
  vtkVolumeMask* CurrentMask;

  std::map<int, vtkTimeStamp> InitializationTime;
  std::map<int, vtkTimeStamp> InputUpdateTime;
  std::map<int, vtkTimeStamp> VolumeUpdateTime;
  vtkTimeStamp MaskUpdateTime;
  vtkTimeStamp ReleaseResourcesTime;
  vtkTimeStamp DepthPassTime;
  vtkTimeStamp DepthPassSetupTime;
  vtkTimeStamp SelectionStateTime;
  int CurrentSelectionPass;
  bool IsPicking;

  std::map<int, bool> NeedToInitializeResources;
  bool PreserveViewport;
  bool PreserveGLState;

  vtkShaderProgram* ShaderProgram;
  vtkOpenGLShaderCache* ShaderCache;

  vtkOpenGLFramebufferObject* FBO;
  vtkTextureObject* RTTDepthBufferTextureObject;
  vtkTextureObject* RTTDepthTextureObject;
  vtkTextureObject* RTTColorTextureObject;
  int RTTDepthTextureType;

  vtkOpenGLFramebufferObject* DPFBO;
  vtkTextureObject* DPDepthBufferTextureObject;
  vtkTextureObject* DPColorTextureObject;

  vtkOpenGLFramebufferObject* ImageSampleFBO = nullptr;
  std::vector<vtkSmartPointer<vtkTextureObject>> ImageSampleTexture;
  std::vector<std::string> ImageSampleTexNames;
  vtkShaderProgram* ImageSampleProg = nullptr;
  vtkOpenGLVertexArrayObject* ImageSampleVAO = nullptr;
  vtkOpenGLBufferObject* ImageSampleVBO = nullptr;
  size_t NumImageSampleDrawBuffers = 0;
  bool RebuildImageSampleProg = false;
  bool RenderPassAttached = false;

  vtkNew<vtkContourFilter>  ContourFilter;
  vtkNew<vtkPolyDataMapper> ContourMapper;
  vtkNew<vtkActor> ContourActor;
};

//----------------------------------------------------------------------------
template<typename T>
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::ToFloat(
  const T& in1, const T& in2, float (&out)[2])
{
  out[0] = static_cast<float>(in1);
  out[1] = static_cast<float>(in2);
}

template<typename T>
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::ToFloat(
  const T& in1, const T& in2, const T& in3, float (&out)[3])
{
  out[0] = static_cast<float>(in1);
  out[1] = static_cast<float>(in2);
  out[2] = static_cast<float>(in3);
}

//----------------------------------------------------------------------------
template<typename T>
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::ToFloat(
  T* in, float* out, int noOfComponents)
{
  for (int i = 0; i < noOfComponents; ++i)
  {
    out[i] = static_cast<float>(in[i]);
  }
}

//----------------------------------------------------------------------------
template<typename T>
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::ToFloat(
  T (&in)[3], float (&out)[3])
{
  out[0] = static_cast<float>(in[0]);
  out[1] = static_cast<float>(in[1]);
  out[2] = static_cast<float>(in[2]);
}

//----------------------------------------------------------------------------
template<typename T>
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::ToFloat(
  T (&in)[2], float (&out)[2])
{
  out[0] = static_cast<float>(in[0]);
  out[1] = static_cast<float>(in[1]);
}

//----------------------------------------------------------------------------
template<typename T>
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::ToFloat(
  T& in, float& out)
{
  out = static_cast<float>(in);
}

//----------------------------------------------------------------------------
template<typename T>
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::ToFloat(
  T (&in)[4][2], float (&out)[4][2])
{
  out[0][0] = static_cast<float>(in[0][0]);
  out[0][1] = static_cast<float>(in[0][1]);
  out[1][0] = static_cast<float>(in[1][0]);
  out[1][1] = static_cast<float>(in[1][1]);
  out[2][0] = static_cast<float>(in[2][0]);
  out[2][1] = static_cast<float>(in[2][1]);
  out[3][0] = static_cast<float>(in[3][0]);
  out[3][1] = static_cast<float>(in[3][1]);
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::Initialize(int in,
  vtkRenderer* vtkNotUsed(ren), vtkMultiVolume2* vol, int
  noOfComponents, int independentComponents)
{
  this->DeleteTransferFunctions(in);

  // Create RGB lookup table
  if (noOfComponents > 1 && independentComponents)
  {
		this->RGBTables[in] = new vtkOpenGLMultiVolumeRGBTables(noOfComponents);
  }
  else
  {
		this->RGBTables[in] = new vtkOpenGLMultiVolumeRGBTables(1);
  }

  if (this->Parent->MaskInput != 0 &&
      this->Parent->MaskType == LabelMapMaskType)
  {
    if(this->Mask1RGBTable == NULL)
    {
      this->Mask1RGBTable = vtkOpenGLMultiVolumeRGBTable::New();
    }
    if(this->Mask2RGBTable == NULL)
    {
      this->Mask2RGBTable = vtkOpenGLMultiVolumeRGBTable::New();
    }
  }

  // We support up to four components
  if (noOfComponents > 1 && independentComponents)
  {
		this->OpacityTables[in] = new vtkOpenGLMultiVolumeOpacityTables(noOfComponents);
  }
  else
  {
		this->OpacityTables[in] = new vtkOpenGLMultiVolumeOpacityTables(1);
  }

  if (noOfComponents > 1 && independentComponents)
  {
    // Assuming that all four components has gradient opacity for now
		  this->GradientOpacityTables[in] = new vtkOpenGLMultiVolumeGradientOpacityTables(noOfComponents);
  }
  else
  {
    if (vol->GetProperty(in)->HasGradientOpacity())
    {
		this->GradientOpacityTables[in] = new vtkOpenGLMultiVolumeGradientOpacityTables(1);
    }
  }

  std::ostringstream numeric;
  for (int i = 0; i < noOfComponents; ++i)
  {
	  numeric << i;
	  this->OpacityTablesMap[in][i] = std::string("in_opacityTransferFunc_") +
		  numeric.str() + "[" + std::to_string(in) + "]";
	  this->RGBTablesMap[in][i] = std::string("in_colorTransferFunc_") +
		  numeric.str() + "[" + std::to_string(in) + "]";
	  if (vol->GetProperty(in)->HasGradientOpacity(i))
		  this->GradientOpacityTablesMap[in][i] = std::string("in_gradientTransferFunc_") +
			  numeric.str() + "[" + std::to_string(in) + "]";
	  numeric.str("");
	  numeric.clear();
  }

  this->InitializationTime[in].Modified();
}

//-----------------------------------------------------------------------------
bool vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::LoadMask(vtkRenderer* ren,
  vtkImageData* vtkNotUsed(input), vtkImageData* maskInput,
  int textureExtent[6], vtkMultiVolume2* vtkNotUsed(volume))
{
  bool result = true;
  if (maskInput &&
    (maskInput->GetMTime() > this->MaskUpdateTime))
  {
    // Find the texture.
    std::map<vtkImageData *,vtkVolumeMask*>::iterator it2 =
      this->MaskTextures->Map.find(maskInput);

    vtkVolumeMask* mask = 0;
    if(it2 == this->MaskTextures->Map.end())
    {
      mask = new vtkVolumeMask();
      this->MaskTextures->Map[maskInput] = mask;
    }
    else
    {
      mask = (*it2).second;
    }

    mask->Update(ren,
                 maskInput,
                 this->Parent->CellFlag[0],
                 textureExtent,
                 this->Parent->ScalarMode,
                 this->Parent->ArrayAccessMode,
                 this->Parent->ArrayId,
                 this->Parent->ArrayName,
                 static_cast<vtkIdType>(static_cast<float>(
                 this->Parent->MaxMemoryInBytes) *
                 this->Parent->MaxMemoryFraction));

    result = result && mask->IsLoaded();
    this->CurrentMask = mask;
    this->MaskUpdateTime.Modified();
  }

  return result;
}

//----------------------------------------------------------------------------
bool vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::LoadData(int i, vtkRenderer* ren,
  vtkMultiVolume2* vol, vtkVolumeProperty* volProp, vtkImageData* input,
  vtkDataArray* scalars)
{
    // Update bounds, data, and geometry
    bool success = this->Parent->GetVolumeTexture(i)->LoadVolume(ren, input, scalars,
      volProp->GetInterpolationType());

	this->ComputeBounds(i, input);
	this->ComputeCellToPointMatrix(i);
	//this->LoadMask(ren, input, this->Parent->MaskInput, this->Extents, vol);
    this->InputUpdateTime[i].Modified();

    return success;
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::DeleteTransferFunctions(int in)
{
	if (this->RGBTables[in]) {
		delete this->RGBTables[in];
		this->RGBTables[in] = NULL;
	}

  if (this->Mask1RGBTable)
  {
    this->Mask1RGBTable->Delete();
    this->Mask1RGBTable = NULL;
  }

  if (this->Mask2RGBTable)
  {
    this->Mask2RGBTable->Delete();
    this->Mask2RGBTable = NULL;
  }

  if (this->OpacityTables[in]) {
	  delete this->OpacityTables[in];
	  this->OpacityTables[in] = NULL;
  }

  if (this->GradientOpacityTables[in]) {
	  delete this->GradientOpacityTables[in];
	  this->GradientOpacityTables[in] = NULL;
  }
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::ComputeBounds(int in,
  vtkImageData* input)
{
  double origin[3];

  input->GetSpacing(this->CellSpacing[in].data());
  input->GetOrigin(origin);
  this->Extents[in][0] = VTK_INT_MAX;
  this->Extents[in][1] = VTK_INT_MIN;
  this->Extents[in][2] = VTK_INT_MAX;
  this->Extents[in][3] = VTK_INT_MIN;
  this->Extents[in][4] = VTK_INT_MAX;
  this->Extents[in][5] = VTK_INT_MIN;
  input->GetExtent(this->Extents[in].data());

  int swapBounds[3];
  swapBounds[0] = (this->CellSpacing[in][0] < 0);
  swapBounds[1] = (this->CellSpacing[in][1] < 0);
  swapBounds[2] = (this->CellSpacing[in][2] < 0);

  // Loaded data represents points
  if (!this->Parent->CellFlag[in])
  {
    // If spacing is negative, we may have to rethink the equation
    // between real point and texture coordinate...
    this->LoadedBounds[in][0] = origin[0] +
      static_cast<double>(this->Extents[in][0 + swapBounds[0]]) *
      this->CellSpacing[in][0];
    this->LoadedBounds[in][2] = origin[1] +
      static_cast<double>(this->Extents[in][2 + swapBounds[1]]) *
      this->CellSpacing[in][1];
    this->LoadedBounds[in][4] = origin[2] +
      static_cast<double>(this->Extents[in][4 + swapBounds[2]]) *
      this->CellSpacing[in][2];
    this->LoadedBounds[in][1] = origin[0] +
      static_cast<double>(this->Extents[in][1 - swapBounds[0]]) *
      this->CellSpacing[in][0];
    this->LoadedBounds[in][3] = origin[1] +
      static_cast<double>(this->Extents[in][3 - swapBounds[1]]) *
      this->CellSpacing[in][1];
    this->LoadedBounds[in][5] = origin[2] +
      static_cast<double>(this->Extents[in][5 - swapBounds[2]]) *
      this->CellSpacing[in][2];
  }
  // Loaded extents represent cells
  else
  {
    int i = 0;
    while (i < 3)
    {
      this->LoadedBounds[in][2 * i + swapBounds[i]] = origin[i] +
        (static_cast<double>(this->Extents[in][2 * i])) *
        this->CellSpacing[in][i];

      this->LoadedBounds[in][2 * i + 1 - swapBounds[i]] = origin[i] +
        (static_cast<double>(this->Extents[in][2 * i + 1]) + 1.0) *
        this->CellSpacing[in][i];

      i++;
    }
  }
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::
     UpdateVolume(int i, vtkVolumeProperty* volumeProperty)
{
  if (volumeProperty->GetMTime() > this->VolumeUpdateTime[i].GetMTime())
  {
    int const newInterp = volumeProperty->GetInterpolationType();
	this->Parent->GetVolumeTexture(i)->UpdateInterpolationType(newInterp);
  }

  this->VolumeUpdateTime[i].Modified();
}

//----------------------------------------------------------------------------
int vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::
    UpdateColorTransferFunction(int in, vtkRenderer* ren, vtkMultiVolume2* vol,
                                unsigned int component)
{
    // Volume property cannot be null.
	vtkVolumeProperty* volumeProperty = vol->GetProperty(in);

	// Build the colormap in a 1D texture.
	// 1D RGB-texture=mapping from scalar values to color values
	// build the table.
	vtkColorTransferFunction* colorTransferFunction =
		volumeProperty->GetRGBTransferFunction(component);

	double componentRange[2];
	for (int i = 0; i < 2; ++i)
	{
		componentRange[i] = this->Parent->GetVolumeTexture(in)->ScalarRange[component][i];
	}

	// Add points only if its not being added before
	if (colorTransferFunction->GetSize() < 1)
	{
		colorTransferFunction->AddRGBPoint(componentRange[0], 0.0, 0.0, 0.0);
		colorTransferFunction->AddRGBPoint(componentRange[1], 1.0, 1.0, 1.0);
	}

	int filterVal =
		volumeProperty->GetInterpolationType() == VTK_LINEAR_INTERPOLATION ?
		vtkTextureObject::Linear : vtkTextureObject::Nearest;

		if (this->RGBTables[in])
			this->RGBTables[in]->GetTable(component)->Update(
				volumeProperty->GetRGBTransferFunction(component),
				componentRange,
#if GL_ES_VERSION_3_0 != 1
				filterVal,
#else
				vtkTextureObject::Nearest,
#endif
				vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow()));
	//}

	if (this->Parent->MaskInput != 0 &&
		this->Parent->MaskType == LabelMapMaskType)
	{
		vtkColorTransferFunction* colorTransferFunc =
			volumeProperty->GetRGBTransferFunction(1);
		this->Mask1RGBTable->Update(colorTransferFunc, componentRange,
			vtkTextureObject::Nearest,
			vtkOpenGLRenderWindow::SafeDownCast(
				ren->GetRenderWindow()));

		colorTransferFunc = volumeProperty->GetRGBTransferFunction(2);
		this->Mask2RGBTable->Update(colorTransferFunc, componentRange,
			vtkTextureObject::Nearest,
			vtkOpenGLRenderWindow::SafeDownCast(
				ren->GetRenderWindow()));
	}

  return 0;
}

//----------------------------------------------------------------------------
int vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::
  UpdateOpacityTransferFunction(int in, vtkRenderer* ren, vtkMultiVolume2* vol,
                                unsigned int component)
{
  if (!vol)
  {
    return 1;
  }

  vtkVolumeProperty* volumeProperty = vol->GetProperty(in);

  // Transfer function table index based on whether independent / dependent
  // components. If dependent, use the first scalar opacity transfer function
  unsigned int lookupTableIndex = volumeProperty->GetIndependentComponents() ?
	  component : 0;
  vtkPiecewiseFunction* scalarOpacity =
	  volumeProperty->GetScalarOpacity(lookupTableIndex);

  double componentRange[2];
  for (int i = 0; i < 2; ++i)
  {
	  componentRange[i] = this->Parent->GetVolumeTexture(in)->ScalarRange[component][i];
  }

  if (scalarOpacity->GetSize() < 1)
  {
	  scalarOpacity->AddPoint(componentRange[0], 0.0);
	  scalarOpacity->AddPoint(componentRange[1], 0.5);
  }

  int filterVal =
	  volumeProperty->GetInterpolationType() == VTK_LINEAR_INTERPOLATION ?
	  vtkTextureObject::Linear : vtkTextureObject::Nearest;

  if (!this->OpacityTables[in])
	  return 1;
  this->OpacityTables[in]->GetTable(lookupTableIndex)->Update(
	  scalarOpacity, this->Parent->BlendMode,
	  this->ActualSampleDistance[in],
	  componentRange,
	  volumeProperty->GetScalarOpacityUnitDistance(component),
#if GL_ES_VERSION_3_0 != 1
	  filterVal,
#else
	  vtkTextureObject::Nearest,
#endif
	  vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow()));

  return 0;
}

//----------------------------------------------------------------------------
int vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::
  UpdateGradientOpacityTransferFunction(int in, vtkRenderer* ren, vtkMultiVolume2* vol,
                                        unsigned int component)
{
  if (!vol)
  {
    return 1;
  }

  vtkVolumeProperty* volumeProperty = vol->GetProperty(in);

  // Transfer function table index based on whether independent / dependent
  // components. If dependent, use the first gradient opacity transfer function
  unsigned int lookupTableIndex = volumeProperty->GetIndependentComponents() ?
	  component : 0;
  // TODO Currently we expect the all of the tables will
  // be initialized once and if at that time, the gradient
  // opacity was not enabled then it is not used later.
  if (!volumeProperty->HasGradientOpacity(lookupTableIndex) ||
	  !this->GradientOpacityTables[in])
  {
	  return 1;
  }

  vtkPiecewiseFunction* gradientOpacity =
	  volumeProperty->GetGradientOpacity(lookupTableIndex);

  double componentRange[2];
  for (int i = 0; i < 2; ++i)
  {
	  componentRange[i] = this->Parent->GetVolumeTexture(in)->ScalarRange[component][i];
  }

  if (gradientOpacity->GetSize() < 1)
  {
	  gradientOpacity->AddPoint(componentRange[0], 0.0);
	  gradientOpacity->AddPoint(componentRange[1], 0.5);
  }

  int filterVal =
	  volumeProperty->GetInterpolationType() == VTK_LINEAR_INTERPOLATION ?
	  vtkTextureObject::Linear : vtkTextureObject::Nearest;

  if (!this->GradientOpacityTables[in])
	  return 1;
  this->GradientOpacityTables[in]->GetTable(lookupTableIndex)->Update(
	  gradientOpacity,
	  this->ActualSampleDistance[in],
	  componentRange,
	  volumeProperty->GetScalarOpacityUnitDistance(component),
#if GL_ES_VERSION_3_0 != 1
	  filterVal,
#else
	  vtkTextureObject::Nearest,
#endif
	  vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow()));

  return 0;
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::CreateNoiseTexture(
  vtkRenderer* ren)
{
  vtkOpenGLRenderWindow* glWindow = vtkOpenGLRenderWindow::SafeDownCast(
    ren->GetRenderWindow());

  if (!this->NoiseTextureObject)
  {
    this->NoiseTextureObject = vtkTextureObject::New();
  }
  this->NoiseTextureObject->SetContext(glWindow);

  bool updateSize = false;
  bool useUserSize = this->Parent->NoiseTextureSize[0] > 0 &&
    this->Parent->NoiseTextureSize[1] > 0;
  if (useUserSize)
  {
    int const twidth = this->NoiseTextureObject->GetWidth();
    int const theight = this->NoiseTextureObject->GetHeight();
    updateSize = this->Parent->NoiseTextureSize[0] != twidth ||
      this->Parent->NoiseTextureSize[1] != theight;
  }

  if (!this->NoiseTextureObject->GetHandle() || updateSize ||
    this->NoiseTextureObject->GetMTime() < this->Parent->NoiseGenerator->GetMTime())
  {
    int* winSize = ren->GetRenderWindow()->GetSize();
    int sizeX = useUserSize ? this->Parent->NoiseTextureSize[0] : winSize[0];
    int sizeY = useUserSize ? this->Parent->NoiseTextureSize[1] : winSize[1];

    int const maxSize = vtkTextureObject::GetMaximumTextureSize(glWindow);
    if (sizeX > maxSize || sizeY > maxSize)
    {
      sizeX = vtkMath::Max(sizeX, maxSize);
      sizeY = vtkMath::Max(sizeY, maxSize);
    }

    // Allocate buffer. After controlling for the maximum supported size sizeX/Y
    // might have changed, so an additional check is needed.
    int const twidth = this->NoiseTextureObject->GetWidth();
    int const theight = this->NoiseTextureObject->GetHeight();
    bool sizeChanged = sizeX != twidth || sizeY != theight;
    if (sizeChanged || !this->NoiseTextureData)
    {
      delete[] this->NoiseTextureData;
      this->NoiseTextureData = NULL;
      this->NoiseTextureData = new float[sizeX * sizeY];
    }

    // Generate jitter noise
    if (!this->Parent->NoiseGenerator)
    {
      // Use default settings
      vtkPerlinNoise* perlinNoise = vtkPerlinNoise::New();
      perlinNoise->SetPhase(0.0, 0.0, 0.0);
      perlinNoise->SetFrequency(sizeX, sizeY, 1.0);
      perlinNoise->SetAmplitude(0.5); /* [-n, n] */
      this->Parent->NoiseGenerator = perlinNoise;
    }

    int const bufferSize = sizeX * sizeY;
    for (int i = 0; i < bufferSize; i++)
    {
      int const x = i % sizeX;
      int const y = i / sizeY;
      this->NoiseTextureData[i] = static_cast<float>(
        this->Parent->NoiseGenerator->EvaluateFunction(x, y, 0.0) + 0.1);
    }

    // Prepare texture
    this->NoiseTextureObject->Create2DFromRaw(sizeX, sizeY, 1, VTK_FLOAT,
      this->NoiseTextureData);

    this->NoiseTextureObject->SetWrapS(vtkTextureObject::Repeat);
    this->NoiseTextureObject->SetWrapT(vtkTextureObject::Repeat);
    this->NoiseTextureObject->SetMagnificationFilter(vtkTextureObject::Nearest);
    this->NoiseTextureObject->SetMinificationFilter(vtkTextureObject::Nearest);
    this->NoiseTextureObject->SetBorderColor(0.0f, 0.0f, 0.0f, 0.0f);
    this->NoiseTextureObject->Modified();
  }
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::CaptureDepthTexture(
  vtkRenderer* ren, vtkMultiVolume2* vtkNotUsed(vol))
{
  // Make sure our render window is the current OpenGL context
  ren->GetRenderWindow()->MakeCurrent();

  // Load required extensions for grabbing depth sampler buffer
  if (!this->LoadDepthTextureExtensionsSucceeded)
  {
    this->LoadRequireDepthTextureExtensions(ren->GetRenderWindow());
  }

  // If we can't load the necessary extensions, provide
  // feedback on why it failed.
  if(!this->LoadDepthTextureExtensionsSucceeded)
  {
    std::cerr << this->ExtensionsStringStream.str() << std::endl;
    return;
  }

  if (!this->DepthTextureObject)
  {
    this->DepthTextureObject = vtkTextureObject::New();
  }

  this->DepthTextureObject->SetContext(vtkOpenGLRenderWindow::SafeDownCast(
                                        ren->GetRenderWindow()));
  if (!this->DepthTextureObject->GetHandle())
  {
    // First set the parameters
    this->DepthTextureObject->SetWrapS(vtkTextureObject::ClampToEdge);
    this->DepthTextureObject->SetWrapT(vtkTextureObject::ClampToEdge);
    this->DepthTextureObject->SetMagnificationFilter(vtkTextureObject::Linear);
    this->DepthTextureObject->SetMinificationFilter(vtkTextureObject::Linear);
    this->DepthTextureObject->AllocateDepth(this->WindowSize[0],
                                            this->WindowSize[1],
                                            4);
  }

#if GL_ES_VERSION_3_0 != 1
  // currently broken on ES
  this->DepthTextureObject->CopyFromFrameBuffer(this->WindowLowerLeft[0],
                                                this->WindowLowerLeft[1],
                                                0, 0,
                                                this->WindowSize[0],
                                                this->WindowSize[1]);
#endif
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::SetLightingParameters(
  vtkRenderer* ren, vtkShaderProgram* prog, vtkMultiVolume2* vol)
{
  if (!ren || !prog || !vol)
  {
    return;
  }

  if (vol && !vol->GetProperty(0)->GetShade())
  {
    return;
  }

  prog->SetUniformi("in_twoSidedLighting", ren->GetTwoSidedLighting());

  // for lightkit case there are some parameters to set
  vtkCamera* cam = ren->GetActiveCamera();
  vtkTransform* viewTF = cam->GetModelViewTransformObject();

  // Bind some light settings
  int numberOfLights = 0;
  vtkLightCollection *lc = ren->GetLights();
  vtkLight *light;

  vtkCollectionSimpleIterator sit;
  float lightAmbientColor[6][3];
  float lightDiffuseColor[6][3];
  float lightSpecularColor[6][3];
  float lightDirection[6][3];
  for(lc->InitTraversal(sit);
      (light = lc->GetNextLight(sit)); )
  {
    float status = light->GetSwitch();
    if (status > 0.0)
    {
      double* aColor = light->GetAmbientColor();
      double* dColor = light->GetDiffuseColor();
      double* sColor = light->GetDiffuseColor();
      double intensity = light->GetIntensity();
      lightAmbientColor[numberOfLights][0] = aColor[0] * intensity;
      lightAmbientColor[numberOfLights][1] = aColor[1] * intensity;
      lightAmbientColor[numberOfLights][2] = aColor[2] * intensity;
      lightDiffuseColor[numberOfLights][0] = dColor[0] * intensity;
      lightDiffuseColor[numberOfLights][1] = dColor[1] * intensity;
      lightDiffuseColor[numberOfLights][2] = dColor[2] * intensity;
      lightSpecularColor[numberOfLights][0] = sColor[0] * intensity;
      lightSpecularColor[numberOfLights][1] = sColor[1] * intensity;
      lightSpecularColor[numberOfLights][2] = sColor[2] * intensity;
      // Get required info from light
      double* lfp = light->GetTransformedFocalPoint();
      double* lp = light->GetTransformedPosition();
      double lightDir[3];
      vtkMath::Subtract(lfp, lp, lightDir);
      vtkMath::Normalize(lightDir);
      double *tDir = viewTF->TransformNormal(lightDir);
      lightDirection[numberOfLights][0] = tDir[0];
      lightDirection[numberOfLights][1] = tDir[1];
      lightDirection[numberOfLights][2] = tDir[2];
      numberOfLights++;
    }
  }

  prog->SetUniform3fv("in_lightAmbientColor",
                      numberOfLights, lightAmbientColor);
  prog->SetUniform3fv("in_lightDiffuseColor",
                      numberOfLights, lightDiffuseColor);
  prog->SetUniform3fv("in_lightSpecularColor",
                      numberOfLights, lightSpecularColor);
  prog->SetUniform3fv("in_lightDirection",
                      numberOfLights, lightDirection);
  prog->SetUniformi("in_numberOfLights",
                    numberOfLights);

  // we are done unless we have positional lights
  if (this->LightComplexity < 3)
  {
    return;
  }

  // if positional lights pass down more parameters
  float lightAttenuation[6][3];
  float lightPosition[6][3];
  float lightConeAngle[6];
  float lightExponent[6];
  int lightPositional[6];
  numberOfLights = 0;
  for(lc->InitTraversal(sit);
      (light = lc->GetNextLight(sit)); )
  {
    float status = light->GetSwitch();
    if (status > 0.0)
    {
      double* attn = light->GetAttenuationValues();
      lightAttenuation[numberOfLights][0] = attn[0];
      lightAttenuation[numberOfLights][1] = attn[1];
      lightAttenuation[numberOfLights][2] = attn[2];
      lightExponent[numberOfLights] = light->GetExponent();
      lightConeAngle[numberOfLights] = light->GetConeAngle();
      double* lp = light->GetTransformedPosition();
      double* tlp = viewTF->TransformPoint(lp);
      lightPosition[numberOfLights][0] = tlp[0];
      lightPosition[numberOfLights][1] = tlp[1];
      lightPosition[numberOfLights][2] = tlp[2];
      lightPositional[numberOfLights] = light->GetPositional();
      numberOfLights++;
    }
  }
  prog->SetUniform3fv("in_lightAttenuation", numberOfLights, lightAttenuation);
  prog->SetUniform1iv("in_lightPositional", numberOfLights, lightPositional);
  prog->SetUniform3fv("in_lightPosition", numberOfLights, lightPosition);
  prog->SetUniform1fv("in_lightExponent", numberOfLights, lightExponent);
  prog->SetUniform1fv("in_lightConeAngle", numberOfLights, lightConeAngle);
}

//-----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::ComputeCellToPointMatrix(int i)
{
  this->GetCellToPointMatrix(i);
  this->AdjustedTexMin[i][0] = this->AdjustedTexMin[i][1] = this->AdjustedTexMin[i][2] = 0.0f;
  this->AdjustedTexMin[i][3] = 1.0f;
  this->AdjustedTexMax[i][0] = this->AdjustedTexMax[i][1] = this->AdjustedTexMax[i][2] = 1.0f;
  this->AdjustedTexMax[i][3] = 1.0f;

  if (!this->Parent->CellFlag[i]) // point data
  {
    float delta[3];
    delta[0] = this->Extents[i][1] - this->Extents[i][0];
    delta[1] = this->Extents[i][3] - this->Extents[i][2];
    delta[2] = this->Extents[i][5] - this->Extents[i][4];

    float min[3];
    min[0] = 0.5f / delta[0];
    min[1] = 0.5f / delta[1];
    min[2] = 0.5f / delta[2];

    float range[3]; // max - min
    range[0] = (delta[0] - 0.5f) / delta[0] - min[0];
    range[1] = (delta[1] - 0.5f) / delta[1] - min[1];
    range[2] = (delta[2] - 0.5f) / delta[2] - min[2];

    this->CellToPointMatrix[i]->SetElement(0, 0, range[0]); // Scale diag
    this->CellToPointMatrix[i]->SetElement(1, 1, range[1]);
    this->CellToPointMatrix[i]->SetElement(2, 2, range[2]);
    this->CellToPointMatrix[i]->SetElement(0, 3, min[0]);   // t vector
    this->CellToPointMatrix[i]->SetElement(1, 3, min[1]);
    this->CellToPointMatrix[i]->SetElement(2, 3, min[2]);

    // Adjust limit coordinates for texture access.
    float const zeros[4] = {0.0f, 0.0f, 0.0f, 1.0f}; // GL tex min
    float const ones[4]  = {1.0f, 1.0f, 1.0f, 1.0f}; // GL tex max
    this->CellToPointMatrix[i]->MultiplyPoint(zeros, this->AdjustedTexMin[i].data());
    this->CellToPointMatrix[i]->MultiplyPoint(ones, this->AdjustedTexMax[i].data());
  }
}

//----------------------------------------------------------------------------
bool vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::IsCameraInside(int i,
  vtkRenderer* ren, vtkMultiVolume2* vol)
{
  vtkNew<vtkMatrix4x4> tempMatrix;
  tempMatrix->DeepCopy(vol->GetVolumeMatrix(i));
  tempMatrix->Invert();

  vtkCamera* cam = ren->GetActiveCamera();
  double camWorldRange[2];
  double camWorldPos[4];
  double camFocalWorldPoint[4];
  double camWorldDirection[4];
  double camPos[4];
  double camPlaneNormal[4];

  cam->GetPosition(camWorldPos);
  camWorldPos[3] = 1.0;
  tempMatrix->MultiplyPoint( camWorldPos, camPos );

  cam->GetFocalPoint(camFocalWorldPoint);
  camFocalWorldPoint[3] = 1.0;

  // The range (near/far) must also be transformed
  // into the local coordinate system.
  camWorldDirection[0] = camFocalWorldPoint[0] - camWorldPos[0];
  camWorldDirection[1] = camFocalWorldPoint[1] - camWorldPos[1];
  camWorldDirection[2] = camFocalWorldPoint[2] - camWorldPos[2];
  camWorldDirection[3] = 0.0;

  // Compute the normalized near plane normal
  tempMatrix->MultiplyPoint(camWorldDirection, camPlaneNormal);

  vtkMath::Normalize(camWorldDirection);
  vtkMath::Normalize(camPlaneNormal);

  double camNearWorldPoint[4];
  double camNearPoint[4];

  cam->GetClippingRange(camWorldRange);
  camNearWorldPoint[0] = camWorldPos[0] + camWorldRange[0] * camWorldDirection[0];
  camNearWorldPoint[1] = camWorldPos[1] + camWorldRange[0] * camWorldDirection[1];
  camNearWorldPoint[2] = camWorldPos[2] + camWorldRange[0] * camWorldDirection[2];
  camNearWorldPoint[3] = 1.;

  tempMatrix->MultiplyPoint(camNearWorldPoint, camNearPoint);

  int const result = vtkMath::PlaneIntersectsAABB(this->LoadedBounds[i].data(),
    camPlaneNormal, camNearPoint);

  if (result == 0)
  {
    return true;
  }

  return false;
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::RenderVolumeGeometry(
  vtkRenderer* ren, vtkShaderProgram* prog, vtkMultiVolume2* vol)
{
  bool needToInitializeResources = false;
  for (auto& n : this->NeedToInitializeResources)
	  needToInitializeResources |= n.second;
  bool cameraWasInsideInLastUpdate = false;
  for (auto& c : this->CameraWasInsideInLastUpdate)
	  cameraWasInsideInLastUpdate |= c.second;
  bool toRender = needToInitializeResources ||
	              cameraWasInsideInLastUpdate || !this->BBoxPolyData;
  if (!toRender)
	  for (int i = 0; i < this->Parent->GetNumberOfInputConnections(0); ++i)
		  toRender = toRender ||
					 (this->Parent->GetVolumeTexture(i)->UploadTime >
					  this->BBoxPolyData->GetMTime()) ||
					  this->IsCameraInside(i, ren, vol);
  for (int in = 0; in < this->Parent->GetNumberOfInputConnections(0); ++in) {
	  if (toRender) {
		  vtkNew<vtkTessellatedBoxSource> boxSource;
		  boxSource->SetBounds(this->LoadedBounds[in].data());
		  boxSource->QuadsOn();
		  boxSource->SetLevel(0);

		  vtkNew<vtkDensifyPolyData> densityPolyData;

		  if (this->CameraWasInsideInLastUpdate[in] = this->IsCameraInside(in, ren, vol)) {
			  // Normals should be transformed using the transpose of inverse
			  // InverseVolumeMat
			  vtkNew<vtkMatrix4x4> tempMatrix;
			  tempMatrix->DeepCopy(vol->GetVolumeMatrix(in));
			  tempMatrix->Invert();

			  vtkCamera* cam = ren->GetActiveCamera();
			  double camWorldRange[2];
			  double camWorldPos[4];
			  double camFocalWorldPoint[4];
			  double camWorldDirection[4];
			  double camPos[4];
			  double camPlaneNormal[4];

			  cam->GetPosition(camWorldPos);
			  camWorldPos[3] = 1.0;
			  tempMatrix->MultiplyPoint(camWorldPos, camPos);

			  cam->GetFocalPoint(camFocalWorldPoint);
			  camFocalWorldPoint[3] = 1.0;

			  // The range (near/far) must also be transformed
			  // into the local coordinate system.
			  camWorldDirection[0] = camFocalWorldPoint[0] - camWorldPos[0];
			  camWorldDirection[1] = camFocalWorldPoint[1] - camWorldPos[1];
			  camWorldDirection[2] = camFocalWorldPoint[2] - camWorldPos[2];
			  camWorldDirection[3] = 0.0;

			  // Compute the normalized near plane normal
			  tempMatrix->MultiplyPoint(camWorldDirection, camPlaneNormal);

			  vtkMath::Normalize(camWorldDirection);
			  vtkMath::Normalize(camPlaneNormal);

			  double camNearWorldPoint[4];
			  double camFarWorldPoint[4];
			  double camNearPoint[4];
			  double camFarPoint[4];

			  cam->GetClippingRange(camWorldRange);
			  camNearWorldPoint[0] = camWorldPos[0] + camWorldRange[0] * camWorldDirection[0];
			  camNearWorldPoint[1] = camWorldPos[1] + camWorldRange[0] * camWorldDirection[1];
			  camNearWorldPoint[2] = camWorldPos[2] + camWorldRange[0] * camWorldDirection[2];
			  camNearWorldPoint[3] = 1.;

			  camFarWorldPoint[0] = camWorldPos[0] + camWorldRange[1] * camWorldDirection[0];
			  camFarWorldPoint[1] = camWorldPos[1] + camWorldRange[1] * camWorldDirection[1];
			  camFarWorldPoint[2] = camWorldPos[2] + camWorldRange[1] * camWorldDirection[2];
			  camFarWorldPoint[3] = 1.;

			  tempMatrix->MultiplyPoint(camNearWorldPoint, camNearPoint);
			  tempMatrix->MultiplyPoint(camFarWorldPoint, camFarPoint);

			  vtkNew<vtkPlane> nearPlane;

			  // We add an offset to the near plane to avoid hardware clipping of the
			  // near plane due to floating-point precision.
			  // camPlaneNormal is a unit vector, if the offset is larger than the
			  // distance between near and far point, it will not work. Hence, we choose
			  // a fraction of the near-far distance. However, care should be taken
			  // to avoid hardware clipping in volumes with very small spacing where the
			  // distance between near and far plane is also very small. In that case,
			  // a minimum offset is chosen. This is chosen based on the typical
			  // epsilon values on x86 systems.
			  double offset = sqrt(vtkMath::Distance2BetweenPoints(
				  camNearPoint, camFarPoint)) / 1000.0;
			  // Minimum offset to avoid floating point precision issues for volumes
			  // with very small spacing
			  double minOffset = static_cast<double>(
				  std::numeric_limits<float>::epsilon()) * 1000.0;
			  offset = offset < minOffset ? minOffset : offset;

			  camNearPoint[0] += camPlaneNormal[0] * offset;
			  camNearPoint[1] += camPlaneNormal[1] * offset;
			  camNearPoint[2] += camPlaneNormal[2] * offset;

			  nearPlane->SetOrigin(camNearPoint);
			  nearPlane->SetNormal(camPlaneNormal);

			  vtkNew<vtkPlaneCollection> planes;
			  planes->RemoveAllItems();
			  planes->AddItem(nearPlane.GetPointer());

			  vtkNew<vtkClipConvexPolyData> clip;
			  clip->SetInputConnection(boxSource->GetOutputPort());
			  clip->SetPlanes(planes.GetPointer());

			  densityPolyData->SetInputConnection(clip->GetOutputPort());
		  }
		  else
			  densityPolyData->SetInputConnection(boxSource->GetOutputPort());

		  densityPolyData->SetNumberOfSubdivisions(2);
		  densityPolyData->Update();

		  this->BBoxPolyData = vtkSmartPointer<vtkPolyData>::New();
		  this->BBoxPolyData->ShallowCopy(densityPolyData->GetOutput());
		  vtkPoints* points = this->BBoxPolyData->GetPoints();
		  vtkCellArray* cells = this->BBoxPolyData->GetPolys();

		  vtkNew<vtkUnsignedIntArray> polys;
		  polys->SetNumberOfComponents(3);
		  vtkIdType npts;
		  vtkIdType *pts;

		  // See if the volume transform is orientation-preserving
		  // and orient polygons accordingly
		  vtkMatrix4x4* volMat = vol->GetMatrix();
		  double det = vtkMath::Determinant3x3(
			  volMat->GetElement(0, 0), volMat->GetElement(0, 1), volMat->GetElement(0, 2),
			  volMat->GetElement(1, 0), volMat->GetElement(1, 1), volMat->GetElement(1, 2),
			  volMat->GetElement(2, 0), volMat->GetElement(2, 1), volMat->GetElement(2, 2));
		  bool preservesOrientation = det > 0.0;

		  const vtkIdType indexMap[3] = {
			preservesOrientation ? 0 : 2,
			1,
			preservesOrientation ? 2 : 0
		  };

		  while (cells->GetNextCell(npts, pts))
		  {
			  polys->InsertNextTuple3(pts[indexMap[0]], pts[indexMap[1]], pts[indexMap[2]]);
		  }

		  // Dispose any previously created buffers
		  this->DeleteBufferObjects(in);

		  // Now create new ones
		  this->CreateBufferObjects(in);

#ifdef __APPLE__
		  if (vtkOpenGLRenderWindow::GetContextSupportsOpenGL32())
#endif
		  {
			  glBindVertexArray(this->CubeVAOId[in]);
		  }

		  // Pass cube vertices to buffer object memory
		  glBindBuffer(GL_ARRAY_BUFFER, this->CubeVBOId[in]);
		  glBufferData(GL_ARRAY_BUFFER, points->GetData()->GetDataSize() *
			  points->GetData()->GetDataTypeSize(),
			  points->GetData()->GetVoidPointer(0), GL_STATIC_DRAW);

		  prog->EnableAttributeArray("in_vertexPos");
		  prog->UseAttributeArray("in_vertexPos", 0, 0, VTK_FLOAT,
			  3, vtkShaderProgram::NoNormalize);

		  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->CubeIndicesId[in]);
		  glBufferData(GL_ELEMENT_ARRAY_BUFFER, polys->GetDataSize() *
			  polys->GetDataTypeSize(), polys->GetVoidPointer(0),
			  GL_STATIC_DRAW);
	  }
	  else
	  {
#ifdef __APPLE__
		  if (!vtkOpenGLRenderWindow::GetContextSupportsOpenGL32())
		  {
			  glBindBuffer(GL_ARRAY_BUFFER, this->CubeVBOId);
			  prog->EnableAttributeArray("in_vertexPos");
			  prog->UseAttributeArray("in_vertexPos", 0, 0, VTK_FLOAT,
				  3, vtkShaderProgram::NoNormalize);
			  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->CubeIndicesId);
		  }
		  else
#endif
		  {
			  glBindVertexArray(this->CubeVAOId[in]);
		  }
	  }

	  // ???
	  glDepthMask(true);
	  this->ShaderProgram->SetUniformi("in_instanceID", in);
	  glDrawElements(GL_TRIANGLES,
					 this->BBoxPolyData->GetNumberOfCells() * 3,
					 GL_UNSIGNED_INT, 0);

	  vtkOpenGLStaticCheckErrorMacro("Error after glDrawElements in RenderVolumeGeometry");
#ifdef __APPLE__
	  if (!vtkOpenGLRenderWindow::GetContextSupportsOpenGL32())
	  {
		  glBindBuffer(GL_ARRAY_BUFFER, 0);
		  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	  }
	  else
#endif
	  {
		  glBindVertexArray(0);
		  glBindBuffer(GL_ARRAY_BUFFER, 0);
		  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	  }
  }
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::SetCroppingRegions(
  vtkRenderer* vtkNotUsed(ren), vtkShaderProgram* prog,
  vtkMultiVolume2* vtkNotUsed(vol))
{
  /*if (this->Parent->GetCropping())
  {
    int cropFlags = this->Parent->GetCroppingRegionFlags();
    double croppingRegionPlanes[6];
    this->Parent->GetCroppingRegionPlanes(croppingRegionPlanes);

    // Clamp it
    croppingRegionPlanes[0] = croppingRegionPlanes[0] < this->LoadedBounds[0] ?
                              this->LoadedBounds[0] : croppingRegionPlanes[0];
    croppingRegionPlanes[0] = croppingRegionPlanes[0] > this->LoadedBounds[1] ?
                              this->LoadedBounds[1] : croppingRegionPlanes[0];
    croppingRegionPlanes[1] = croppingRegionPlanes[1] < this->LoadedBounds[0] ?
                              this->LoadedBounds[0] : croppingRegionPlanes[1];
    croppingRegionPlanes[1] = croppingRegionPlanes[1] > this->LoadedBounds[1] ?
                              this->LoadedBounds[1] : croppingRegionPlanes[1];

    croppingRegionPlanes[2] = croppingRegionPlanes[2] < this->LoadedBounds[2] ?
                              this->LoadedBounds[2] : croppingRegionPlanes[2];
    croppingRegionPlanes[2] = croppingRegionPlanes[2] > this->LoadedBounds[3] ?
                              this->LoadedBounds[3] : croppingRegionPlanes[2];
    croppingRegionPlanes[3] = croppingRegionPlanes[3] < this->LoadedBounds[2] ?
                              this->LoadedBounds[2] : croppingRegionPlanes[3];
    croppingRegionPlanes[3] = croppingRegionPlanes[3] > this->LoadedBounds[3] ?
                              this->LoadedBounds[3] : croppingRegionPlanes[3];

    croppingRegionPlanes[4] = croppingRegionPlanes[4] < this->LoadedBounds[4] ?
                              this->LoadedBounds[4] : croppingRegionPlanes[4];
    croppingRegionPlanes[4] = croppingRegionPlanes[4] > this->LoadedBounds[5] ?
                              this->LoadedBounds[5] : croppingRegionPlanes[4];
    croppingRegionPlanes[5] = croppingRegionPlanes[5] < this->LoadedBounds[4] ?
                              this->LoadedBounds[4] : croppingRegionPlanes[5];
    croppingRegionPlanes[5] = croppingRegionPlanes[5] > this->LoadedBounds[5] ?
                              this->LoadedBounds[5] : croppingRegionPlanes[5];

    float cropPlanes[6] = { static_cast<float>(croppingRegionPlanes[0]),
                            static_cast<float>(croppingRegionPlanes[1]),
                            static_cast<float>(croppingRegionPlanes[2]),
                            static_cast<float>(croppingRegionPlanes[3]),
                            static_cast<float>(croppingRegionPlanes[4]),
                            static_cast<float>(croppingRegionPlanes[5]) };

    prog->SetUniform1fv("in_croppingPlanes", 6, cropPlanes);
    const int numberOfRegions = 32;
    int cropFlagsArray[numberOfRegions];
    cropFlagsArray[0] = 0;
    int i = 1;
    while(cropFlags && i < 32)
    {
      cropFlagsArray[i] = cropFlags & 1;
      cropFlags = cropFlags >> 1;
      ++i;
    }
    for (; i < 32; ++i)
    {
      cropFlagsArray[i] = 0;
    }

    prog->SetUniform1iv("in_croppingFlags", numberOfRegions, cropFlagsArray);
  }*/
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::SetClippingPlanes(
  vtkRenderer* vtkNotUsed(ren), vtkShaderProgram* prog,
  vtkMultiVolume2* vtkNotUsed(vol))
{
  /*if (this->Parent->GetClippingPlanes())
  {
    std::vector<float> clippingPlanes;
    // Currently we don't have any clipping plane
    clippingPlanes.push_back(0);

    this->Parent->ClippingPlanes->InitTraversal();
    vtkPlane* plane;
    while ((plane = this->Parent->ClippingPlanes->GetNextItem()))
    {
      // Planes are in world coordinates
      double planeOrigin[3], planeNormal[3];
      plane->GetOrigin(planeOrigin);
      plane->GetNormal(planeNormal);

      clippingPlanes.push_back(planeOrigin[0]);
      clippingPlanes.push_back(planeOrigin[1]);
      clippingPlanes.push_back(planeOrigin[2]);
      clippingPlanes.push_back(planeNormal[0]);
      clippingPlanes.push_back(planeNormal[1]);
      clippingPlanes.push_back(planeNormal[2]);
    }

    clippingPlanes[0] = clippingPlanes.size() > 1 ?
                          static_cast<int>(clippingPlanes.size() - 1): 0;

    prog->SetUniform1fv("in_clippingPlanes",
                        static_cast<int>(clippingPlanes.size()),
                        &clippingPlanes[0]);
  }*/
}

// -----------------------------------------------------------------------------
void
vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::CheckPropertyKeys(vtkMultiVolume2* vol)
{
  // Check the property keys to see if we should modify the blend/etc state:
  // Otherwise this breaks volume/translucent geo depth peeling.
  vtkInformation *volumeKeys = vol->GetPropertyKeys();
  this->PreserveGLState = false;
  if (volumeKeys && volumeKeys->Has(vtkOpenGLActor::GLDepthMaskOverride()))
  {
    int override = volumeKeys->Get(vtkOpenGLActor::GLDepthMaskOverride());
    if (override != 0 && override != 1)
    {
      this->PreserveGLState = true;
    }
  }

  // Some render passes (e.g. DualDepthPeeling) adjust the viewport for
  // intermediate passes so it is necessary to preserve it. This is a
  // temporary fix for vtkDualDepthPeelingPass to work when various viewports
  // are defined.  The correct way of fixing this would be to avoid setting the
  // viewport within the mapper.  It is enough for now to check for the
  // RenderPasses() vtkInfo given that vtkDualDepthPeelingPass is the only pass
  // currently supported by this mapper, the viewport will have to be adjusted
  // externally before adding support for other passes.
  vtkInformation *info = vol->GetPropertyKeys();
  this->PreserveViewport = info && info->Has(
    vtkOpenGLRenderPass::RenderPasses());
}

// -----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::CheckPickingState(vtkRenderer* ren)
{
  vtkHardwareSelector* selector = ren->GetSelector();
  bool selectorPicking = selector != NULL;
  if (selector)
  {
    // this mapper currently only supports cell picking
    selectorPicking &= selector->GetFieldAssociation() == vtkDataObject::FIELD_ASSOCIATION_CELLS;
  }

  this->IsPicking = selectorPicking || ren->GetRenderWindow()->GetIsPicking();
  if (this->IsPicking)
  {
    // rebuild the shader on every pass
    this->SelectionStateTime.Modified();
    this->CurrentSelectionPass = selector ? selector->GetCurrentPass() : vtkHardwareSelector::ACTOR_PASS;
  }
  else if (this->CurrentSelectionPass != vtkHardwareSelector::MIN_KNOWN_PASS - 1)
  {
    // return to the regular rendering state
    this->SelectionStateTime.Modified();
    this->CurrentSelectionPass = vtkHardwareSelector::MIN_KNOWN_PASS - 1;
  }
}

// -----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::BeginPicking(vtkRenderer* ren)
{
  vtkHardwareSelector* selector = ren->GetSelector();
  if (selector && this->IsPicking)
  {
    selector->BeginRenderProp();

    //if (this->CurrentSelectionPass >= vtkHardwareSelector::ID_LOW24)
    //{
    //  selector->RenderAttributeId(0);
    //}
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::SetPickingId
  (vtkRenderer* ren)
{
  float propIdColor[3] = {0.0, 0.0, 0.0};
  vtkHardwareSelector* selector = ren->GetSelector();

  if (selector && this->IsPicking)
  {
    // query the selector for the appropriate id
    selector->GetPropColorValue(propIdColor);
  }
  else // RenderWindow is picking
  {
    //unsigned int const idx = ren->GetCurrentPickId();
    //vtkHardwareSelector::Convert(idx, propIdColor);
  }

  this->ShaderProgram->SetUniform3f("in_propId", propIdColor);
}

// ---------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::EndPicking(vtkRenderer* ren)
{
  /*vtkHardwareSelector* selector = ren->GetSelector();
  if(selector && this->IsPicking)
  {
    if (this->CurrentSelectionPass >= vtkHardwareSelector::ID_LOW24)
    {
      // tell the selector the maximum number of cells that the mapper could render
      unsigned int const numVoxels = (this->Extents[1] - this->Extents[0]) *
        (this->Extents[3] - this->Extents[2]) * (this->Extents[5] - this->Extents[4]);
      selector->RenderAttributeId(numVoxels);
    }
    selector->EndRenderProp();
  }*/
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::UpdateSamplingDistance(int in,
  vtkImageData* input, vtkRenderer* vtkNotUsed(ren), vtkMultiVolume2* vol)
{
  if (!this->Parent->AutoAdjustSampleDistances)
  {
    if (this->Parent->LockSampleDistanceToInputSpacing)
    {
      float const d = static_cast<float>(this->Parent->SpacingAdjustedSampleDistance(
        this->CellSpacing[in].data(), this->Extents[in].data()));
      float const sample = this->Parent->SampleDistance;

      // ActualSampleDistance will grow proportionally to numVoxels^(1/3) (see
      // vtkVolumeMapper.cxx). Until it reaches 1/2 average voxel size when number of
      // voxels is 1E6.
      this->ActualSampleDistance[in] = (sample / d < 0.999f || sample / d > 1.001f) ?
        d : this->Parent->SampleDistance;

      return;
    }

    this->ActualSampleDistance[in] = this->Parent->SampleDistance;
  }
  else
  {
    input->GetSpacing(this->CellSpacing[in].data());
    vtkMatrix4x4* worldToDataset = vol->GetVolumeMatrix(in);
    double minWorldSpacing = VTK_DOUBLE_MAX;
    int i = 0;
    while (i < 3)
    {
      double tmp = worldToDataset->GetElement(0,i);
      double tmp2 = tmp * tmp;
      tmp = worldToDataset->GetElement(1,i);
      tmp2 += tmp * tmp;
      tmp = worldToDataset->GetElement(2,i);
      tmp2 += tmp * tmp;

      // We use fabs() in case the spacing is negative.
      double worldSpacing = fabs(this->CellSpacing[in][i] * sqrt(tmp2));
      if(worldSpacing < minWorldSpacing)
      {
        minWorldSpacing = worldSpacing;
      }
      ++i;
    }

    // minWorldSpacing is the optimal sample distance in world space.
    // To go faster (reduceFactor<1.0), we multiply this distance
    // by 1/reduceFactor.
    this->ActualSampleDistance[in] = static_cast<float>(minWorldSpacing);

    if (this->Parent->ReductionFactor < 1.0 &&
        this->Parent->ReductionFactor != 0.0)
    {
      this->ActualSampleDistance[in] /= static_cast<GLfloat>(this->Parent->ReductionFactor);
    }
  }
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::
  LoadRequireDepthTextureExtensions(vtkRenderWindow* vtkNotUsed(renWin))
{
  // Reset the message stream for extensions
  //if (vtkOpenGLRenderWindow::GetContextSupportsOpenGL32())
  {
    this->LoadDepthTextureExtensionsSucceeded = true;
    return;
  }

  this->ExtensionsStringStream.str("");
  this->ExtensionsStringStream.clear();

#if GL_ES_VERSION_3_0 != 1
  // Check for float texture support. This extension became core
  // in 3.0
  if (!glewIsSupported("GL_ARB_texture_float"))
  {
    this->ExtensionsStringStream << "Required extension "
      << " GL_ARB_texture_float is not supported";
    return;
  }
#endif

  // NOTE: Support for depth sampler texture made into the core since version
  // 1.4 and therefore we are no longer checking for it.
  this->LoadDepthTextureExtensionsSucceeded = true;
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::CreateBufferObjects(const int i)
{
#ifdef __APPLE__
  if (vtkOpenGLRenderWindow::GetContextSupportsOpenGL32())
#endif
  {
    glGenVertexArrays(1, &this->CubeVAOId[i]);
  }
  glGenBuffers(1, &this->CubeVBOId[i]);
  glGenBuffers(1, &this->CubeIndicesId[i]);
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::DeleteBufferObjects(const int i)
{
  if (this->CubeVBOId[i])
  {
    glBindBuffer (GL_ARRAY_BUFFER, this->CubeVBOId[i]);
    glDeleteBuffers(1, &this->CubeVBOId[i]);
    this->CubeVBOId[i] = 0;
  }

  if (this->CubeIndicesId[i])
  {
   glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, this->CubeIndicesId[i]);
   glDeleteBuffers(1, &this->CubeIndicesId[i]);
    this->CubeIndicesId[i] = 0;
  }

  if (this->CubeVAOId[i])
  {
#ifdef __APPLE__
  if (vtkOpenGLRenderWindow::GetContextSupportsOpenGL32())
#endif
  {
      glDeleteVertexArrays(1, &this->CubeVAOId[i]);
  }
    this->CubeVAOId[i] = 0;
  }
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::
  ConvertTextureToImageData(vtkTextureObject* texture, vtkImageData* output)
{
  if (!texture)
  {
    return;
  }
  unsigned int tw = texture->GetWidth();
  unsigned int th = texture->GetHeight();
  unsigned int tnc = texture->GetComponents();
  int tt = texture->GetVTKDataType();

  vtkPixelExtent texExt(0U, tw-1U, 0U, th-1U);

  int dataExt[6]={0,0, 0,0, 0,0};
  texExt.GetData(dataExt);

  double dataOrigin[6] = {0, 0, 0, 0, 0, 0};

  vtkImageData *id = vtkImageData::New();
  id->SetOrigin(dataOrigin);
  id->SetDimensions(tw, th, 1);
  id->SetExtent(dataExt);
  id->AllocateScalars(tt, tnc);

  vtkPixelBufferObject *pbo = texture->Download();

  vtkPixelTransfer::Blit(texExt,
                         texExt,
                         texExt,
                         texExt,
                         tnc,
                         tt,
                         pbo->MapPackedBuffer(),
                         tnc,
                         tt,
                         id->GetScalarPointer(0, 0, 0));

  pbo->UnmapPackedBuffer();
  pbo->Delete();

  if (!output)
  {
    output = vtkImageData::New();
  }
  output->DeepCopy(id);
  id->Delete();
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::BeginImageSample(
  vtkRenderer* ren, vtkMultiVolume2* vol)
{
  const auto numBuffers = this->GetNumImageSampleDrawBuffers(vol);
  if (numBuffers != this->NumImageSampleDrawBuffers)
  {
    if (numBuffers > this->NumImageSampleDrawBuffers)
    {
      this->ReleaseImageSampleGraphicsResources(ren->GetRenderWindow());
    }

    this->NumImageSampleDrawBuffers = numBuffers;
    this->RebuildImageSampleProg = true;
  }

  float const xySampleDist = this->Parent->ImageSampleDistance;
  if (xySampleDist != 1.f && this->InitializeImageSampleFBO(ren))
  {
    this->ImageSampleFBO->SaveCurrentBindingsAndBuffers(GL_DRAW_FRAMEBUFFER);
    this->ImageSampleFBO->DeactivateDrawBuffers();
    this->ImageSampleFBO->Bind(GL_DRAW_FRAMEBUFFER);
    this->ImageSampleFBO->ActivateDrawBuffers(static_cast<unsigned int>(
      this->NumImageSampleDrawBuffers));

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
  }
}

//----------------------------------------------------------------------------
bool vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::InitializeImageSampleFBO(
  vtkRenderer* ren)
{
  // Set the FBO viewport size. These are used in the shader to normalize the
  // fragment coordinate, the normalized coordinate is used to fetch the depth
  // buffer.
  this->WindowSize[0] /= this->Parent->ImageSampleDistance;
  this->WindowSize[1] /= this->Parent->ImageSampleDistance;
  this->WindowLowerLeft[0] = 0;
  this->WindowLowerLeft[1] = 0;

  // Set FBO viewport
  glViewport(this->WindowLowerLeft[0], this->WindowLowerLeft[1],
    this->WindowSize[0], this->WindowSize[1]);

  if (!this->ImageSampleFBO)
  {
    vtkOpenGLRenderWindow* win = vtkOpenGLRenderWindow::SafeDownCast(
      ren->GetRenderWindow());

    this->ImageSampleTexture.reserve(this->NumImageSampleDrawBuffers);
    this->ImageSampleTexNames.reserve(this->NumImageSampleDrawBuffers);
    for (size_t i = 0; i < this->NumImageSampleDrawBuffers; i++)
    {
      auto tex = vtkSmartPointer<vtkTextureObject>::New();
      tex->SetContext(win);
      tex->Create2D(this->WindowSize[0], this->WindowSize[1],
        4, VTK_UNSIGNED_CHAR, false);
      tex->Activate();
      tex->SetMinificationFilter(vtkTextureObject::Linear);
      tex->SetMagnificationFilter(vtkTextureObject::Linear);
      tex->SetWrapS(vtkTextureObject::ClampToEdge);
      tex->SetWrapT(vtkTextureObject::ClampToEdge);
      this->ImageSampleTexture.push_back(tex);

      std::stringstream ss; ss << i;
      const std::string name = "renderedTex_" + ss.str();
      this->ImageSampleTexNames.push_back(name);
    }

    this->ImageSampleFBO = vtkOpenGLFramebufferObject::New();
    this->ImageSampleFBO->SetContext(win);
    this->ImageSampleFBO->SaveCurrentBindingsAndBuffers(GL_FRAMEBUFFER);
    this->ImageSampleFBO->Bind(GL_FRAMEBUFFER);
    this->ImageSampleFBO->InitializeViewport(this->WindowSize[0],
      this->WindowSize[1]);

    auto num = static_cast<unsigned int>(this->NumImageSampleDrawBuffers);
    for (unsigned int i = 0; i < num; i++)
    {
      this->ImageSampleFBO->AddColorAttachment(GL_FRAMEBUFFER, i,
        this->ImageSampleTexture[i]);
    }

    // Verify completeness
    const int complete = this->ImageSampleFBO->CheckFrameBufferStatus(GL_FRAMEBUFFER);
    for (auto& tex : this->ImageSampleTexture)
    {
      tex->Deactivate();
    }
    this->ImageSampleFBO->RestorePreviousBindingsAndBuffers(GL_FRAMEBUFFER);

    if(!complete)
    {
      vtkGenericWarningMacro(<< "Failed to attach ImageSampleFBO!");
      this->ReleaseImageSampleGraphicsResources(win);
      return false;
    }

    this->RebuildImageSampleProg = true;
    return true;
  }

  // Resize if necessary
  int lastSize[2];
  this->ImageSampleFBO->GetLastSize(lastSize);
  if (lastSize[0] != this->WindowSize[0] || lastSize[1] != this->WindowSize[1])
  {
    this->ImageSampleFBO->Resize(this->WindowSize[0], this->WindowSize[1]);
  }

  return true;
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::EndImageSample(
  vtkRenderer* ren)
{
  if (this->Parent->ImageSampleDistance != 1.f)
  {
    this->ImageSampleFBO->DeactivateDrawBuffers();
    this->ImageSampleFBO->RestorePreviousBindingsAndBuffers(GL_DRAW_FRAMEBUFFER);
    if (this->RenderPassAttached)
    {
      this->ImageSampleFBO->ActivateDrawBuffers(static_cast<unsigned int>(
        this->NumImageSampleDrawBuffers));
    }

    // Render the contents of ImageSampleFBO as a quad to intermix with the
    // rest of the scene.
    typedef vtkOpenGLRenderUtilities GLUtil;
    vtkOpenGLRenderWindow* win = static_cast<vtkOpenGLRenderWindow*>(
      ren->GetRenderWindow());

    if (this->RebuildImageSampleProg)
    {
      std::string frag = GLUtil::GetFullScreenQuadFragmentShaderTemplate();

      vtkShaderProgram::Substitute(frag, "//VTK::FSQ::Decl",
        vtkmvolume::ImageSampleDeclarationFrag(this->ImageSampleTexNames,
          this->NumImageSampleDrawBuffers));
      vtkShaderProgram::Substitute(frag, "//VTK::FSQ::Impl",
        vtkmvolume::ImageSampleImplementationFrag(this->ImageSampleTexNames,
          this->NumImageSampleDrawBuffers));

      this->ImageSampleProg = win->GetShaderCache()->ReadyShaderProgram(
        GLUtil::GetFullScreenQuadVertexShader().c_str(), frag.c_str(),
        GLUtil::GetFullScreenQuadGeometryShader().c_str());
    }
    else
    {
      win->GetShaderCache()->ReadyShaderProgram(this->ImageSampleProg);
    }

    if (!this->ImageSampleProg)
    {
      vtkGenericWarningMacro(<< "Failed to initialize ImageSampleProgram!");
      return;
    }

    if (!this->ImageSampleVAO)
    {
      this->ImageSampleVBO = vtkOpenGLBufferObject::New();
      this->ImageSampleVAO = vtkOpenGLVertexArrayObject::New();
      GLUtil::PrepFullScreenVAO(this->ImageSampleVBO, this->ImageSampleVAO,
        this->ImageSampleProg);
    }

    // Adjust the GL viewport to VTK's defined viewport
    ren->GetTiledSizeAndOrigin(this->WindowSize, this->WindowSize + 1,
      this->WindowLowerLeft, this->WindowLowerLeft + 1);
    glViewport(this->WindowLowerLeft[0], this->WindowLowerLeft[1],
      this->WindowSize[0], this->WindowSize[1]);

    // Bind objects and draw
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    for (size_t i = 0; i < this->NumImageSampleDrawBuffers; i++)
    {
      this->ImageSampleTexture[i]->Activate();
      this->ImageSampleProg->SetUniformi(this->ImageSampleTexNames[i].c_str(),
        this->ImageSampleTexture[i]->GetTextureUnit());
    }

    this->ImageSampleVAO->Bind();
    GLUtil::DrawFullScreenQuad();
    this->ImageSampleVAO->Release();
    vtkOpenGLStaticCheckErrorMacro("Error after DrawFullScreenQuad()!");

    for (auto& tex : this->ImageSampleTexture)
    {
      tex->Deactivate();
    }
  }
}

//------------------------------------------------------------------------------
size_t vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::GetNumImageSampleDrawBuffers(
  vtkMultiVolume2* vol)
{
  if (this->RenderPassAttached)
  {
    vtkInformation* info = vol->GetPropertyKeys();
    const int num = info->Length(vtkOpenGLRenderPass::RenderPasses());
    vtkObjectBase *rpBase = info->Get(vtkOpenGLRenderPass::RenderPasses(), num - 1);
    vtkOpenGLRenderPass *rp = static_cast<vtkOpenGLRenderPass*>(rpBase);
    return static_cast<size_t>(rp->GetActiveDrawBuffers());
  }

  return 1;
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::SetupRenderToTexture(
  vtkRenderer* ren)
{
  if (this->Parent->RenderToImage && this->Parent->CurrentPass == RenderPass)
  {
    if (this->Parent->ImageSampleDistance != 1.f)
    {
      this->WindowSize[0] /= this->Parent->ImageSampleDistance;
      this->WindowSize[1] /= this->Parent->ImageSampleDistance;
    }

    if ( (this->LastRenderToImageWindowSize[0] != this->WindowSize[0]) ||
         (this->LastRenderToImageWindowSize[1] != this->WindowSize[1]) )
    {
      this->LastRenderToImageWindowSize[0] = this->WindowSize[0];
      this->LastRenderToImageWindowSize[1] = this->WindowSize[1];
      this->ReleaseRenderToTextureGraphicsResources(ren->GetRenderWindow());
    }

    if (!this->FBO)
    {
      this->FBO = vtkOpenGLFramebufferObject::New();
    }

    this->FBO->SetContext(vtkOpenGLRenderWindow::SafeDownCast(
                          ren->GetRenderWindow()));

    this->FBO->SaveCurrentBindingsAndBuffers();
    this->FBO->Bind(GL_FRAMEBUFFER);
    this->FBO->InitializeViewport(this->WindowSize[0], this->WindowSize[1]);

    int depthImageScalarType = this->Parent->GetDepthImageScalarType();
    bool initDepthTexture = true;
    // Re-instantiate the depth texture object if the scalar type requested has
    // changed from the last frame
    if (this->RTTDepthTextureObject &&
        this->RTTDepthTextureType == depthImageScalarType)
    {
      initDepthTexture = false;
    }

    if (initDepthTexture)
    {
      if (this->RTTDepthTextureObject)
      {
        this->RTTDepthTextureObject->Delete();
        this->RTTDepthTextureObject = 0;
      }
      this->RTTDepthTextureObject = vtkTextureObject::New();
      this->RTTDepthTextureObject->SetContext(
        vtkOpenGLRenderWindow::SafeDownCast(
        ren->GetRenderWindow()));
      this->RTTDepthTextureObject->Create2D(this->WindowSize[0],
                                            this->WindowSize[1], 1,
                                            depthImageScalarType, false);
      this->RTTDepthTextureObject->Activate();
      this->RTTDepthTextureObject->SetMinificationFilter(
        vtkTextureObject::Nearest);
      this->RTTDepthTextureObject->SetMagnificationFilter(
        vtkTextureObject::Nearest);
      this->RTTDepthTextureObject->SetAutoParameters(0);

      // Cache the value of the scalar type
      this->RTTDepthTextureType = depthImageScalarType;
    }

    if (!this->RTTColorTextureObject)
    {
      this->RTTColorTextureObject = vtkTextureObject::New();

      this->RTTColorTextureObject->SetContext(
        vtkOpenGLRenderWindow::SafeDownCast(
        ren->GetRenderWindow()));
      this->RTTColorTextureObject->Create2D(this->WindowSize[0],
                                            this->WindowSize[1], 4,
                                            VTK_UNSIGNED_CHAR, false);
      this->RTTColorTextureObject->Activate();
      this->RTTColorTextureObject->SetMinificationFilter(
        vtkTextureObject::Nearest);
      this->RTTColorTextureObject->SetMagnificationFilter(
        vtkTextureObject::Nearest);
      this->RTTColorTextureObject->SetAutoParameters(0);
    }

    if (!this->RTTDepthBufferTextureObject)
    {
      this->RTTDepthBufferTextureObject = vtkTextureObject::New();
      this->RTTDepthBufferTextureObject->SetContext(
        vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow()));
      this->RTTDepthBufferTextureObject->AllocateDepth(
          this->WindowSize[0], this->WindowSize[1], vtkTextureObject::Float32);
      this->RTTDepthBufferTextureObject->Activate();
      this->RTTDepthBufferTextureObject->SetMinificationFilter(
        vtkTextureObject::Nearest);
      this->RTTDepthBufferTextureObject->SetMagnificationFilter(
        vtkTextureObject::Nearest);
      this->RTTDepthBufferTextureObject->SetAutoParameters(0);
    }

    this->FBO->Bind(GL_FRAMEBUFFER);
    this->FBO->AddDepthAttachment(
      GL_FRAMEBUFFER,
      this->RTTDepthBufferTextureObject);
    this->FBO->AddColorAttachment(
      GL_FRAMEBUFFER, 0U,
      this->RTTColorTextureObject);
    this->FBO->AddColorAttachment(
      GL_FRAMEBUFFER, 1U,
      this->RTTDepthTextureObject);
    this->FBO->ActivateDrawBuffers(2);

    this->FBO->CheckFrameBufferStatus(GL_FRAMEBUFFER);

    glClearColor(1.0, 1.0, 1.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  }
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::ExitRenderToTexture(
  vtkRenderer* vtkNotUsed(ren))
{
  if (this->Parent->RenderToImage && this->Parent->CurrentPass == RenderPass)
  {
    this->FBO->RemoveTexDepthAttachment(GL_FRAMEBUFFER);
    this->FBO->RemoveTexColorAttachment(GL_FRAMEBUFFER, 0U);
    this->FBO->RemoveTexColorAttachment(GL_FRAMEBUFFER, 1U);
    this->FBO->DeactivateDrawBuffers();
    this->FBO->RestorePreviousBindingsAndBuffers();

    this->RTTDepthBufferTextureObject->Deactivate();
    this->RTTColorTextureObject->Deactivate();
    this->RTTDepthTextureObject->Deactivate();
  }
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::SetupDepthPass(
  vtkRenderer* ren)
{
  if (this->Parent->ImageSampleDistance != 1.f)
  {
    this->WindowSize[0] /= this->Parent->ImageSampleDistance;
    this->WindowSize[1] /= this->Parent->ImageSampleDistance;
  }

  if ( (this->LastDepthPassWindowSize[0] != this->WindowSize[0]) ||
       (this->LastDepthPassWindowSize[1] != this->WindowSize[1]) )
  {
    this->LastDepthPassWindowSize[0] = this->WindowSize[0];
    this->LastDepthPassWindowSize[1] = this->WindowSize[1];
    this->ReleaseDepthPassGraphicsResources(ren->GetRenderWindow());
  }

  if (!this->DPFBO)
  {
    this->DPFBO = vtkOpenGLFramebufferObject::New();
  }

  this->DPFBO->SetContext(vtkOpenGLRenderWindow::SafeDownCast(
                        ren->GetRenderWindow()));

  this->DPFBO->SaveCurrentBindingsAndBuffers();
  this->DPFBO->Bind(GL_FRAMEBUFFER);
  this->DPFBO->InitializeViewport(this->WindowSize[0], this->WindowSize[1]);

  if (!this->DPDepthBufferTextureObject ||
      !this->DPColorTextureObject)
  {
    this->DPDepthBufferTextureObject = vtkTextureObject::New();
    this->DPDepthBufferTextureObject->SetContext(
      vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow()));
    this->DPDepthBufferTextureObject->AllocateDepth(
        this->WindowSize[0], this->WindowSize[1], vtkTextureObject::Native);
    this->DPDepthBufferTextureObject->Activate();
    this->DPDepthBufferTextureObject->SetMinificationFilter(
      vtkTextureObject::Nearest);
    this->DPDepthBufferTextureObject->SetMagnificationFilter(
      vtkTextureObject::Nearest);
    this->DPDepthBufferTextureObject->SetAutoParameters(0);
    this->DPDepthBufferTextureObject->Bind();


    this->DPColorTextureObject = vtkTextureObject::New();

    this->DPColorTextureObject->SetContext(
      vtkOpenGLRenderWindow::SafeDownCast(
      ren->GetRenderWindow()));
    this->DPColorTextureObject->Create2D(this->WindowSize[0],
                                         this->WindowSize[1], 4,
                                         VTK_UNSIGNED_CHAR, false);
    this->DPColorTextureObject->Activate();
    this->DPColorTextureObject->SetMinificationFilter(
      vtkTextureObject::Nearest);
    this->DPColorTextureObject->SetMagnificationFilter(
      vtkTextureObject::Nearest);
    this->DPColorTextureObject->SetAutoParameters(0);

    this->DPFBO->AddDepthAttachment(
      GL_FRAMEBUFFER,
      this->DPDepthBufferTextureObject);

    this->DPFBO->AddColorAttachment(
      GL_FRAMEBUFFER, 0U,
      this->DPColorTextureObject);
  }


  this->DPFBO->ActivateDrawBuffers(1);
  this->DPFBO->CheckFrameBufferStatus(GL_FRAMEBUFFER);

  // Setup the contour polydata mapper to render to DPFBO
  this->ContourMapper->SetInputConnection(
    this->ContourFilter->GetOutputPort());

  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal::ExitDepthPass(
  vtkRenderer* vtkNotUsed(ren))
{
  this->DPFBO->DeactivateDrawBuffers();
  this->DPFBO->RestorePreviousBindingsAndBuffers();

  this->DPDepthBufferTextureObject->Deactivate();
  this->DPColorTextureObject->Deactivate();
  glDisable(GL_DEPTH_TEST);
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal
  ::ReleaseRenderToTextureGraphicsResources(vtkWindow* win)
{
  vtkOpenGLRenderWindow *rwin =
   vtkOpenGLRenderWindow::SafeDownCast(win);

  if (rwin)
  {
    if (this->FBO)
    {
      this->FBO->Delete();
      this->FBO = 0;
    }

    if (this->RTTDepthBufferTextureObject)
    {
      this->RTTDepthBufferTextureObject->ReleaseGraphicsResources(win);
      this->RTTDepthBufferTextureObject->Delete();
      this->RTTDepthBufferTextureObject = 0;
    }

    if (this->RTTDepthTextureObject)
    {
      this->RTTDepthTextureObject->ReleaseGraphicsResources(win);
      this->RTTDepthTextureObject->Delete();
      this->RTTDepthTextureObject = 0;
    }

    if (this->RTTColorTextureObject)
    {
      this->RTTColorTextureObject->ReleaseGraphicsResources(win);
      this->RTTColorTextureObject->Delete();
      this->RTTColorTextureObject = 0;
    }
  }
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal
  ::ReleaseDepthPassGraphicsResources(vtkWindow* win)
{
  vtkOpenGLRenderWindow *rwin =
   vtkOpenGLRenderWindow::SafeDownCast(win);

  if (rwin)
  {
    if (this->DPFBO)
    {
      this->DPFBO->Delete();
      this->DPFBO = 0;
    }

    if (this->DPDepthBufferTextureObject)
    {
      this->DPDepthBufferTextureObject->ReleaseGraphicsResources(win);
      this->DPDepthBufferTextureObject->Delete();
      this->DPDepthBufferTextureObject = 0;
    }

    if (this->DPColorTextureObject)
    {
      this->DPColorTextureObject->ReleaseGraphicsResources(win);
      this->DPColorTextureObject->Delete();
      this->DPColorTextureObject = 0;
    }

    this->ContourMapper->ReleaseGraphicsResources(win);
  }
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::vtkInternal
  ::ReleaseImageSampleGraphicsResources(vtkWindow* win)
{
  vtkOpenGLRenderWindow *rwin =
   vtkOpenGLRenderWindow::SafeDownCast(win);

  if (rwin)
  {
    if (this->ImageSampleFBO)
    {
      this->ImageSampleFBO->Delete();
      this->ImageSampleFBO = nullptr;
    }

    for (auto& tex : this->ImageSampleTexture)
    {
      tex->ReleaseGraphicsResources(win);
      tex = nullptr;
    }
    this->ImageSampleTexture.clear();
    this->ImageSampleTexNames.clear();

    if (this->ImageSampleVBO)
    {
      this->ImageSampleVBO->Delete();
      this->ImageSampleVBO = nullptr;
    }

    if (this->ImageSampleVAO)
    {
      this->ImageSampleVAO->Delete();
      this->ImageSampleVAO = nullptr;
    }

    // Do not delete the shader program - Let the cache clean it up.
    this->ImageSampleProg = nullptr;
  }
}

//----------------------------------------------------------------------------
vtkOpenGLGPUMultiVolumeRayCastMapper::vtkOpenGLGPUMultiVolumeRayCastMapper() :
  vtkGPUMultiVolumeRayCastMapper()
{
  this->Impl = new vtkInternal(this);
  this->ReductionFactor = 1.0;
  this->CurrentPass = RenderPass;
  this->NoiseTextureSize[0] = this->NoiseTextureSize[1] = -1;
  this->NoiseGenerator = NULL;

  this->ResourceCallback = new vtkOpenGLResourceFreeCallback<vtkOpenGLGPUMultiVolumeRayCastMapper>(this,
    &vtkOpenGLGPUMultiVolumeRayCastMapper::ReleaseGraphicsResources);
}

//----------------------------------------------------------------------------
vtkOpenGLGPUMultiVolumeRayCastMapper::~vtkOpenGLGPUMultiVolumeRayCastMapper()
{
  if (this->ResourceCallback)
  {
    this->ResourceCallback->Release();
    delete this->ResourceCallback;
    this->ResourceCallback = NULL;
  }

  if (this->NoiseGenerator)
  {
    this->NoiseGenerator->Delete();
    this->NoiseGenerator = NULL;
  }

  delete this->Impl;
  this->Impl = NULL;

  for (int i = 0; i < this->VolumeTexture.size(); ++i) {
	  if (this->VolumeTexture[i]) {
		  this->VolumeTexture[i]->Delete();
		  this->VolumeTexture[i] = NULL;
	  }
  }
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "ReductionFactor: " << this->ReductionFactor << "\n";
  os << indent << "CurrentPass: " << this->CurrentPass << "\n";
}

//----------------------------------------------------------------------------
vtkTextureObject* vtkOpenGLGPUMultiVolumeRayCastMapper::GetDepthTexture()
{
  return this->Impl->RTTDepthTextureObject;
}

//----------------------------------------------------------------------------
vtkTextureObject* vtkOpenGLGPUMultiVolumeRayCastMapper::GetColorTexture()
{
 return this->Impl->RTTColorTextureObject;
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::GetDepthImage(vtkImageData* output)
{
  return this->Impl->ConvertTextureToImageData(
    this->Impl->RTTDepthTextureObject, output);
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::GetColorImage(vtkImageData* output)
{
  return this->Impl->ConvertTextureToImageData(
    this->Impl->RTTColorTextureObject, output);
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::ReleaseGraphicsResources(
  vtkWindow *window)
{
  if (!this->ResourceCallback->IsReleasing())
  {
    this->ResourceCallback->Release();
    return;
  }

  for (int i = 0; i < this->Impl->CubeVAOId.size(); ++i)
	  this->Impl->DeleteBufferObjects(i);

  for (int i = 0; i < this->VolumeTexture.size(); ++i)
  if (this->VolumeTexture[i])
  {
    this->VolumeTexture[i]->ReleaseGraphicsResources(window);
  }

  if (this->Impl->NoiseTextureObject)
  {
    this->Impl->NoiseTextureObject->ReleaseGraphicsResources(window);
    this->Impl->NoiseTextureObject->Delete();
    this->Impl->NoiseTextureObject = NULL;
  }

  if (this->Impl->DepthTextureObject)
  {
    this->Impl->DepthTextureObject->ReleaseGraphicsResources(window);
    this->Impl->DepthTextureObject->Delete();
    this->Impl->DepthTextureObject = NULL;
  }

  this->Impl->ReleaseRenderToTextureGraphicsResources(window);
  this->Impl->ReleaseDepthPassGraphicsResources(window);
  this->Impl->ReleaseImageSampleGraphicsResources(window);

  if(this->Impl->MaskTextures != NULL)
  {
    if(!this->Impl->MaskTextures->Map.empty())
    {
      std::map<vtkImageData*, vtkVolumeMask*>::iterator it =
        this->Impl->MaskTextures->Map.begin();
      while(it != this->Impl->MaskTextures->Map.end())
      {
        vtkVolumeMask* texture = (*it).second;
        texture->ReleaseGraphicsResources(window);
        delete texture;
        ++it;
      }
      this->Impl->MaskTextures->Map.clear();
    }
  }

  for (int i = 0; i < this->Impl->RGBTables.size(); ++i)
  {
	  if (this->Impl->RGBTables[i])
	  {
		  this->Impl->RGBTables[i]->ReleaseGraphicsResources(window);
		  delete this->Impl->RGBTables[i];
		  this->Impl->RGBTables[i] = NULL;
	  }
  }

  if(this->Impl->Mask1RGBTable)
  {
    this->Impl->Mask1RGBTable->ReleaseGraphicsResources(window);
    this->Impl->Mask1RGBTable->Delete();
    this->Impl->Mask1RGBTable = NULL;
  }

  if(this->Impl->Mask2RGBTable)
  {
    this->Impl->Mask2RGBTable->ReleaseGraphicsResources(window);
    this->Impl->Mask2RGBTable->Delete();
    this->Impl->Mask2RGBTable = NULL;
  }

  for (int i = 0; i < this->Impl->OpacityTables.size(); ++i)
	  if(this->Impl->OpacityTables[i])
	  {
		this->Impl->OpacityTables[i]->ReleaseGraphicsResources(window);
		delete this->Impl->OpacityTables[i];
		this->Impl->OpacityTables[i] = NULL;
	  }

  for (int i = 0; i < this->Impl->GradientOpacityTables.size(); ++i)
	  if (this->Impl->GradientOpacityTables[i])
	  {
		this->Impl->GradientOpacityTables[i]->ReleaseGraphicsResources(window);
		delete this->Impl->GradientOpacityTables[i];
		this->Impl->GradientOpacityTables[i] = NULL;
	  }

  this->Impl->ReleaseResourcesTime.Modified();
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::BuildShader(vtkRenderer* ren,
                                                       vtkMultiVolume2* vol,
                                                       int noOfComponents)
{
  std::string vertexShader (multiraycastervs);
  std::string fragmentShader (multiraycasterfs);

  this->ReplaceShaderRenderPass(vertexShader, fragmentShader, vol, true);

  // Every volume should have a property (cannot be NULL);
  vtkVolumeProperty* volumeProperty = vol->GetProperty(0);
  int independentComponents = volumeProperty->GetIndependentComponents();

  if (volumeProperty->GetShade())
  {
    vtkLightCollection* lc = ren->GetLights();
    vtkLight* light;
    this->Impl->NumberOfLights = 0;

    // Compute light complexity.
    vtkCollectionSimpleIterator sit;
    for (lc->InitTraversal(sit); (light = lc->GetNextLight(sit)); )
    {
      float status = light->GetSwitch();
      if (status > 0.0)
      {
        this->Impl->NumberOfLights++;
        if (this->Impl->LightComplexity == 0)
        {
          this->Impl->LightComplexity = 1;
        }
      }

      if (this->Impl->LightComplexity == 1
          && (this->Impl->NumberOfLights > 1
            || light->GetIntensity() != 1.0
            || light->GetLightType() != VTK_LIGHT_TYPE_HEADLIGHT))
      {
        this->Impl->LightComplexity = 2;
      }

      if (this->Impl->LightComplexity < 3
          && (light->GetPositional()))
      {
        this->Impl->LightComplexity = 3;
        break;
      }
    }
  }

  // Base methods replacements
  //--------------------------------------------------------------------------
  vertexShader = vtkmvolume::replace(
    vertexShader,
    "//VTK::Base::Dec",
    vtkmvolume::BaseDeclarationVertex(ren, this, vol),
    true);

  vertexShader = vtkmvolume::replace(
    vertexShader,
    "__NUMBER_OF_VOLUMES__",
    std::to_string(this->GetNumberOfInputConnections(0)),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "__NUMBER_OF_VOLUMES__",
    std::to_string(this->GetNumberOfInputConnections(0)),
    true);

  //fragmentShader = vtkmvolume::replace(fragmentShader, "//VTK::CallWorker::Impl",
  //  vtkmvolume::WorkerImplementation(ren, this, vol), true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Base::Dec",
    vtkmvolume::BaseDeclarationFragment(ren, this, vol, this->Impl->NumberOfLights,
                                        this->Impl->LightComplexity,
                                        vol->GetProperty(0)->HasGradientOpacity(),
                                        noOfComponents, independentComponents),
    true);

  /*fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Base::Init",
    vtkmvolume::BaseInit(ren, this, vol, this->Impl->LightComplexity),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Base::Impl",
    vtkmvolume::BaseImplementation(ren, this, vol),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Base::Exit",
    vtkmvolume::BaseExit(ren, this, vol),
    true);
*/
  // Termination methods replacements
  //--------------------------------------------------------------------------
  vertexShader = vtkmvolume::replace(
    vertexShader,
    "//VTK::Termination::Dec",
    vtkmvolume::TerminationDeclarationVertex(ren, this, vol),
    true);
/*  
  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Termination::Dec",
    vtkmvolume::TerminationDeclarationFragment(ren, this, vol),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Terminate::Init",
    vtkmvolume::TerminationInit(ren, this, vol),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Terminate::Impl",
    vtkmvolume::TerminationImplementation(ren, this, vol),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Terminate::Exit",
    vtkmvolume::TerminationExit(ren, this, vol),
    true);

  // Shading methods replacements
  //--------------------------------------------------------------------------
  vertexShader = vtkmvolume::replace(
    vertexShader,
    "//VTK::Shading::Dec",
    vtkmvolume::ShadingDeclarationVertex(ren, this, vol),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Shading::Dec",
    vtkmvolume::ShadingDeclarationFragment(ren, this, vol),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Shading::Init",
    vtkmvolume::ShadingInit(ren, this, vol),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Shading::Impl",
    vtkmvolume::ShadingImplementation(ren, this, vol, this->MaskInput,
                                     this->Impl->CurrentMask,
                                     this->MaskType, noOfComponents,
                                     independentComponents),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Shading::Exit",
    vtkmvolume::ShadingExit(ren, this, vol, noOfComponents,
                           independentComponents),
    true);

	*/
  // Compute methods replacements
  //--------------------------------------------------------------------------
  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::ComputeOpacity::Dec",
    vtkmvolume::ComputeOpacityDeclaration(ren, this, vol, noOfComponents,
                                         independentComponents,
                                         this->Impl->OpacityTablesMap),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::ComputeGradient::Dec",
    vtkmvolume::ComputeGradientDeclaration(ren, this, vol, noOfComponents,
                                          independentComponents,
                                          this->Impl->GradientOpacityTablesMap),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::ComputeColor::Dec",
    vtkmvolume::ComputeColorDeclaration(ren, this, vol, noOfComponents,
                                       independentComponents,
                                       this->Impl->RGBTablesMap),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::ComputeLighting::Dec",
    vtkmvolume::ComputeLightingDeclaration(ren, this, vol, noOfComponents,
                                          independentComponents,
                                          this->Impl->NumberOfLights,
                                          this->Impl->LightComplexity),
    true);
/*
  fragmentShader = vtkmvolume::replace(fragmentShader,
                                      "//VTK::ComputeRayDirection::Dec",
      vtkmvolume::ComputeRayDirectionDeclaration(ren, this, vol,noOfComponents),
    true);

  // Cropping methods replacements
  //--------------------------------------------------------------------------
  /*vertexShader = vtkmvolume::replace(
    vertexShader,
    "//VTK::Cropping::Dec",
    vtkmvolume::CroppingDeclarationVertex(ren, this, vol),
    true);
  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Cropping::Dec",
    vtkmvolume::CroppingDeclarationFragment(ren, this, vol),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Cropping::Init",
    vtkmvolume::CroppingInit(ren, this, vol),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Cropping::Impl",
    vtkmvolume::CroppingImplementation(ren, this, vol),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Cropping::Exit",
    vtkmvolume::CroppingExit(ren, this, vol),
    true);*/

  // Clipping methods replacements
  //--------------------------------------------------------------------------
  /*vertexShader = vtkmvolume::replace(
    vertexShader,
    "//VTK::Clipping::Dec",
    vtkmvolume::ClippingDeclarationVertex(ren, this, vol),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Clipping::Dec",
    vtkmvolume::ClippingDeclarationFragment(ren, this, vol),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Clipping::Init",
    vtkmvolume::ClippingInit(ren, this, vol),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Clipping::Impl",
    vtkmvolume::ClippingImplementation(ren, this, vol),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::Clipping::Exit",
    vtkmvolume::ClippingExit(ren, this, vol),
    true);*/

  // Masking methods replacements
  //--------------------------------------------------------------------------
  /*fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::BinaryMask::Dec",
    vtkmvolume::BinaryMaskDeclaration(ren, this, vol, this->MaskInput,
                                     this->Impl->CurrentMask,
                                     this->MaskType),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::BinaryMask::Impl",
    vtkmvolume::BinaryMaskImplementation(ren, this, vol, this->MaskInput,
                                        this->Impl->CurrentMask,
                                        this->MaskType),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::CompositeMask::Dec",
    vtkmvolume::CompositeMaskDeclarationFragment(
      ren, this, vol, this->MaskInput,
      this->Impl->CurrentMask,
      this->MaskType),
    true);

  fragmentShader = vtkmvolume::replace(
    fragmentShader,
    "//VTK::CompositeMask::Impl",
    vtkmvolume::CompositeMaskImplementation(
      ren, this, vol, this->MaskInput,
      this->Impl->CurrentMask,
      this->MaskType,
      noOfComponents),
    true);*/

  // Picking replacements
  //--------------------------------------------------------------------------
  /*if (this->Impl->CurrentSelectionPass != (vtkHardwareSelector::MIN_KNOWN_PASS - 1))
  {
    switch(this->Impl->CurrentSelectionPass)
    {
      case vtkHardwareSelector::ID_LOW24:
        fragmentShader = vtkmvolume::replace(fragmentShader, "//VTK::Picking::Exit",
          vtkmvolume::PickingIdLow24PassExit(ren, this, vol), true);
        break;
      case vtkHardwareSelector::ID_MID24:
        fragmentShader = vtkmvolume::replace(fragmentShader, "//VTK::Picking::Exit",
          vtkmvolume::PickingIdMid24PassExit(ren, this, vol), true);
        break;
      default: // ACTOR_PASS, PROCESS_PASS
        fragmentShader = vtkmvolume::replace(fragmentShader, "//VTK::Picking::Dec",
          vtkmvolume::PickingActorPassDeclaration(ren, this, vol), true);

        fragmentShader = vtkmvolume::replace(fragmentShader, "//VTK::Picking::Exit",
          vtkmvolume::PickingActorPassExit(ren, this, vol), true);
        break;
    }
  }*/

  // Render to texture
  //--------------------------------------------------------------------------
  /*if (this->RenderToImage)
  {
    fragmentShader = vtkmvolume::replace(
      fragmentShader,
      "//VTK::RenderToImage::Dec",
      vtkmvolume::RenderToImageDeclarationFragment(
        ren, this, vol), true);

    fragmentShader = vtkmvolume::replace(
      fragmentShader,
      "//VTK::RenderToImage::Init",
      vtkmvolume::RenderToImageInit(
        ren, this, vol), true);

    fragmentShader = vtkmvolume::replace(
      fragmentShader,
      "//VTK::RenderToImage::Impl",
      vtkmvolume::RenderToImageImplementation(
        ren, this, vol), true);

    fragmentShader = vtkmvolume::replace(
      fragmentShader,
      "//VTK::RenderToImage::Exit",
      vtkmvolume::RenderToImageExit(
        ren, this, vol), true);
  }*/

  this->ReplaceShaderRenderPass(vertexShader, fragmentShader, vol, false);

  // Now compile the shader
  std::ofstream file;
  file.open("multi.txt");
  file << vertexShader << std::endl;
  file << std::endl;
  file << fragmentShader << std::endl;
  file.close();
  //--------------------------------------------------------------------------
  this->Impl->ShaderProgram = this->Impl->ShaderCache->ReadyShaderProgram(
    vertexShader.c_str(), fragmentShader.c_str(), "");
  if (!this->Impl->ShaderProgram || !this->Impl->ShaderProgram->GetCompiled())
  {
    vtkErrorMacro("Shader failed to compile");
  }

  this->Impl->ShaderBuildTime.Modified();
}

//-----------------------------------------------------------------------------
// Update the reduction factor of the render viewport (this->ReductionFactor)
// according to the time spent in seconds to render the previous frame
// (this->TimeToDraw) and a time in seconds allocated to render the next
// frame (allocatedTime).
// \pre valid_current_reduction_range: this->ReductionFactor>0.0 && this->ReductionFactor<=1.0
// \pre positive_TimeToDraw: this->TimeToDraw>=0.0
// \pre positive_time: allocatedTime>0.0
// \post valid_new_reduction_range: this->ReductionFactor>0.0 && this->ReductionFactor<=1.0
//-----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::ComputeReductionFactor(
  double allocatedTime)
{
  if ( !this->AutoAdjustSampleDistances )
  {
    this->ReductionFactor = 1.0 / this->ImageSampleDistance;
    return;
  }

  if ( this->TimeToDraw )
  {
    double oldFactor = this->ReductionFactor;

    double timeToDraw;
    if (allocatedTime < 1.0)
    {
      timeToDraw = this->SmallTimeToDraw;
      if ( timeToDraw == 0.0 )
      {
        timeToDraw = this->BigTimeToDraw/3.0;
      }
    }
    else
    {
      timeToDraw = this->BigTimeToDraw;
    }

    // This should be the case when rendering the volume very first time
    // 10.0 is an arbitrary value chosen which happen to a large number
    // in this context
    if ( timeToDraw == 0.0 )
    {
      timeToDraw = 10.0;
    }

    double fullTime = timeToDraw / this->ReductionFactor;
    double newFactor = allocatedTime / fullTime;

    // Compute average factor
    this->ReductionFactor = (newFactor + oldFactor)/2.0;

    // Discretize reduction factor so that it doesn't cause
    // visual artifacts when used to reduce the sample distance
    this->ReductionFactor = (this->ReductionFactor > 1.0) ? 1.0 :
                              (this->ReductionFactor);

    if (this->ReductionFactor < 0.20)
    {
      this->ReductionFactor = 0.10;
    }
    else if (this->ReductionFactor < 0.50)
    {
      this->ReductionFactor = 0.20;
    }
    else if (this->ReductionFactor < 1.0)
    {
      this->ReductionFactor = 0.50;
    }

    // Clamp it
    if ( 1.0/this->ReductionFactor > this->MaximumImageSampleDistance )
    {
      this->ReductionFactor = 1.0 / this->MaximumImageSampleDistance;
    }
    if ( 1.0/this->ReductionFactor < this->MinimumImageSampleDistance )
    {
      this->ReductionFactor = 1.0 / this->MinimumImageSampleDistance;
    }
  }
}

//----------------------------------------------------------------------------
bool vtkOpenGLGPUMultiVolumeRayCastMapper::PreLoadData(vtkRenderer* ren,
  vtkMultiVolume2* vol)
{
  if (!this->ValidateRender(ren, vol))
  {
    return false;
  }

  bool success = true;
  for (int i = 0; i < this->GetNumberOfInputConnections(0) && success; ++i) {
	  vtkImageData* input = this->GetTransformedInput(i);
	  vtkVolumeProperty* volProp = vol->GetProperty(i);

	  vtkDataArray* arr = this->GetScalars(input, this->ScalarMode,
		  this->ArrayAccessMode, this->ArrayId, this->ArrayName, this->CellFlag[i]);

	  success = this->Impl->LoadData(i, ren, vol, volProp, input, arr);
  }

  return success;
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::GPURender(vtkRenderer* ren,
                                                     vtkMultiVolume2* vol)
{
  vtkOpenGLClearErrorMacro();

  this->ResourceCallback->RegisterGraphicsResources(
    static_cast<vtkOpenGLRenderWindow *>(ren->GetRenderWindow()));

  //this->Impl->TempMatrix1->Identity();
  vtkNew<vtkMatrix4x4> tempMatrix;

  // Make sure the context is current
  vtkOpenGLRenderWindow* renWin =
    vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow());
  renWin->MakeCurrent();

  // Update in_volume first to make sure states are current
  vol->Update();

  // Get the camera
  vtkOpenGLCamera* cam = vtkOpenGLCamera::SafeDownCast(ren->GetActiveCamera());

  // Get window size and corners
  if (!this->Impl->PreserveViewport)
  {
	  ren->GetTiledSizeAndOrigin(
		  this->Impl->WindowSize, this->Impl->WindowSize + 1,
		  this->Impl->WindowLowerLeft, this->Impl->WindowLowerLeft + 1);
  }
  else
  {
	  int vp[4];
	  glGetIntegerv(GL_VIEWPORT, vp);
	  this->Impl->WindowLowerLeft[0] = vp[0];
	  this->Impl->WindowLowerLeft[1] = vp[1];
	  this->Impl->WindowSize[0] = vp[2];
	  this->Impl->WindowSize[1] = vp[3];
  }

  for (int i = 0; i < this->GetNumberOfInputConnections(0); ++i)
  {
	  this->Impl->NeedToInitializeResources[i] =
		(this->Impl->ReleaseResourcesTime.GetMTime() >
		this->Impl->InitializationTime[i].GetMTime());
	  // Get the input
	  vtkImageData* input = this->GetTransformedInput(i);

	  // Get the volume property (must have one)
	  vtkVolumeProperty* volumeProperty = vol->GetProperty(i);

	  // Check whether we have independent components or not
	  int independentComponents = volumeProperty->GetIndependentComponents();

	  this->Impl->CheckPropertyKeys(vol);

	  vtkDataArray* scalars = this->GetScalars(input,
		  this->ScalarMode,
		  this->ArrayAccessMode,
		  this->ArrayId,
		  this->ArrayName,
		  this->CellFlag[i]);

	  // How many components are there?
	  int noOfComponents = scalars->GetNumberOfComponents();

	  // Allocate important variables
	  //this->Impl->Bias.resize(noOfComponents, 0.0);

	  if (this->Impl->NeedToInitializeResources[i] ||
		  (volumeProperty->GetMTime() >
			  this->Impl->InitializationTime[i].GetMTime()))
	  {
		  this->Impl->Initialize(i, ren, vol, noOfComponents,
			  independentComponents);
	  }

	  // Three dependent components are not supported
	  if ((noOfComponents == 3) && !independentComponents)
	  {
		  vtkErrorMacro("Three dependent components are not supported");
	  }

	  // Update the volume if needed
	  if (this->Impl->NeedToInitializeResources[i] ||
		  (input->GetMTime() > this->Impl->InputUpdateTime[i].GetMTime()))
	  {
		  this->Impl->LoadData(i, ren, vol, volumeProperty, input, scalars);
	  }
	  else
	  {
		  //this->Impl->LoadMask(ren, input, this->MaskInput, this->Impl->Extents, vol);
		  this->Impl->UpdateVolume(i, volumeProperty);
	  }
  }

  // Mask
  vtkVolumeMask* mask = 0;
  if(this->MaskInput != 0)
  {
    std::map<vtkImageData *,vtkVolumeMask*>::iterator it2 =
      this->Impl->MaskTextures->Map.find(this->MaskInput);
    if(it2 == this->Impl->MaskTextures->Map.end())
    {
      mask=0;
    }
    else
    {
      mask=(*it2).second;
    }
  }
  this->Impl->CurrentMask = mask;

  this->ComputeReductionFactor(vol->GetAllocatedRenderTime());

  vtkImageData* input0 = this->GetTransformedInput(0);
  vtkDataArray* scalars0 = this->GetScalars(input0,
		  this->ScalarMode,
		  this->ArrayAccessMode,
		  this->ArrayId,
		  this->ArrayName,
		  this->CellFlag[0]);

  for (int in = 0; in < this->GetNumberOfInputConnections(0); ++in) {
	  // Get the input
	  vtkImageData* input = this->GetTransformedInput(in);

	  // Get the volume property (must have one)
	  vtkVolumeProperty* volumeProperty = vol->GetProperty(in);

	  // Check whether we have independent components or not
	  int independentComponents = volumeProperty->GetIndependentComponents();

	  this->Impl->UpdateSamplingDistance(in, input, ren, vol);

	  vtkDataArray* scalars = this->GetScalars(input,
		  this->ScalarMode,
		  this->ArrayAccessMode,
		  this->ArrayId,
		  this->ArrayName,
		  this->CellFlag[in]);

	  // How many components are there?
	  int noOfComponents = scalars->GetNumberOfComponents();

	  // Update the transfer functions
	if (independentComponents) {
		for (int i = 0; i < noOfComponents; ++i) {
			this->Impl->UpdateOpacityTransferFunction(in, ren, vol, i);
				if (vol->GetProperty(in)->HasGradientOpacity(i))
					this->Impl->UpdateGradientOpacityTransferFunction(in, ren, vol, i);
			this->Impl->UpdateColorTransferFunction(in, ren, vol, i);
		}
	}
  }
  /*else
  {
    if (noOfComponents == 2 || noOfComponents == 4)
    {
      this->Impl->UpdateOpacityTransferFunction(ren, vol, noOfComponents - 1);
      this->Impl->UpdateGradientOpacityTransferFunction(ren, vol,
                                                        noOfComponents - 1);
      this->Impl->UpdateColorTransferFunction(ren, vol, 0);
    }
  }*/

  // Update noise sampler texture
  if (this->UseJittering)
  {
    this->Impl->CreateNoiseTexture(ren);
  }

  // Grab depth sampler buffer (to handle cases when we are rendering geometry
  // and in_volume together
  this->Impl->CaptureDepthTexture(ren, vol);

  this->Impl->ShaderCache = vtkOpenGLRenderWindow::SafeDownCast(
    ren->GetRenderWindow())->GetShaderCache();

  this->Impl->CheckPickingState(ren);

  vtkMTimeType renderPassTime = this->GetRenderPassStageMTime(vol);

  bool needToInitializeResources = false;
  for (auto& n : this->Impl->NeedToInitializeResources)
    needToInitializeResources |= n.second;
  if (this->UseDepthPass && this->GetBlendMode() ==
      vtkMultiVolumeMapper::COMPOSITE_BLEND)
  {
    this->CurrentPass = DepthPass;

	double propMTime = 0.;
	for (int i = 0; i < this->GetNumberOfInputConnections(0); ++i)
		propMTime = vol->GetProperty(i)->GetMTime() > propMTime ? vol->GetProperty(i)->GetMTime() : propMTime;

    if (needToInitializeResources ||
        propMTime > this->Impl->DepthPassSetupTime.GetMTime() ||
        this->GetMTime() > this->Impl->DepthPassSetupTime.GetMTime() ||
        cam->GetParallelProjection() != this->Impl->LastProjectionParallel ||
        this->Impl->SelectionStateTime.GetMTime() > this->Impl->ShaderBuildTime.GetMTime() ||
        renderPassTime > this->Impl->ShaderBuildTime)
    {
      this->Impl->LastProjectionParallel =
        cam->GetParallelProjection();

      this->Impl->ContourFilter->SetInputData(input0);
      for (int i = 0; i < this->GetDepthPassContourValues()->GetNumberOfContours(); ++i)
      {
        this->Impl->ContourFilter->SetValue(i,
          this->DepthPassContourValues->GetValue(i));
      }

      vtkNew<vtkMatrix4x4> newMatrix;
      newMatrix->DeepCopy(vol->GetMatrix());

      this->Impl->SetupDepthPass(ren);

      this->Impl->ContourActor->Render(ren,
                                       this->Impl->ContourMapper.GetPointer());

      this->Impl->ExitDepthPass(ren);

      this->Impl->DepthPassSetupTime.Modified();
      this->Impl->DepthPassTime.Modified();

      this->CurrentPass = RenderPass;
      this->BuildShader(ren, vol, scalars0->GetNumberOfComponents());
    }
    else if (cam->GetMTime() > this->Impl->DepthPassTime.GetMTime())
    {
      this->Impl->SetupDepthPass(ren);

      this->Impl->ContourActor->Render(ren,
                                       this->Impl->ContourMapper.GetPointer());

      this->Impl->ExitDepthPass(ren);
      this->Impl->DepthPassTime.Modified();

      this->CurrentPass = RenderPass;
    }

    // Configure picking begin (changes blending, so needs to be called before
    // vtkVolumeStateRAII)
    if (this->Impl->IsPicking)
    {
      this->Impl->BeginPicking(ren);
    }

    // Set OpenGL states
    // TODO vtkVolumeStateRAII glState(this->Impl->PreserveGLState);

    if (this->RenderToImage)
    {
      this->Impl->SetupRenderToTexture(ren);
    }

    if (!this->Impl->PreserveViewport)
    {
      // NOTE: This is a must call or else, multiple viewport
      // rendering would not work. We need this primarily because
      // FBO set it otherwise.
      // TODO The viewport should not be set within the mapper,
      // causes issues when vtkOpenGLRenderPass instances modify it too.
      glViewport(this->Impl->WindowLowerLeft[0],
                 this->Impl->WindowLowerLeft[1],
                 this->Impl->WindowSize[0],
                 this->Impl->WindowSize[1]);
    }

    renWin->GetShaderCache()->ReadyShaderProgram(this->Impl->ShaderProgram);

    this->Impl->DPDepthBufferTextureObject->Activate();
    this->Impl->ShaderProgram->SetUniformi("in_depthPassSampler",
      this->Impl->DPDepthBufferTextureObject->GetTextureUnit());

	  // Get the volume property (must have one)
	  vtkVolumeProperty* volumeProperty = vol->GetProperty(0);

	  // Check whether we have independent components or not
	  int independentComponents = volumeProperty->GetIndependentComponents();

    this->DoGPURender(ren, vol, cam, this->Impl->ShaderProgram,
                      scalars0->GetNumberOfComponents(), independentComponents);

    this->Impl->DPDepthBufferTextureObject->Deactivate();
  }
  else
  {
    // Configure picking begin (changes blending, so needs to be called before
    // vtkVolumeStateRAII)
    if (this->Impl->IsPicking)
    {
      this->Impl->BeginPicking(ren);
    }
    // Set OpenGL states
    // TODO vtkVolumeStateRAII glState(this->Impl->PreserveGLState);

	double propMTime = 0.;
	for (int i = 0; i < this->GetNumberOfInputConnections(0); ++i)
		propMTime = vol->GetProperty(i)->GetMTime() > propMTime ? vol->GetProperty(i)->GetMTime() : propMTime;

    // Build shader now
    // First get the shader cache from the render window. This is important
    // to make sure that shader cache knows the state of various shader programs
    // in use.
    if (needToInitializeResources ||
        propMTime > this->Impl->ShaderBuildTime.GetMTime() ||
        this->GetMTime() > this->Impl->ShaderBuildTime.GetMTime() ||
        cam->GetParallelProjection() != this->Impl->LastProjectionParallel ||
        this->Impl->SelectionStateTime.GetMTime() > this->Impl->ShaderBuildTime.GetMTime() ||
        renderPassTime > this->Impl->ShaderBuildTime)
    {
      this->Impl->LastProjectionParallel =
        cam->GetParallelProjection();
      this->BuildShader(ren, vol, scalars0->GetNumberOfComponents());
    }
    else
    {
      // Bind the shader
      this->Impl->ShaderCache->ReadyShaderProgram(
        this->Impl->ShaderProgram);
    }

    if (this->RenderToImage)
    {
      this->Impl->SetupRenderToTexture(ren);


      this->DoGPURender(ren, vol, cam, this->Impl->ShaderProgram,
                         scalars0->GetNumberOfComponents(), vol->GetProperty(0)->GetIndependentComponents());

      this->Impl->ExitRenderToTexture(ren);
    }
    else
    {
      this->Impl->BeginImageSample(ren, vol);
      this->DoGPURender(ren, vol, cam, this->Impl->ShaderProgram,
                         scalars0->GetNumberOfComponents(), vol->GetProperty(0)->GetIndependentComponents());
      this->Impl->EndImageSample(ren);
    }
  }

  // Configure picking end
  /*if (this->Impl->IsPicking)
  {
    this->Impl->EndPicking(ren);
  }*/

  glFinish();
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::DoGPURender(vtkRenderer* ren,
                                                  vtkMultiVolume2* vol,
                                                  vtkOpenGLCamera* cam,
                                                  vtkShaderProgram* prog,
                                                  int noOfComponents,
                                                  int independentComponents)
{
  // if the shader didn't compile return
  if (!prog)
  {
    return;
  }

  this->SetShaderParametersRenderPass(vol);

  // Sort blocks in case the viewpoint changed, it immediately returns if there
  // is a single block.

  prog->SetUniformi("in_clampDepthToBackface", this->GetClampDepthToBackface());

  // Bind cropping
  //--------------------------------------------------------------------------
  this->Impl->SetCroppingRegions(ren, prog, vol);

  // Bind clipping
  //--------------------------------------------------------------------------
  this->Impl->SetClippingPlanes(ren, prog, vol);

  // Bind the prop Id
  //--------------------------------------------------------------------------
  /*if (this->Impl->CurrentSelectionPass < vtkHardwareSelector::ID_LOW24)
  {
	  this->Impl->SetPickingId(ren);
  }*/

  // Set the scalar range to be considered for average ip blend
  //--------------------------------------------------------------------------
  double avgRange[2];
  this->GetAverageIPScalarRange(avgRange);
  if (avgRange[1] < avgRange[0])
  {
	  double tmp = avgRange[1];
	  avgRange[1] = avgRange[0];
	  avgRange[0] = tmp;
  }

  // Temporary variables
  float fvalue2[2];
  float fvalue4[4];
  vtkInternal::ToFloat(avgRange[0], avgRange[1], fvalue2);
  prog->SetUniform2fv("in_averageIPRange", 1, &fvalue2);

  // Opacity, color, and gradient opacity samplers / textures
  int numberOfSamplers = (independentComponents ? noOfComponents : 1);

  // Activate/bind DepthTextureObject to a texture unit first as it was already
  // activated in CaptureDepthTexture. Certain APPLE implementations seem to be
  // sensitive to swaping the activation order (causing GL_INVALID_OPERATION after
  // the glDraw call).
#if GL_ES_VERSION_3_0 != 1
  // currently broken on ES
  this->Impl->DepthTextureObject->Activate();
  prog->SetUniformi("in_depthSampler", this->Impl->DepthTextureObject->GetTextureUnit());
#endif

  /*if (this->Impl->NoiseTextureObject)
  {
	  this->Impl->NoiseTextureObject->Activate();
	  prog->SetUniformi("in_noiseSampler",
		  this->Impl->NoiseTextureObject->GetTextureUnit());
  }

  if (this->Impl->CurrentMask)
  {
	  this->Impl->CurrentMask->Activate();
	  prog->SetUniformi(
		  "in_mask", this->Impl->CurrentMask->GetTextureUnit());
  }

  if (noOfComponents == 1 &&
	  this->BlendMode != vtkGPUMultiVolumeRayCastMapper::ADDITIVE_BLEND)
  {
	  if (this->MaskInput != 0 && this->MaskType == LabelMapMaskType)
	  {
		  this->Impl->Mask1RGBTable->Activate();
		  prog->SetUniformi("in_mask1",
			  this->Impl->Mask1RGBTable->GetTextureUnit());

		  this->Impl->Mask2RGBTable->Activate();
		  prog->SetUniformi("in_mask2", this->Impl->Mask2RGBTable->GetTextureUnit());
		  prog->SetUniformf("in_maskBlendFactor", this->MaskBlendFactor);
	  }
  }*/

  // Set the scale and bias for color correction
  //--------------------------------------------------------------------------
  prog->SetUniformf("in_scale", 1.0 / this->FinalColorWindow);
  prog->SetUniformf("in_bias", (0.5 - (this->FinalColorLevel / this->FinalColorWindow)));

  float fvalue3[3];
  vtkInternal::ToFloat(cam->GetPosition(), fvalue3, 3);
  prog->SetUniform3fv("in_cameraPos", 1, &fvalue3);

  vtkMatrix4x4* glTransformMatrix;
  vtkMatrix4x4* modelViewMatrix;
  vtkMatrix3x3* normalMatrix;
  vtkMatrix4x4* projectionMatrix;
  cam->GetKeyMatrices(ren, modelViewMatrix, normalMatrix, projectionMatrix, glTransformMatrix);

  vtkNew<vtkMatrix4x4> inverseProjectionMat;
  inverseProjectionMat->DeepCopy(projectionMatrix);
  inverseProjectionMat->Invert();
  prog->SetUniformMatrix("in_projectionMatrix", projectionMatrix);
  prog->SetUniformMatrix("in_inverseProjectionMatrix",
	  inverseProjectionMat.GetPointer());

  vtkNew<vtkMatrix4x4> inverseModelViewMat;
  inverseModelViewMat->DeepCopy(modelViewMatrix);
  inverseModelViewMat->Invert();
  prog->SetUniformMatrix("in_modelViewMatrix", modelViewMatrix);
  prog->SetUniformMatrix("in_inverseModelViewMatrix", inverseModelViewMat.GetPointer());

  // Bind light properties
  //--------------------------------------------------------------------------
  this->Impl->SetLightingParameters(ren, prog, vol);

  // Bind other misc parameters
  //--------------------------------------------------------------------------
  if (cam->GetParallelProjection())
  {
	  double dir[4];
	  cam->GetDirectionOfProjection(dir);
	  vtkInternal::ToFloat(dir[0], dir[1], dir[2], fvalue3);
	  prog->SetUniform3fv("in_projectionDirection", 1, &fvalue3);
  }

  // Pass constant uniforms at initialization
  prog->SetUniformi("in_noOfComponents", noOfComponents);
  prog->SetUniformi("in_independentComponents", independentComponents);

  // TODO Take consideration of reduction factor
  vtkInternal::ToFloat(this->Impl->WindowLowerLeft, fvalue2);
  prog->SetUniform2fv("in_windowLowerLeftCorner", 1, &fvalue2);

  vtkInternal::ToFloat(1.0 / this->Impl->WindowSize[0],
	  1.0 / this->Impl->WindowSize[1], fvalue2);
  prog->SetUniform2fv("in_inverseOriginalWindowSize", 1, &fvalue2);

  vtkInternal::ToFloat(1.0 / this->Impl->WindowSize[0],
	  1.0 / this->Impl->WindowSize[1], fvalue2);
  prog->SetUniform2fv("in_inverseWindowSize", 1, &fvalue2);

  prog->SetUniformi("in_useJittering", this->GetUseJittering());

  // Load up volumes.
  //--------------------------------------------------------------------------
  for (int in = 0; in < this->GetNumberOfInputConnections(0); ++in) {
	  const std::string index = std::to_string(in);
	  // Cell spacing is required to be computed globally (full volume extents)
	  // given that gradients are computed globally (not per block).
	  vtkInternal::ToFloat(this->Impl->CellSpacing[in][0],
		                   this->Impl->CellSpacing[in][1],
	                       this->Impl->CellSpacing[in][2],
		                   fvalue3);
	  prog->SetUniform3fv(("in_cellSpacing[" + index + "]").c_str(), 1, &fvalue3);

	  this->GetVolumeTexture(in)->SortBlocksBackToFront(ren, vol->GetVolumeMatrix(in));

	  vtkMultiVolumeTexture::VolumeBlock* block = this->GetVolumeTexture(in)->GetNextBlock();

	  //while (block != NULL)
	  {
		  this->Impl->ComputeBounds(in, block->ImageData);

		  // Cell step/scale are adjusted per block.
		  // Step should be dependent on the bounds and not on the texture size
		  // since we can have a non-uniform voxel size / spacing / aspect ratio.
		  double cellStep[3];
		  cellStep[0] = 1. / static_cast<double>(this->Impl->Extents[in][1] - this->Impl->Extents[in][0]);
		  cellStep[1] = 1. / static_cast<double>(this->Impl->Extents[in][3] - this->Impl->Extents[in][2]);
		  cellStep[2] = 1. / static_cast<double>(this->Impl->Extents[in][5] - this->Impl->Extents[in][4]);

		  vtkInternal::ToFloat(cellStep, fvalue3);
		  prog->SetUniform3fv(("in_cellStep[" + index + "]").c_str(), 1, &fvalue3);

		  /*double cellScale[3];
		  cellScale[0] = (this->Impl->LoadedBounds[in][1] -
			  this->Impl->LoadedBounds[in][0]) * 0.5;
		  cellScale[1] = (this->Impl->LoadedBounds[in][3] -
			  this->Impl->LoadedBounds[in][2]) * 0.5;
		  cellScale[2] = (this->Impl->LoadedBounds[in][5] -
			  this->Impl->LoadedBounds[in][4]) * 0.5;

		  vtkInternal::ToFloat(cellScale, fvalue3);
		  prog->SetUniform3fv(("in_cellScale[" + index + "]").c_str(), 1, &fvalue3);*/

		  // Update sampling distance
		  double datasetStepSize[3] = {0.};
		  datasetStepSize[0] = 1.0 / (this->Impl->LoadedBounds[in][1] - this->Impl->LoadedBounds[in][0]);
		  datasetStepSize[1] = 1.0 / (this->Impl->LoadedBounds[in][3] - this->Impl->LoadedBounds[in][2]);
		  datasetStepSize[2] = 1.0 / (this->Impl->LoadedBounds[in][5] - this->Impl->LoadedBounds[in][4]);

		  // Compute texture to dataset matrix
		  vtkNew<vtkMatrix4x4> textureToDataSetMat;
		  textureToDataSetMat->Identity();
		  textureToDataSetMat->SetElement(0, 0, (1.0 / datasetStepSize[0]));
		  textureToDataSetMat->SetElement(1, 1, (1.0 / datasetStepSize[1]));
		  textureToDataSetMat->SetElement(2, 2, (1.0 / datasetStepSize[2]));
		  textureToDataSetMat->SetElement(3, 3, 1.0);
		  textureToDataSetMat->SetElement(0, 3, this->Impl->LoadedBounds[in][0]);
		  textureToDataSetMat->SetElement(1, 3, this->Impl->LoadedBounds[in][2]);
		  textureToDataSetMat->SetElement(2, 3, this->Impl->LoadedBounds[in][4]);

		  // Bind current volume texture
		  block->TextureObject->Activate();
		  prog->SetUniformi(("in_volume[" + index + "]").c_str(),
		                    block->TextureObject->GetTextureUnit());

		  vtkVolumeProperty* volumeProperty = vol->GetProperty(in);

		  // Bind textures
		  //--------------------------------------------------------------------------

		  for (int i = 0; i < numberOfSamplers; ++i)
		  {
			  this->Impl->OpacityTables[in]->GetTable(i)->Activate();
			  prog->SetUniformi(this->Impl->OpacityTablesMap[in][i].c_str(),
			                    this->Impl->OpacityTables[in]->GetTable(i)->GetTextureUnit());

			  if (this->BlendMode != vtkGPUMultiVolumeRayCastMapper::ADDITIVE_BLEND)
			  {
				  this->Impl->RGBTables[in]->GetTable(i)->Activate();
				  prog->SetUniformi(this->Impl->RGBTablesMap[in][i].c_str(),
					                this->Impl->RGBTables[in]->GetTable(i)->GetTextureUnit());
			  }

			  if (vol->GetProperty(in)->HasGradientOpacity(i) && this->Impl->GradientOpacityTables[in])
			  {
				  this->Impl->GradientOpacityTables[in]->GetTable(i)->Activate();
				  prog->SetUniformi(this->Impl->GradientOpacityTablesMap[in][i].c_str(),
					                this->Impl->GradientOpacityTables[in]->GetTable(i)->GetTextureUnit());
			  }
		  }

		  // Bind material properties
		  //--------------------------------------------------------------------------
		  float ambient[4][3];
		  float diffuse[4][3];
		  float specular[4][3];
		  float specularPower[4];

		  for (int i = 0; i < numberOfSamplers; ++i)
		  {
			  ambient[i][0] = ambient[i][1] = ambient[i][2] = volumeProperty->GetAmbient(i);
			  diffuse[i][0] = diffuse[i][1] = diffuse[i][2] = volumeProperty->GetDiffuse(i);
			  specular[i][0] = specular[i][1] = specular[i][2] = volumeProperty->GetSpecular(i);
			  specularPower[i] = volumeProperty->GetSpecularPower(i);
		  }

		  prog->SetUniform3fv(("in_ambient[" + index + "].data").c_str(), numberOfSamplers, ambient);
		  prog->SetUniform3fv(("in_diffuse[" + index + "].data").c_str(), numberOfSamplers, diffuse);
		  prog->SetUniform3fv(("in_specular[" + index + "].data").c_str(), numberOfSamplers, specular);
		  prog->SetUniform1fv(("in_shininess[" + index + "].data").c_str(), numberOfSamplers, specularPower);

		  // Bind matrices
		  //--------------------------------------------------------------------------
		  vtkNew<vtkMatrix4x4> tempMatrix;
		  tempMatrix->DeepCopy(vol->GetVolumeMatrix(in));
		  tempMatrix->Transpose();
		  vtkNew<vtkMatrix4x4> inverseVolumeMat;
		  inverseVolumeMat->DeepCopy(tempMatrix.GetPointer());
		  inverseVolumeMat->Invert();

		  prog->SetUniformMatrix(("in_volumeMatrix[" + index + "]").c_str(), tempMatrix.GetPointer());
		  prog->SetUniformMatrix(("in_inverseVolumeMatrix[" + index + "]").c_str(), inverseVolumeMat.GetPointer());

		  vtkNew<vtkMatrix4x4> textureToEyeTransposeInverse;
		  tempMatrix->DeepCopy(textureToDataSetMat.GetPointer());
		  vtkMatrix4x4::Multiply4x4(vol->GetVolumeMatrix(in),
			  tempMatrix.GetPointer(),
			  textureToEyeTransposeInverse.GetPointer());

		  vtkMatrix4x4::Multiply4x4(modelViewMatrix,
			  textureToEyeTransposeInverse.GetPointer(),
			  textureToEyeTransposeInverse.GetPointer());

		  tempMatrix->Transpose();
		  vtkNew<vtkMatrix4x4> inverseTextureToDataSetMat;
		  inverseTextureToDataSetMat->DeepCopy(tempMatrix.GetPointer());
		  inverseTextureToDataSetMat->Invert();

		  prog->SetUniformMatrix(("in_textureDatasetMatrix[" + index + "]").c_str(), tempMatrix.GetPointer());
		  prog->SetUniformMatrix(("in_inverseTextureDatasetMatrix[" + index + "]").c_str(), inverseTextureToDataSetMat.GetPointer());
		  prog->SetUniformMatrix(("in_textureToEye[" + index + "]").c_str(), textureToEyeTransposeInverse.GetPointer());

		  // LargeDataTypes have been already biased and scaled so in those cases 0s
		  // and 1s are passed respectively.
		  float tscale[4] = { 1.0, 1.0, 1.0, 1.0 };
		  float tbias[4] = { 0.0, 0.0, 0.0, 0.0 };
		  float(*scalePtr)[4] = &tscale;
		  float(*biasPtr)[4] = &tbias;
		  if (!this->GetVolumeTexture(in)->HandleLargeDataTypes &&
			  (noOfComponents == 1 || noOfComponents == 2 || independentComponents))
		  {
			  scalePtr = &this->GetVolumeTexture(in)->Scale;
			  biasPtr = &this->GetVolumeTexture(in)->Bias;
		  }
		  prog->SetUniform4fv(("in_volume_scale[" + index + "]").c_str(), 1, scalePtr);
		  prog->SetUniform4fv(("in_volume_bias[" + index + "]").c_str(), 1, biasPtr);

		  prog->SetUniformf(("in_sampleDistance[" + index + "]").c_str(), this->Impl->ActualSampleDistance[in]);

		  float scalarsRange[4][2];
		  vtkInternal::ToFloat(this->GetVolumeTexture(in)->ScalarRange, scalarsRange);
		  prog->SetUniform2fv(("in_scalarsRange[" + index + "].data").c_str(), 4, scalarsRange);

		  vtkInternal::ToFloat(this->Impl->LoadedBounds[in][0],
			                   this->Impl->LoadedBounds[in][2],
			                   this->Impl->LoadedBounds[in][4],
		                       fvalue3);
		  prog->SetUniform3fv(("in_volumeExtentsMin[" + index + "]").c_str(), 1, &fvalue3);

		  vtkInternal::ToFloat(this->Impl->LoadedBounds[in][1],
			  this->Impl->LoadedBounds[in][3],
			  this->Impl->LoadedBounds[in][5], fvalue3);
		  prog->SetUniform3fv(("in_volumeExtentsMax[" + index + "]").c_str(), 1, &fvalue3);

		  vtkInternal::ToFloat(this->Impl->Extents[in][0], this->Impl->Extents[in][2], this->Impl->Extents[in][4], fvalue3);
		  prog->SetUniform3fv(("in_textureExtentsMin[" + index + "]").c_str(), 1, &fvalue3);

		  vtkInternal::ToFloat(this->Impl->Extents[in][1], this->Impl->Extents[in][3], this->Impl->Extents[in][5], fvalue3);
		  prog->SetUniform3fv(("in_textureExtentsMax[" + index + "]").c_str(), 1, &fvalue3);

		  prog->SetUniformi(("in_cellFlag[" + index + "]").c_str(), this->CellFlag[in]);

		  vtkInternal::ToFloat(this->Impl->AdjustedTexMin[in][0],
			  this->Impl->AdjustedTexMin[in][1],
			  this->Impl->AdjustedTexMin[in][2], fvalue3);
		  prog->SetUniform3fv(("in_texMin[" + index + "]").c_str(), 1, &fvalue3);

		  vtkInternal::ToFloat(this->Impl->AdjustedTexMax[in][0],
			  this->Impl->AdjustedTexMax[in][1],
			  this->Impl->AdjustedTexMax[in][2], fvalue3);
		  prog->SetUniform3fv(("in_texMax[" + index + "]").c_str(), 1, &fvalue3);

		  tempMatrix->DeepCopy(this->Impl->CellToPointMatrix[in].GetPointer());
		  tempMatrix->Transpose();
		  prog->SetUniformMatrix(("in_cellToPoint[" + index + "]").c_str(), tempMatrix.GetPointer());

		  if (noOfComponents > 1 && independentComponents)
		  {
			  for (int i = 0; i < noOfComponents; ++i)
			  {
				  fvalue4[i] = static_cast<float>(volumeProperty->GetComponentWeight(i));
			  }
			  prog->SetUniform4fv(("in_componentWeight[" + index + "]").c_str(), 1, &fvalue4);
		  }

		  // Update next block to render
		  //---------------------------------------------------------------------------
		  block = this->GetVolumeTexture(in)->GetNextBlock();
	  }
  }

  // Render volume geometry to trigger render
  //--------------------------------------------------------------------------
  this->Impl->RenderVolumeGeometry(ren, prog, vol);

  // Undo binds and de-activate buffers
  //--------------------------------------------------------------------------
  //block->TextureObject->Deactivate();
  if (this->Impl->NoiseTextureObject)
  {
	  this->Impl->NoiseTextureObject->Deactivate();
  }
#if GL_ES_VERSION_3_0 != 1
  if (this->Impl->DepthTextureObject)
  {
	  this->Impl->DepthTextureObject->Deactivate();
  }
#endif

  for (int in = 0; in < this->GetNumberOfInputConnections(0); ++in)
	  for (int i = 0; i < numberOfSamplers; ++i)
	  {
		  this->Impl->OpacityTables[in]->GetTable(i)->Deactivate();
		  if (this->BlendMode != vtkGPUMultiVolumeRayCastMapper::ADDITIVE_BLEND)
		  {
			  this->Impl->RGBTables[in]->GetTable(i)->Deactivate();
		  }
		  if (vol->GetProperty(in)->HasGradientOpacity(i) && this->Impl->GradientOpacityTables[in])
		  {
			  this->Impl->GradientOpacityTables[in]->GetTable(i)->Deactivate();
		  }
	  }

  if (this->Impl->CurrentMask)
  {
	  this->Impl->CurrentMask->Deactivate();
  }

  if (noOfComponents == 1 &&
	  this->BlendMode != vtkGPUMultiVolumeRayCastMapper::ADDITIVE_BLEND)
  {
	  if (this->MaskInput != 0 && this->MaskType == LabelMapMaskType)
	  {
		  this->Impl->Mask1RGBTable->Deactivate();
		  this->Impl->Mask2RGBTable->Deactivate();
	  }
  }

  vtkOpenGLCheckErrorMacro("failed after Render");
}

//----------------------------------------------------------------------------
vtkCxxSetObjectMacro(vtkOpenGLGPUMultiVolumeRayCastMapper, NoiseGenerator,
  vtkImplicitFunction);

//-----------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::SetPartitions(int i, unsigned short x,
  unsigned short y, unsigned short z)
{
  this->GetVolumeTexture(i)->SetPartitions(x, y, z);
}

//-----------------------------------------------------------------------------
vtkMTimeType vtkOpenGLGPUMultiVolumeRayCastMapper::GetRenderPassStageMTime(vtkMultiVolume2* vol)
{
  vtkInformation *info = vol->GetPropertyKeys();
  vtkMTimeType renderPassMTime = 0;

  int curRenderPasses = 0;
  this->Impl->RenderPassAttached = false;
  if (info && info->Has(vtkOpenGLRenderPass::RenderPasses()))
  {
    curRenderPasses = info->Length(vtkOpenGLRenderPass::RenderPasses());
    this->Impl->RenderPassAttached = true;
  }

  int lastRenderPasses = 0;
  if (this->LastRenderPassInfo->Has(vtkOpenGLRenderPass::RenderPasses()))
  {
    lastRenderPasses =
        this->LastRenderPassInfo->Length(vtkOpenGLRenderPass::RenderPasses());
  }

  // Determine the last time a render pass changed stages:
  if (curRenderPasses != lastRenderPasses)
  {
    // Number of passes changed, definitely need to update.
    // Fake the time to force an update:
    renderPassMTime = VTK_MTIME_MAX;
  }
  else
  {
    // Compare the current to the previous render passes:
    for (int i = 0; i < curRenderPasses; ++i)
    {
      vtkObjectBase *curRP = info->Get(vtkOpenGLRenderPass::RenderPasses(), i);
      vtkObjectBase *lastRP =
          this->LastRenderPassInfo->Get(vtkOpenGLRenderPass::RenderPasses(), i);

      if (curRP != lastRP)
      {
        // Render passes have changed. Force update:
        renderPassMTime = VTK_MTIME_MAX;
        break;
      }
      else
      {
        // Render passes have not changed -- check MTime.
        vtkOpenGLRenderPass *rp = static_cast<vtkOpenGLRenderPass*>(curRP);
        renderPassMTime = std::max(renderPassMTime, rp->GetShaderStageMTime());
      }
    }
  }

  // Cache the current set of render passes for next time:
  if (info)
  {
    this->LastRenderPassInfo->CopyEntry(info,
                                        vtkOpenGLRenderPass::RenderPasses());
  }
  else
  {
    this->LastRenderPassInfo->Clear();
  }

  return renderPassMTime;
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::ReplaceShaderRenderPass(std::string& vertShader,
  std::string& fragShader, vtkMultiVolume2* vol, bool prePass)
{
  std::string geomShader; // Currently unused
  vtkInformation* info = vol->GetPropertyKeys();
  if (info && info->Has(vtkOpenGLRenderPass::RenderPasses()))
  {
    int numRenderPasses = info->Length(vtkOpenGLRenderPass::RenderPasses());
    for (int i = 0; i < numRenderPasses; ++i)
    {
      vtkObjectBase *rpBase = info->Get(vtkOpenGLRenderPass::RenderPasses(), i);
      vtkOpenGLRenderPass *rp = static_cast<vtkOpenGLRenderPass*>(rpBase);
      if (prePass)
      {
        if (!rp->PreReplaceShaderValues(vertShader, geomShader, fragShader,
          this, vol))
        {
          vtkErrorMacro("vtkOpenGLRenderPass::PreReplaceShaderValues failed for "
            << rp->GetClassName());
        }
      }
      else
      {
        if (!rp->PostReplaceShaderValues(vertShader, geomShader, fragShader,
          this, vol))
        {
          vtkErrorMacro("vtkOpenGLRenderPass::PostReplaceShaderValues failed for "
            << rp->GetClassName());
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUMultiVolumeRayCastMapper::SetShaderParametersRenderPass(
  vtkMultiVolume2* vol)
{
  vtkInformation *info = vol->GetPropertyKeys();
  if (info && info->Has(vtkOpenGLRenderPass::RenderPasses()))
  {
    int numRenderPasses = info->Length(vtkOpenGLRenderPass::RenderPasses());
    for (int i = 0; i < numRenderPasses; ++i)
    {
      vtkObjectBase *rpBase = info->Get(vtkOpenGLRenderPass::RenderPasses(), i);
      vtkOpenGLRenderPass *rp = static_cast<vtkOpenGLRenderPass*>(rpBase);
      if (!rp->SetShaderParameters(this->Impl->ShaderProgram, this, vol))
      {
        vtkErrorMacro("RenderPass::SetShaderParameters failed for renderpass: "
                      << rp->GetClassName());
      }
    }
  }
}

//------------------------------------------------------------------------------
vtkMultiVolumeTexture* vtkOpenGLGPUMultiVolumeRayCastMapper::GetVolumeTexture(int i)
{
	if (!this->VolumeTexture[i]) {
		this->VolumeTexture[i] = vtkMultiVolumeTexture::New();
		this->VolumeTexture[i]->SetMapper(this);
		this->VolumeTexture[i]->SetIndex(i);
	}
	return this->VolumeTexture[i];
}