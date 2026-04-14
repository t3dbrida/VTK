// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "MeshCacheMockAlgorithms.h"

#include "vtkDataObjectMeshCache.h"
#include "vtkLogger.h"
#include "vtkMeshCacheRunner.h"
#include "vtkNew.h"
#include "vtkPolyData.h"
#include "vtkTestUtilities.h"

#include <cstdlib>

namespace
{
/**
 * Without the automatic update, leaving scope lets the cache empty.
 */
bool FirstStepNoUpdate(vtkDataObjectMeshCache* cache, vtkPolyData* input, vtkPolyData* output)
{
  vtkLogScopeFunction(INFO);
  {
    vtkMeshCacheRunner runner{ cache, input, output, false };
    auto status = cache->GetStatus();
    if (status.enabled())
    {
      vtkLog(ERROR, "ERROR: cache should not have been initialized.");
      return false;
    }
    if (output->GetNumberOfPoints() != 0)
    {
      vtkLog(ERROR, "ERROR: cache is empty and output should be empty too.");
      return false;
    }
  }
  auto status = cache->GetStatus();
  if (status.enabled() || status.CacheDefined)
  {
    vtkLog(ERROR, "ERROR: cache should not have been init.");
    return false;
  }

  return true;
}

/**
 * The cache is initialized from the runner destructor, when is leaving its scope.
 */
bool SecondStepAutomaticUpdate(
  vtkDataObjectMeshCache* cache, vtkPolyData* input, vtkPolyData* output)
{
  vtkLogScopeFunction(INFO);
  {
    vtkMeshCacheRunner runner{ cache, input, output, true };
    if (output->GetNumberOfPoints() != 0)
    {
      vtkLog(ERROR, "Output should still be empty");
      return false;
    }
    // cache will be init from output: be sure it is not empty.
    output->ShallowCopy(input);
  }
  auto status = cache->GetStatus();
  if (!status.enabled())
  {
    vtkLog(ERROR, "runner with auto update should have initialized cache.");
    return false;
  }
  return true;
}

/**
 * Output is finally updated using cache.
 */
bool ThirdStepInitOutput(vtkDataObjectMeshCache* cache, vtkPolyData* input, vtkPolyData* output)
{
  vtkLogScopeFunction(INFO);
  // cleanup to be able to check the content after cache usage.
  output->Initialize();
  {
    vtkMeshCacheRunner runner{ cache, input, output, true };
    if (output->GetNumberOfPoints() == 0)
    {
      vtkLog(ERROR, "Output should not be empty");
      return false;
    }
  }
  if (!vtkTestUtilities::CompareDataObjects(input, output))
  {
    vtkLog(ERROR, "Output should have been initialized from input");
    return false;
  }

  return true;
}
}

int TestMeshCacheRunner(int vtkNotUsed(argc), char* vtkNotUsed(argv)[])
{
  vtkNew<vtkDataObjectMeshCache> cache;
  vtkNew<vtkStaticDataSource> source;
  vtkNew<vtkConsumerDataFilter> passThroughFilter;
  passThroughFilter->SetInputConnection(source->GetOutputPort());
  passThroughFilter->Update();

  auto data = passThroughFilter->GetPolyDataOutput();
  vtkNew<vtkPolyData> output;

  cache->SetConsumer(passThroughFilter);
  cache->SetOriginalDataObject(data);
  cache->PreserveAttributesOn();
  cache->ForwardAttribute(vtkDataObject::POINT);

  if (!::FirstStepNoUpdate(cache, data, output))
  {
    return EXIT_FAILURE;
  }

  if (!::SecondStepAutomaticUpdate(cache, data, output))
  {
    return EXIT_FAILURE;
  }

  if (!::ThirdStepInitOutput(cache, data, output))
  {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
