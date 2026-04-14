set(classes
  vtkCriticalTime
  vtkForceStaticMesh
  vtkDataObjectMeshCache
  vtkMeshCacheRunner
  vtkTemporalSmoothing)

vtk_module_add_module(VTK::FiltersTemporal
  CLASSES ${classes})

vtk_add_test_mangling(VTK::FiltersTemporal)
