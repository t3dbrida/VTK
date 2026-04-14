## Fix vtkDICOMImageReader incorrectly reading scientific notation

A regression in VTK 9.6.0 caused metadata stored in scientific notation
to be read incorrectly or, when built in debug mode, to cause a failed
assertion and a crash.  This issue is now fixed.
