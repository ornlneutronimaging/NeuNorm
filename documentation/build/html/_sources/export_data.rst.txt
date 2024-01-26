***********
Export Data
***********

It is possible to export any of the data you worked on (sample, ob, df or normalized) either
as a 'tif' or as a 'fits' file (default being 'tif')

>>> output_folder = '/users/my_output_folder'
>>> o_norm.export(folder=output_folder, data_type='normalized', file_type='tif')

or if you prefer 'fits'

>>> o_norm.export(folder=output_folder, data_type='normalized', file_type='fits')