*******
Loading
*******

You can load your data into the library using different methods:

- by defining the input folder
- by defining the full file name, one at a time
- by defining the the list of full file names
- by giving directly the arrays of images


Loading via folders
###################
  
Let's pretend that our images are in the folder **/Users/me/sample/** and named 

- image001.fits
- image002.fits
- image003.fits

>>> sample_folder = '/Users/me/sample/'
>>> o_norm = Normalization()
>>> o_norm.load(folder=sample_folder)

At this point all the data have been loaded in memory and can be accessed as followed

>>> image001 = o_norm.data['sample']['data'][0]
>>> image002 = o_norm.data['sample']['data'][1]

and the file names

>>> image003_file_name = o_norm.data['sample']['file_name'][2]

Let's now load the rest of our data, the OB and the DF

Our OB are in the folder **/Users/me/ob/** and named

- ob001.fits
- ob002.fits
- ob003.fits

>>> o_norm.load(folder='/Users/me/ob', data_type='ob')

again, all the data can be retrieved as followed

>>> ob1 = o_norm.data['ob']['data'][0]
>>> ob2_file_name = o_norm.data['ob']['file_name'][1]

For this library, DF are optional but for the sake of this exercise, let's load them 

>>> o_norm.load(folder='/Users/me/df', data_type='df')

By default, a gamma filtering will take place when you load your data. You can manually turn off
this filtering by adding the following False flag

>>> o_norm.load(folder='/Users/me/df', data_type='df', gamma_filter=False)

The gamma filtering is an algorithm that replaces all the very bright pixel counts with the average value
of the 8 neighbors. What do we mean by very bright? The pixel counts that have 10% of their value above the average
counts of the entire image. The threshold value can be change by doing

>>> o_norm.gamma_filter_threshold = 0.2

**WARNING:**

 * #1 From this point, any operation on your data will overwrite the inital data loaded.
 * #2 The program won't let you run the same algorithm twice (normalization, df_correction, oscillation, rebin). But it's possible to overwrite this option by making a flag **force** equal to True. Use this feature at your own risk!

>>> data = o_norm.data['sample']['data']
>>> ob = o_norm.data['ob']['data']

Loading via individual file name
################################
  
Let's pretend that our images are in the folder **/Users/me/sample/** and named 

- image001.fits
- image002.fits
- image003.fits

>>> o_norm = Normalization()
>>> o_norm.load(file='/Users/me/sample/image001.fits')
>>> o_norm.load(file='/Users/me/sample/image002.fits')
>>> o_norm.load(file='/Users/me/sample/image003.fits')

At this point all the data have been loaded in memory and can be accessed as followed

>>> image001 = o_norm.data['sample']['data'][0]
>>> image002 = o_norm.data['sample']['data'][1]

and the file names

>>> image003_file_name = o_norm.data['sample']['file_name'][2]

Let's now load the rest of our data, the OB and the DF

Our OB are in the folder **/Users/me/ob/** and named

- ob001.fits
- ob002.fits
- ob003.fits

>>> o_norm.load(file='/Users/me/ob/ob001.fits', data_type='ob')
>>> o_norm.load(file='/Users/me/ob/ob002.fits', data_type='ob')
>>> o_norm.load(file='/Users/me/ob/ob003.fits', data_type='ob')

again, all the data can be retrieved as followed

>>> ob1 = o_norm.data['ob']['data'][0]
>>> ob2_file_name = o_norm.data['ob']['file_name'][1]

For this library, DF are optional but for the sake of this exercise, let's load them 

- df001.fits
- df002.fits

>>> o_norm.load(file='/Users/me/df/df001.fits', data_type='df')
>>> o_norm.load(file='/Users/me/df/df002.fits', data_type='df')

By default, a gamma filtering will take place when you load your data. You can manually turn off
this filtering by adding the following False flag

>>> o_norm.load(file='/Users/me/df/df002.fits', data_type='df', gamma_filter=False)

The gamma filtering is an algorithm that replaces all the very bright pixel counts with the average value
of the 8 neighbors. What do we mean by very bright? The pixel counts that have 10% of their value above the average
counts of the entire image. The threshold value can be change by doing

>>> o_norm.gamma_filter_threshold = 0.2

**WARNING:**
#1 From this point, any operation on your data will overwrite the inital data loaded. Those
data can be retrieved at any point by doing
#2 The program won't let you run the same algorithm twice (normalization, df_correction, 
oscillation, rebin). But it's possible to overwrite this option by making a flag **force**
equal to True. Use this feature at your own risk!

>>> data = o_norm.data['sample']['data']
>>> ob = o_norm.data['ob']['data']

Loading via list file names
###########################
  
Let's pretend that our images are in the folder **/Users/me/sample/** and named 

- image001.fits
- image002.fits
- image003.fits

But from this list, we only want to load image001 and image002. It is possible to specify a list of
file names to load

>>> o_norm = Normalization()
>>> list_files = ['/Users/me/sample/image001.fits', '/Users/me/sample/image002.fits']
>>> o_norm.load(file=list_files)

At this point all the data have been loaded in memory and can be accessed as followed

>>> image001 = o_norm.data['sample']['data'][0]
>>> image002 = o_norm.data['sample']['data'][1]

and the file names

>>> image002_file_name = o_norm.data['sample']['file_name'][1]

Let's now load the rest of our data, the OB and the DF

Our OB are in the folder **/Users/me/ob/** and named

- ob001.fits
- ob002.fits

>>> list_ob = ['/Users/me/ob/ob001.fits', '/Users/me/ob/ob002.fits']
>>> o_norm.load(file=list_ob, data_type='ob')

again, all the data can be retrieved as followed

>>> ob1 = o_norm.data['ob']['data'][0]
>>> ob2_file_name = o_norm.data['ob']['file_name'][1]

For this library, DF are optional but for the sake of this exercise, let's load them 

- df001.fits
- df002.fits

>>> list_df = ['/Users/me/df/df001.fits', '/Users/me/df/df002.fits']
>>> o_norm.load(file=list_df, data_type='df')

By default, a gamma filtering will take place when you load your data. You can manually turn off
this filtering by adding the following False flag

>>> o_norm.load(file=list_df, data_type='df', gamma_filter=False)

The gamma filtering is an algorithm that replaces all the very bright pixel counts with the average value
of the 8 neighbors. What do we mean by very bright? The pixel counts that have 10% of their value above the average
counts of the entire image. The threshold value can be change by doing

>>> o_norm.gamma_filter_threshold = 0.2

**WARNING:**
#1 From this point, any operation on your data will overwrite the inital data loaded. Those
data can be retrieved at any point by doing
#2 The program won't let you run the same algorithm twice (normalization, df_correction, 
oscillation, rebin). But it's possible to overwrite this option by making a flag **force**
equal to True. Use this feature at your own risk!

>>> data = o_norm.data['sample']['data']
>>> ob = o_norm.data['ob']['data']

Loading via arrays
##################
  
Let's pretend that our images are in the folder **/Users/me/sample/** and named 

- image001.tif
- image002.tif
- image003.tif

In order to load the arrays, we first need to load ourselves the data

>>> data = []
>>> from PIL import Image
>>> _data1 = Image.open('/Users/me/sample/image001.tif')
>>> data.append(_data1)
>>> _data2 = Image.open('/Users/me/sample/image002.tif')
>>> data.append(_data2)
>>> _data3 = Image.open('/Users/me/sample/image003.tif')
>>> data.append(_data3)

Now, we can load the data

>>> o_norm = Normalization()
>>> o_norm.load(data=data)

At this point all the sample data have been loaded in memory and can be accessed as followed

>>> image001 = o_norm.data['sample']['data'][0]
>>> image002 = o_norm.data['sample']['data'][1]

and the file names

>>> image003_file_name = o_norm.data['sample']['file_name'][2]

Let's now load the rest of our data, the OB and the DF

Our OB are in the folder **/Users/me/ob/** and named

- ob001.tif
- ob002.tif
- ob003.tif

>>> _ob1 = Image.open('/Users/me/sample/ob001.tif')
>>> o_norm.load(data=_ob1, data_type='ob')
>>> _ob2 = Image.open('/Users/me/sample/ob002.tif')
>>> o_norm.load(data=_ob2, data_type='ob')
>>> _ob3 = Image.open('/Users/me/sample/ob003.tif')
>>> o_norm.load(data=_ob3, data_type='ob')

again, all the data can be retrieved as followed

>>> ob1 = o_norm.data['ob']['data'][0]
>>> ob2_file_name = o_norm.data['ob']['file_name'][1]

For this library, DF are optional but for the sake of this exercise, let's load them 

- df001.tif
- df002.tif

>>> _df1 = Image.open('/Users/me/sample/df001.tif')
>>> o_norm.load(data=_df1, data_type='df')
>>> _df2 = Image.open('/Users/me/sample/df002.tif')
>>> o_norm.load(data=_df2, data_type='df')

By default, a gamma filtering will take place when you load your data. You can manually turn off
this filtering by adding the following False flag

>>> o_norm.load(data=_df2, data_type='df', gamma_filter=False)

The gamma filtering is an algorithm that replaces all the very bright pixel counts with the average value
of the 8 neighbors. What do we mean by very bright? The pixel counts that have 10% of their value above the average
counts of the entire image. The threshold value can be change by doing

>>> o_norm.gamma_filter_threshold = 0.2

**WARNING:**
#1 From this point, any operation on your data will overwrite the inital data loaded. Those
data can be retrieved at any point by doing
#2 The program won't let you run the same algorithm twice (normalization, df_correction, 
oscillation, rebin). But it's possible to overwrite this option by making a flag **force**
equal to True. Use this feature at your own risk!

>>> data = o_norm.data['sample']['data']
>>> ob = o_norm.data['ob']['data']

Loading with Auto Gamma Filtering
#################################

By default the data are loaded with **automatic gamma correction** turned **ON**. You can easily turn off this
auto gamma correction this way

>>> sample_folder = '/Users/me/sample/'
>>> o_norm = Normalization()
>>> o_norm.load(folder=sample_folder, auto_gamma_filter=False)

How does the Auto Gamma filter works?
-------------------------------------

The program used the format of the input data files and will replace all the pixels for which their intensity is
equal or greater to the maximum value provided by this data file format - 5 (marging).

For example, if you are loading an image of type int16, the maximum value provided by this image is 32767. All pixels
with more counts than 32762 will be replaced by the average of the 8 surrounding pixels.

Loading with Manual Gamma Filtering
###################################

NeuNorm also allows you to define yourself your gama filtering threshold. To do so, load the data this way

>>> sample_folder = '/Users/me/sample/'
>>> o_norm = Normalization()
>>> o_norm.load(folder=sample_folder, auto_gamma_filter=False, gamma_filter=True, threshold=0.5)

You must turn off the auto gamma filter otherwise the manual gamma filtering won't be trigger.
In this case, the pixel will be considered as gamma pixels, and then be replaced the same way the auto gamma filter
does, this way

    1. The average value of the entire image is calculated
    2. a copy of the raw image multiply by the threshold value is created
    3. if there is any pixels in this image that is still above the raw image, it is a gamma pixel!
