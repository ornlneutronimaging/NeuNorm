*************
Normalization
*************

Normalization using ROI (optional)
##################################

If you want to specify a region of your sample to match with the OB

Let's use the following region 

- x0 = 10
- y0 = 10
- x1 = 50
- y1 = 50

>>> my_norm_roi = ROI(x0=10, y0=10, x1=50, y1=50)

then the normalization can be run

>>> o_norm.normalization(roi=my_norm_roi)


Normalization without ROI
#########################

If you don't want any normalization ROI, simply run the normalization

>>> o_norm.normalization()

How to get the normalized data

Each of the data set in the sample and ob will then be normalized.
If a norm_roi has been provided, the sample arrays will be divided by the average of the 
region defined. Same thing for the ob. Those normalized array can be retrieved this way

>>> sample_normalized_array = o_norm.data['sample']['data']
>>> ob_normalized_array = o_gretting.data['ob']['data']


Forcing normalization by mean OB
################################

By default, if the number of sample and OB is the same, each sample is normalized by the equivalent index ob. But
it's possible to force the normalization by the mean OB

>>> o_norm.normalization(force_mean_ob=True)


Forcing normalization by median OB
##################################

By default, if the number of sample and OB is the same, each sample is normalized by the equivalent index ob. But
it's possible to force the normalization by the median OB

>>> o_norm.normalization(force_median_ob=True)


Normalization by a region defined within the sample itself
##########################################################

It's also possible to normalize the stack of data by using a region of the sample we define as background. In this case
you need to define a ROI and then use the flag *use_only_sample* as shown here

>>> o_norm.normalization(use_only_sample=True)

In this case, the program will determine for each image the *mean* counts of the ROI defined, and will divide each
pixel counts by this value.
