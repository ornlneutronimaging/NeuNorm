*************
Cropping Data
*************

You have the option to crop the data but if you do, this must be done after running the normalization. 
The algorithm only cropped the normalized sample and ob data

- the 4 corners of the region of interest (ROI)
- the top left corner coordinates, width and height of the ROI

let's use the first method and let's pretend the ROI is defined by

- x0 = 5
- y0 = 5
- x1 = 200
- y1 = 250

>>> my_crop_roi = ROI(x0=5, y0=5, x1=200, y1=250)
>>> o_norm.crop(roi=my_crop_roi)
