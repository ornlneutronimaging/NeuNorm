"""Normalization module for NeuNorm"""
#!/usr/bin/env python
from pathlib import Path
import numpy as np
import os
import copy
from scipy.ndimage import convolve
from tqdm.auto import tqdm

from NeuNorm.loader import load_hdf, load_tiff, load_fits
from NeuNorm.exporter import make_fits, make_tif
from NeuNorm.roi import ROI
from NeuNorm._utilities import get_sorted_list_images, average_df
from NeuNorm import DataType


class Normalization:
    working_data_type = np.float32

    def __init__(self):
        self.shape = {'width' : np.NaN,
                      'height': np.NaN}
        self.dict_image = {'data'      : None,
                           'oscilation': None,
                           'file_name' : None,
                           'metadata'  : None,
                           'shape'     : copy.deepcopy(self.shape)}
        self.dict_ob = {'data'      : None,
                        'oscilation': None,
                        'metadata'  : None,
                        'file_name' : None,
                        'data_mean' : None,
                        'shape'     : copy.deepcopy(self.shape)}
        self.dict_df = {'data'        : None,
                        'metadata'    : None,
                        'data_average': None,
                        'file_name'   : None,
                        'shape'       : copy.deepcopy(self.shape)}

        __roi_dict = {'x0': np.NaN,
                      'x1': np.NaN,
                      'y0': np.NaN,
                      'y1': np.NaN}
        self.roi = {'normalization': copy.deepcopy(__roi_dict),
                    'crop'         : copy.deepcopy(__roi_dict)}

        self.__exec_process_status = {'df_correction': False,
                                      'normalization': False,
                                      'crop'         : False,
                                      'oscillation'  : False,
                                      'bin'          : False}

        self.data = {}
        self.data['sample'] = self.dict_image
        self.data['ob'] = self.dict_ob
        self.data['df'] = self.dict_df
        self.data['normalized'] = None
        self.export_file_name = None

    def load(self, file='', folder='', data=None, data_type='sample', auto_gamma_filter=True,
             manual_gamma_filter=False, notebook=False, manual_gamma_threshold=0.1, check_shape=True):
        """
        Function to read individual files, entire files from folder, list of files or event data arrays.
        Data are also gamma filtered if requested.
        
        Parameters:
           file: list -  full path to a single file, or list of files
           folder: string - full path to folder containing files to load
           data: numpy array - 2D array of data to load
           data_type: string - 'sample', 'ob' or 'df (default 'sample')
           auto_gamma_filter: boolean - will correct the gamma filter automatically (highest count possible
                for the data type will be replaced by the average of the 9 neighboring pixels) (default True)
           manual_gamma_filter: boolean - apply or not gamma filtering to the data loaded (default False)
           notebooks: boolean - turn on this option if you run the library from a
             notebook to have a progress bar displayed showing you the progress of the loading (default False)
            manual_gamma_threshold: float between 0 and 1 - manual gamma coefficient to use (default 0.1)

        Warning:
            Algorithm won't be allowed to run if any of the main algorithm have been run already, such as
                oscillation, crop, binning, df_correction.

        """

        list_exec_flag = [_flag for _flag in self.__exec_process_status.values()]
        if True in list_exec_flag:
            raise IOError("Operation not allowed as you already worked on this data set!")

        if not file == '':
            if isinstance(file, str):
                self.load_file(file=file,
                               data_type=data_type,
                               auto_gamma_filter=auto_gamma_filter,
                               manual_gamma_filter=manual_gamma_filter,
                               manual_gamma_threshold=manual_gamma_threshold,
                               check_shape=check_shape)
            elif isinstance(file, list):
                # use tqdm to handle the progress bar
                for _file in tqdm(file, desc=f'Loading {data_type}', leave=False):
                    self.load_file(file=_file,
                                   data_type=data_type,
                                   auto_gamma_filter=auto_gamma_filter,
                                   manual_gamma_filter=manual_gamma_filter,
                                   manual_gamma_threshold=manual_gamma_threshold,
                                   check_shape=check_shape)

        elif not folder == '':
            # load all files from folder
            list_images = get_sorted_list_images(folder=folder)
            # use tqdm to handle the progress bar
            for _image in tqdm(list_images, desc=f'Loading {data_type}', leave=False):
                full_path_image = os.path.join(folder, _image)
                self.load_file(file=full_path_image,
                               data_type=data_type,
                               auto_gamma_filter=auto_gamma_filter,
                               manual_gamma_filter=manual_gamma_filter,
                               manual_gamma_threshold=manual_gamma_threshold,
                               check_shape=check_shape)

        elif not data is None:
            self.load_data(data=data, data_type=data_type)

    def calculate_how_long_its_going_to_take(self, index_we_are=-1, time_it_took_so_far=0, total_number_of_loop=1):
        """Estimate how long the loading is going to take according to the time it already took to load the
        first images.

        Parameters:
            index_we_are: int - index where we are in the list of files to load (default -1)
            time_it_took_so_far: float - time it took so far to load the data (default 0)
            total_number_of_loop: int - total number of files to load (default 1)

        Returns:
            string
        """
        time_per_loop = time_it_took_so_far / index_we_are
        total_time_it_will_take = time_per_loop * total_number_of_loop
        time_left = total_time_it_will_take - time_per_loop * index_we_are

        # convert to nice format h mn and seconds
        m, s = divmod(time_left, 60)
        h, m = divmod(m, 60)

        if h == 0:
            if m == 0:
                return "%02ds" % (s)
            else:
                return "%02dmn %02ds" % (m, s)
        else:
            return "%dh %02dmn %02ds" % (h, m, s)

    def load_data(self, data=None, data_type='sample', notebook=False):
        """Function to save the data already loaded as arrays

        Paramters:
            data: np array 2D or 3D
            data_type: string  - 'sample', 'ob' or 'df' (default 'sample')
            notebook: boolean - turn on this option if you run the library from a
                 notebook to have a progress bar displayed showing you the progress of the loading (default False)
        """
        if len(np.shape(data)) > 2:
            # use tqdm to handle the progress bar
            for _data in tqdm(data, desc=f'Loading {data_type}', leave=False):
                _data = _data.astype(self.working_data_type)
                self.__load_individual_data(data=_data, data_type=data_type)

        else:
            data = data.astype(self.working_data_type)
            self.__load_individual_data(data=data, data_type=data_type)

    def __load_individual_data(self, data=None, data_type='sample'):
        """method that loads the data one at a time

        Parameters:
            data: np array
            data_type: string - 'data', 'ob' or 'df' (default 'sample')
        """
        if self.data[data_type]['data'] is None:
            self.data[data_type]['data'] = [data]
        else:
            self.data[data_type]['data'].append(data)
        index = len(self.data[data_type]['data'])
        if self.data[data_type]['file_name'] is None:
            self.data[data_type]['file_name'] = ["image_{:04}".format(index)]
        else:
            self.data[data_type]['file_name'].append("image_{:04}".format(index))
        if self.data[data_type]['metadata'] is None:
            self.data[data_type]['metadata'] = ['']
        else:
            self.data[data_type]['metadata'].append('')
        self.save_or_check_shape(data=data, data_type=data_type)

    def load_file(self, file='', data_type='sample',
                  auto_gamma_filter=True,
                  manual_gamma_filter=False,
                  manual_gamma_threshold=0.1,
                  check_shape=True):
        """
        Function to read data from the specified path, it can read FITS, TIFF and HDF.
    
        Parameters
            file : string - full path of the input file with his extension.
            data_type: string - 'sample', 'df' or 'ob' (default 'sample')
            manual_gamma_filter: boolean  - apply or not gamma filtering (default False)
            manual_gamma_threshold: float (between 0 and 1) - manual gamma threshold
            auto_gamma_filter: boolean - flag to turn on or off the auto gamma fitering (default True)

        Raises:
            OSError: if file does not exist
            NotImplementedError: if file is HDF5
            OSError: if any other any file format requested

        """
        my_file = Path(file)
        if my_file.is_file():
            metadata = {}
            if file.lower().endswith('.fits'):
                data = np.array(load_fits(my_file))
            elif file.lower().endswith(('.tiff', '.tif')):
                [data, metadata] = load_tiff(my_file)
                data = np.array(data)
            elif file.lower().endswith(('.hdf', '.h4', '.hdf4', '.he2', 'h5', '.hdf5', '.he5')):
                raise NotImplementedError
            #     data = np.array(load_hdf(my_file))
            else:
                raise OSError('file extension not yet implemented....Do it your own way!')

            if auto_gamma_filter:
                data = self._auto_gamma_filtering(data=data)
            elif manual_gamma_filter:
                data = self._manual_gamma_filtering(data=data, manual_gamma_threshold=manual_gamma_threshold)

            data = np.squeeze(data)

            if self.data[data_type]['data'] is None:
                self.data[data_type]['data'] = [data]
            else:
                self.data[data_type]['data'].append(data)

            if self.data[data_type]['metadata'] is None:
                self.data[data_type]['metadata'] = [metadata]
            else:
                self.data[data_type]['metadata'].append(metadata)

            if self.data[data_type]['file_name'] is None:
                self.data[data_type]['file_name'] = [file]
            else:
                self.data[data_type]['file_name'].append(file)

            if check_shape:
                self.save_or_check_shape(data=data, data_type=data_type)

        else:
            raise OSError("The file name does not exist")

    def _auto_gamma_filtering(self, data=None):
        """perform the automatic gamma filtering

        This algorithm check the data format of the input data file (ex: int16, int32...)
        and will determine the maxixum value for this data type. Any pixel that have a value
        above the max value - 5 (just to give it a little bit of range) will be considered as
        being gamma pixels. Those pixels will be replaced by the average value of the 8 pixels
        surrounding this pixel

        Parameters:
            data: np array

        Returns:
            np array of the data cleaned

        Raises:
            ValueError if array is empty
        """
        if data is None:
            raise ValueError("Data array is empty!")

        # we may be dealing with a float time, that means it does not need any gamma filtering

        try:
            max = np.iinfo(data.dtype).max
        except:
            return data

        manual_gamma_threshold = max - 5
        new_data = np.array(data, self.working_data_type)

        data_gamma_filtered = np.copy(new_data)
        gamma_indexes = np.where(new_data > manual_gamma_threshold)

        mean_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 8.0
        convolved_data = convolve(data_gamma_filtered, mean_kernel, mode='constant')

        data_gamma_filtered[gamma_indexes] = convolved_data[gamma_indexes]

        return data_gamma_filtered

    def _manual_gamma_filtering(self, data=None, manual_gamma_threshold=0.1):
        """perform manual gamma filtering on the data

        This algoritm uses the manual_gamma_threshold value to estimate if a pixel is a gamma or not.
        1. mean value of data array is calculated
        2. pixel is considered gamma if its value times the manual gamma threshold is bigger than the mean value
        3. if pixel is gamma, its value is replaced by the mean value of the 8 pixels surrounding it.

        Parameters:
            data: numpy 2D array
            manual_gamma_threshold: float - coefficient between 0 and 1 used to estimate the threshold of the
            gamma pixels (default 0.1)
        
        Returns:
            numpy 2D array

        Raises:
             ValueError if data is empty
        """
        if data is None:
            raise ValueError("Data array is empty!")

        data_gamma_filtered = np.copy(data)
        mean_counts = np.mean(data_gamma_filtered)
        gamma_indexes = np.where(manual_gamma_threshold * data_gamma_filtered > mean_counts)

        mean_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 8.0
        convolved_data = convolve(data_gamma_filtered, mean_kernel, mode='constant')

        data_gamma_filtered[gamma_indexes] = convolved_data[gamma_indexes]

        return data_gamma_filtered

    def save_or_check_shape(self, data=None, data_type='sample'):
        """save the shape for the first data loaded (of each type) otherwise
        check if the size match

        Parameters:
            data: np array of the data to check or save shape (default [])
            data_type: string - 'ob', 'df' or 'sample' (default 'sample')

        Raises:
            IOError if size do not match
        """
        [height, width] = np.shape(data)
        if np.isnan(self.data[data_type]['shape']['height']):
            _shape = copy.deepcopy(self.shape)
            _shape['height'] = height
            _shape['width'] = width
            self.data[data_type]['shape'] = _shape
        else:
            _prev_width = self.data[data_type]['shape']['width']
            _prev_height = self.data[data_type]['shape']['height']

            if (not (_prev_width == width)) or (not (_prev_height == height)):
                raise IOError("Shape of {} do not match previous loaded data set!".format(data_type))

    def normalization(self, roi=None, force=False, force_mean_ob=False, force_median_ob=False,
                      notebook=False, use_only_sample=False):
        """normalization of the data
                
        Parameters:
            roi: ROI object or list of ROI objects - object defines the region of the sample and OB that have to match
        in intensity
            force: boolean - True will force the normalization to occur, even if it had been
                run before with the same data set (default False)
        notebook: boolean - turn on this option if you run the library from a
             notebook to have a progress bar displayed showing you the progress of the loading (default False)
        use_only_sample - turn on this option to normalize the sample data using the ROI on the sample. each pixel
            counts will be divided by the average counts of all the ROI of the same image

        Return:
            True - status of the normalization (True if every went ok, this is mostly used for the unit test)

        Raises:
            IOError: if no sample loaded
            IOError: if no OB loaded and use_only_sample if False
            IOError: if use_only_sample is True and no ROI provided
            IOError: if size of sample and OB do not match
        
        """
        if not force:
            # does nothing if normalization has already been run
            if self.__exec_process_status['normalization']:
                return
        self.__exec_process_status['normalization'] = True

        # make sure we loaded some sample data
        if self.data['sample']['data'] is None:
            raise IOError("No normalization available as no data have been loaded")

        # make sure we loaded some ob data
        if not use_only_sample:
            if self.data['ob']['data'] is None:
                raise IOError("No normalization available as no OB have been loaded")

            # make sure the data loaded have the same size
            if not self.data_loaded_have_matching_shape():
                raise ValueError("Data loaded do not have the same shape!")

        if notebook:
            from ipywidgets import widgets
        from IPython.display import display

        # make sure, if provided, roi has the right type and fits into the images
        b_list_roi = False

        if not use_only_sample:

            if roi:
                b_list_roi = self.check_roi_format(roi)

                if b_list_roi:
                    _sample_corrected_normalized = self.calculate_corrected_normalized(data_type=DataType.sample,
                                                                                       roi=roi)
                    _ob_corrected_normalized = self.calculate_corrected_normalized(data_type=DataType.ob,
                                                                                   roi=roi)

                else:
                    _x0 = roi.x0
                    _y0 = roi.y0
                    _x1 = roi.x1
                    _y1 = roi.y1

                    _sample_corrected_normalized = [_sample / np.mean(_sample[_y0:_y1 + 1, _x0:_x1 + 1])
                                                    for _sample in self.data['sample']['data']]
                    _ob_corrected_normalized = [_ob / np.mean(_ob[_y0:_y1 + 1, _x0:_x1 + 1])
                                                for _ob in self.data['ob']['data']]

            else:

                _sample_corrected_normalized = copy.deepcopy(self.data['sample']['data'])
                _ob_corrected_normalized = copy.deepcopy(self.data['ob']['data'])

            self.data[DataType.sample]['data'] = _sample_corrected_normalized
            self.data[DataType.ob]['data'] = _ob_corrected_normalized

            # if the number of sample and ob do not match, use mean or median of obs
            nbr_sample = len(self.data['sample']['file_name'])
            nbr_ob = len(self.data['ob']['file_name'])
            if (nbr_sample != nbr_ob) or force_mean_ob or force_median_ob:  # work with mean ob

                if force_median_ob:
                    _ob_corrected_normalized = np.nanmedian(_ob_corrected_normalized, axis=0)
                elif force_mean_ob:
                    _ob_corrected_normalized = np.nanmean(_ob_corrected_normalized, axis=0)
                else:
                    _ob_corrected_normalized = np.nanmedian(_ob_corrected_normalized, axis=0)

                self.data['ob']['data_mean'] = _ob_corrected_normalized
                _working_ob = copy.deepcopy(_ob_corrected_normalized)
                _working_ob[_working_ob == 0] = np.NaN

                if notebook:
                    # turn on progress bar
                    _message = "Normalization"
                    box1 = widgets.HBox([widgets.Label(_message,
                                                       layout=widgets.Layout(width='20%')),
                                         widgets.IntProgress(max=len(self.data['sample']['data']))])
                    display(box1)
                    w1 = box1.children[1]

                normalized_data = []
                for _index, _sample in enumerate(self.data['sample']['data']):
                    _norm = np.divide(_sample, _working_ob)
                    _norm[np.isnan(_norm)] = 0
                    _norm[np.isinf(_norm)] = 0
                    normalized_data.append(_norm)

                    if notebook:
                        w1.value = _index + 1

            else:  # 1 ob for each sample

                # produce normalized data
                sample_ob = zip(self.data[DataType.sample]['data'], self.data[DataType.ob]['data'])

                if notebook:
                    # turn on progress bar
                    _message = "Normalization"
                    box1 = widgets.HBox([widgets.Label(_message,
                                                       layout=widgets.Layout(width='20%')),
                                         widgets.IntProgress(max=len(self.data['sample']['data']))])
                    display(box1)
                    w1 = box1.children[1]

                normalized_data = []
                for _index, [_sample, _ob] in enumerate(sample_ob):
                    _working_ob = copy.deepcopy(_ob)
                    _working_ob[_working_ob == 0] = np.NaN
                    _norm = np.divide(_sample, _working_ob)
                    _norm[np.isnan(_norm)] = 0
                    _norm[np.isinf(_norm)] = 0
                    normalized_data.append(_norm)

                    if notebook:
                        w1.value = _index + 1

            self.data['normalized'] = normalized_data

        else:  # use_sample_only with ROI
            normalized_data = self.calculate_corrected_normalized_without_ob(roi=roi)
            self.data['normalized'] = normalized_data

        return True

    def calculate_corrected_normalized_without_ob(self, roi=None):
        if not roi:
            raise ValueError("You need to provide at least 1 ROI using this use_only_sample mode!")

        b_list_roi = self.check_roi_format(roi)

        if b_list_roi:

            normalized_data = []
            for _sample in self.data['sample']['data']:
                total_counts_of_rois = 0
                total_number_of_pixels = 0
                for _roi in roi:
                    _x0 = _roi.x0
                    _y0 = _roi.y0
                    _x1 = _roi.x1
                    _y1 = _roi.y1
                    total_number_of_pixels += (_y1 - _y0 + 1) * (_x1 - _x0 + 1)
                    total_counts_of_rois += np.sum(_sample[_y0:_y1 + 1, _x0:_x1 + 1])

                full_sample_mean = total_counts_of_rois / total_number_of_pixels
                normalized_data.append(_sample / full_sample_mean)

        else:

            _x0 = roi.x0
            _y0 = roi.y0
            _x1 = roi.x1
            _y1 = roi.y1

            normalized_data = [_sample / np.mean(_sample[_y0:_y1 + 1, _x0:_x1 + 1])
                               for _sample in self.data['sample']['data']]

        return normalized_data

    def calculate_corrected_normalized(self, data_type=DataType.sample, roi=None):
        corrected_normalized = []
        for _sample in self.data[data_type]['data']:
            total_counts_of_rois = 0
            total_number_of_pixels = 0
            for _roi in roi:
                _x0 = _roi.x0
                _y0 = _roi.y0
                _x1 = _roi.x1
                _y1 = _roi.y1
                total_number_of_pixels += (_y1 - _y0 + 1) * (_x1 - _x0 + 1)
                total_counts_of_rois += np.sum(_sample[_y0:_y1 + 1, _x0:_x1 + 1])

            full_sample_mean = total_counts_of_rois / total_number_of_pixels
            corrected_normalized.append(_sample / full_sample_mean)
        return corrected_normalized

    def check_roi_format(self, roi):
        b_list_roi = False
        if type(roi) is list:
            for _roi in roi:
                if not type(_roi) == ROI:
                    raise ValueError("roi must be a ROI object!")
                if not self.__roi_fit_into_sample(roi=_roi):
                    raise ValueError("roi does not fit into sample image!")
            b_list_roi = True

        elif not type(roi) == ROI:
            raise ValueError("roi must be a ROI object!")
        else:
            if not self.__roi_fit_into_sample(roi=roi):
                raise ValueError("roi does not fit into sample image!")

        return b_list_roi

    def data_loaded_have_matching_shape(self):
        """check that data loaded have the same shape
        
        Returns:
            bool: result of the check
        """
        _shape_sample = self.data['sample']['shape']
        _shape_ob = self.data['ob']['shape']

        if not (_shape_sample == _shape_ob):
            return False

        _shape_df = self.data['df']['shape']
        if not np.isnan(_shape_df['height']):
            if not (_shape_sample == _shape_df):
                return False

        return True

    def __roi_fit_into_sample(self, roi=None):
        """check if roi is within the dimension of the image
        
        Returns:
            bool: True if roi is within the image dimension
        
        """
        [sample_height, sample_width] = np.shape(self.data['sample']['data'][0])

        [_x0, _y0, _x1, _y1] = [roi.x0, roi.y0, roi.x1, roi.y1]
        if (_x0 < 0) or (_x1 >= sample_width):
            return False

        if (_y0 < 0) or (_y1 >= sample_height):
            return False

        return True

    def df_correction(self, force=False):
        """dark field correction of sample and ob
        
        Parameters
            force: boolean - that if True will force the df correction to occur, even if it had been
                run before with the same data set (default False)

        sample_df_corrected = sample - DF
        ob_df_corrected = OB - DF

        """
        if not force:
            if self.__exec_process_status['df_correction']:
                return
        self.__exec_process_status['df_correction'] = True

        if not self.data['sample']['data'] is None:
            self.__df_correction(data_type='sample')

        if not self.data['ob']['data'] is None:
            self.__df_correction(data_type='ob')

    def __df_correction(self, data_type='sample'):
        """dark field correction
        
        Parameters:
           data_type: string ['sample','ob]

        Raises:
            KeyError: if data type is not 'sample' or 'ob'
            IOError: if sample and df or ob and df do not have the same shape
        """
        if not data_type in ['sample', 'ob']:
            raise KeyError("Wrong data type passed. Must be either 'sample' or 'ob'!")

        if self.data['df']['data'] is None:
            return

        if self.data['df']['data_average'] is None:
            _df = self.data['df']['data']
            if len(_df) > 1:
                _df = average_df(df=_df)
            self.data['df']['data_average'] = np.squeeze(_df)

        else:
            _df = np.squeeze(self.data['df']['data_average'])

        if np.shape(self.data[data_type]['data'][0]) != np.shape(self.data['df']['data'][0]):
            raise IOError("{} and df data must have the same shape!".format(data_type))

        _data_df_corrected = [_data - _df for _data in self.data[data_type]['data']]
        _data_df_corrected = [np.squeeze(_data) for _data in _data_df_corrected]
        self.data[data_type]['data'] = _data_df_corrected

    def crop(self, roi=None, force=False):
        """ Cropping the sample and ob normalized data
        
        Parameters:
            roi: ROI object that defines the region to crop
            force: Boolean  - that force or not the algorithm to be run more than once
                with the same data set (default False)

        Returns:
            True (for unit test purpose)

        Raises:
            ValueError if sample and ob data have not been normalized yet
        """
        if (self.data['sample']['data'] is None) or \
                (self.data['ob']['data'] is None):
            raise IOError("We need sample and ob Data !")

        if not type(roi) == ROI:
            raise ValueError("roi must be of type ROI")

        if not force:
            if self.__exec_process_status['crop']:
                return
        self.__exec_process_status['crop'] = True

        _x0 = roi.x0
        _y0 = roi.y0
        _x1 = roi.x1
        _y1 = roi.y1

        new_sample = [_data[_y0:_y1 + 1, _x0:_x1 + 1] for
                      _data in self.data['sample']['data']]
        self.data['sample']['data'] = new_sample

        new_ob = [_data[_y0:_y1 + 1, _x0:_x1 + 1] for
                  _data in self.data['ob']['data']]
        self.data['ob']['data'] = new_ob

        if not (self.data['df']['data'] is None):
            new_df = [_data[_y0:_y1 + 1, _x0:_x1 + 1] for
                      _data in self.data['df']['data']]
            self.data['df']['data'] = new_df

        if not (self.data['normalized'] is None):
            new_normalized = [_data[_y0:_y1 + 1, _x0:_x1 + 1] for
                              _data in self.data['normalized']]
            self.data['normalized'] = new_normalized

        return True

    def export(self, folder='./', data_type='normalized', file_type='tif'):
        """export all the data from the type specified into a folder
        
        Parameters:
            folder: String - where to create all the images. Folder must exist otherwise an error is
                raised (default is './')
            data_type: String - Must be one of the following 'sample','ob','df','normalized' (default is 'normalized').
            file_type: String - format in which to export the data. Must be either 'tif' or 'fits' (default is 'tif')

        Raises:
            IOError if the folder does not exist
            KeyError if data_type can not be found in the list ['normalized','sample','ob','df']

        """
        if not os.path.exists(folder):
            raise IOError("Folder '{}' does not exist!".format(folder))

        if not data_type in ['normalized', 'sample', 'ob', 'df']:
            raise KeyError("data_type '{}' is wrong".format(data_type))

        prefix = ''
        if data_type == 'normalized':
            data = self.get_normalized_data()
            prefix = 'normalized'
            data_type = 'sample'
        else:
            data = self.data[data_type]['data']

        if data is None:
            return False

        metadata = self.data[data_type]['metadata']

        list_file_name_raw = self.data[data_type]['file_name']
        self.__create_list_file_names(initial_list=list_file_name_raw,
                                      output_folder=folder,
                                      prefix=prefix,
                                      suffix=file_type)

        self.__export_data(data=data,
                           metadata=metadata,
                           output_file_names=self._export_file_name,
                           suffix=file_type)

    def __export_data(self, data=[], metadata=[], output_file_names=[], suffix='tif'):
        """save the list of files with the data specified
        
        Parameters:
            data: numpy array that contains the array of data to save (default [])
            output_file_names: numpy array of string of full file names (default [])
            suffix: String - format in which the file will be created (default 'tif')
        """
        name_data_metadata_array = zip(output_file_names, data, metadata)
        for _file_name, _data, _metadata in name_data_metadata_array:
            if suffix in ['tif', 'tiff']:
                make_tif(data=_data, metadata=_metadata, file_name=_file_name)
            elif suffix == 'fits':
                make_fits(data=_data, file_name=_file_name)

    def __create_list_file_names(self, initial_list=[], output_folder='', prefix='', suffix=''):
        """create a list of the new file name used to export the images
        
        Parameters:
            initial_list: array of full file name
               ex: ['/users/me/image001.tif',/users/me/image002.tif',/users/me/image003.tif']
            output_folder: String (default is ./ as specified by calling function) where we want to create the data
            prefix: String. what to add to the output file name in front of base name
                ex: 'normalized' will produce 'normalized_image001.tif'
            suffix: String. extension to file. 'tif' for TIFF and 'fits' for FITS
        """
        _base_name = [os.path.basename(_file) for _file in initial_list]
        _raw_name = [os.path.splitext(_file)[0] for _file in _base_name]
        _prefix = ''
        if prefix:
            _prefix = prefix + '_'
        full_file_names = [os.path.join(output_folder, _prefix + _file + '.' + suffix) for _file in _raw_name]
        self._export_file_name = full_file_names

    def get_normalized_data(self):
        """return the normalized data"""
        return self.data['normalized']

    def get_sample_data(self):
        """return the sample data"""
        return self.data['sample']['data']

    def get_ob_data(self):
        """return the ob data"""
        return self.data['ob']['data']

    def get_df_data(self):
        """return the df data"""
        return self.data['df']['data']
