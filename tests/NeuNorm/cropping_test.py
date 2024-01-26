import unittest
import numpy as np
import os

from NeuNorm.normalization import Normalization
from NeuNorm.roi import ROI


class TestCropping(unittest.TestCase):
    
    def setUp(self):    
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, '../data/'))
    
    def test_cropping_raises_error_when_no_data_and_ob_loaded(self):
        '''assert error raised when no sample and ob data loaded'''
        o_norm = Normalization()
        _roi = ROI(x0=0, y0=0, x1=4, y1=4)
        self.assertRaises(IOError, o_norm.crop, roi=_roi)
        
        o_norm = Normalization()
        sample_path = self.data_path + '/tif/sample'
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        self.assertRaises(IOError, o_norm.crop, roi=_roi)
        
        o_norm = Normalization()
        sample_path = self.data_path + '/tif/sample'
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        ob_path = self.data_path + '/tif/ob'
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        o_norm.normalization()
        self.assertTrue(o_norm.crop(roi=_roi))
        
    def test_roi_object_passed_to_crop(self):
        '''assert wrong roi type raises a ValueError'''
        _roi = {'x0':0, 'y0':1}
        o_norm = Normalization()
        sample_path = self.data_path + '/tif/sample'
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        ob_path = self.data_path + '/tif/ob'
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        o_norm.normalization()
        self.assertRaises(ValueError, o_norm.crop, roi=_roi)
        
    def test_crop_works(self):
        '''assert crop of sample and ob works correctly'''
        x0, y0, x1, y1 = 0, 0, 2, 2
        _roi = ROI(x0=x0, y0=y0, x1=x1, y1=y1)
        o_norm = Normalization()
        sample_path = self.data_path + '/tif/sample'
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        ob_path = self.data_path + '/tif/ob'
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        o_norm.normalization()
        _expected_sample = o_norm.data['sample']['data'][0]
        _expected_sample = _expected_sample[y0:y1+1, x0:x1+1]
        _expected_ob = o_norm.data['ob']['data'][0]
        _expected_ob = _expected_ob[y0:y1+1, x0:x1+1]
        o_norm.crop(roi=_roi)
        _returned_sample = o_norm.data['sample']['data'][0]
        _returned_ob = o_norm.data['ob']['data'][0]
        self.assertTrue((_expected_sample == _returned_sample).all())
        self.assertTrue((_expected_ob == _returned_ob).all())
        
    def test_crop_works_only_once_without_force_flag(self):
        '''assert crop of sample and ob works only once if no force flag used'''
        x0, y0, x1, y1 = 0, 0, 2, 2
        _roi = ROI(x0=x0, y0=y0, x1=x1, y1=y1)
        o_norm = Normalization()
        sample_path = self.data_path + '/tif/sample'
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        ob_path = self.data_path + '/tif/ob'
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        o_norm.normalization()
        # crop run first time
        o_norm.crop(roi=_roi)
        _sample_first_time = o_norm.data['sample']['data'][0]
        _ob_first_time = o_norm.data['ob']['data'][0]
        # crop run second time
        o_norm.crop(roi=_roi)
        _sample_second_time = o_norm.data['sample']['data'][0]
        _ob_second_time = o_norm.data['ob']['data'][0]
        self.assertTrue((_sample_first_time == _sample_second_time).all())
        self.assertTrue((_ob_first_time == _ob_second_time).all())
        
    def test_crop_works_again_if_force_flag_is_true(self):
        '''assert crop of sample and ob works more than once if force flag is true'''
        x0, y0, x1, y1 = 0, 0, 2, 2
        _roi = ROI(x0=x0, y0=y0, x1=x1, y1=y1)
        o_norm = Normalization()
        sample_path = self.data_path + '/tif/sample'
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        ob_path = self.data_path + '/tif/ob'
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        o_norm.normalization()
        # crop run first time
        o_norm.crop(roi=_roi)
        _sample_first_time = o_norm.data['sample']['data'][0]
        _ob_first_time = o_norm.data['ob']['data'][0]
        # crop run second time
        x0, y0, x1, y1 = 0, 0, 1, 1
        _roi = ROI(x0=x0, y0=y0, x1=x1, y1=y1)        
        o_norm.crop(roi=_roi, force=True)
        _sample_second_time = o_norm.data['sample']['data'][0]
        _ob_second_time = o_norm.data['ob']['data'][0]
        #checking output with expected results
        _expected_sample_second_run = _sample_first_time[y0:y1+1, x0:x1+1]
        _expected_ob_second_run = _ob_first_time[y0:y1+1, x0:x1+1]
        self.assertTrue((_sample_second_time == _expected_sample_second_run).all())
        self.assertTrue((_ob_second_time == _expected_ob_second_run).all())
        
    def test_crop_works_on_normalized_data(self):
        '''assert crop works on normalized data'''
        x0, y0, x1, y1 = 0, 0, 2, 2
        _roi = ROI(x0=x0, y0=y0, x1=x1, y1=y1)
        o_norm = Normalization()
        sample_path = self.data_path + '/tif/sample'
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        ob_path = self.data_path + '/tif/ob'
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        o_norm.normalization()
        # cropping
        o_norm.crop(roi=_roi)
        _returned_norm = o_norm.data['normalized'][0]
        _expected_norm = np.ones((3,3))
        _expected_norm[:,2] = 2
        self.assertTrue((_expected_norm == _returned_norm).all())

    def test_crop_works_on_df_data(self):
        '''assert crop works on df data'''
        x0, y0, x1, y1 = 0, 0, 2, 2
        _roi = ROI(x0=x0, y0=y0, x1=x1, y1=y1)
        o_norm = Normalization()
        sample_path = self.data_path + '/tif/sample'
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        ob_path = self.data_path + '/tif/ob'
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        df_path = self.data_path + '/tif/df'
        o_norm.load(folder=df_path, data_type='df', auto_gamma_filter=False)
        # cropping
        o_norm.crop(roi=_roi)
        _returned_df = o_norm.data['df']['data'][0]
        _expected_df = np.ones((3,3))
        self.assertTrue((_expected_df == _returned_df).all())
