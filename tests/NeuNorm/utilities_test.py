import unittest
import numpy as np
import os

from NeuNorm.normalization import Normalization
from NeuNorm._utilities import get_sorted_list_images, average_df


class TestUtilites(unittest.TestCase):
    
    def setUp(self):    
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, '../data/'))
        
    def test_all_images_names_retrieved_from_file(self):
        '''assert list of images are correctly retrieved from individual file name'''
        # tif
        path = self.data_path + '/tif/sample'
        o_norm = Normalization()
        o_norm.load(folder=path, auto_gamma_filter=False)
        list_files_expected = ['image001.tif', 'image002.tif', 'image003.tif']
        list_files = get_sorted_list_images(folder=path)
        self.assertTrue(list_files_expected == list_files)
        
        # fits
        path = self.data_path + '/fits/sample'
        o_norm = Normalization()
        o_norm.load(folder=path, auto_gamma_filter=False)
        list_files_expected = ['image001.fits', 'image002.fits', 'image003.fits']
        list_files = get_sorted_list_images(folder=path)
        self.assertTrue(list_files_expected == list_files)    
        
    def test_df_averaging(self):
        '''assert df average works'''
        df_tif_file_2 = self.data_path + '/tif/df/df002.tif'
        df_tif_file_3 = self.data_path + '/tif/df/df003.tif'
        o_norm = Normalization()
        o_norm.load(file=df_tif_file_2, data_type='df', auto_gamma_filter=False)
        o_norm.load(file=df_tif_file_3, data_type='df', auto_gamma_filter=False)
        _average_df = average_df(df=o_norm.data['df']['data'])
        expected_df = np.ones([5,5])
        expected_df[0,0] = 5
        self.assertTrue((expected_df == _average_df).all())    
      
