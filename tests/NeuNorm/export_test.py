import unittest
import os
import shutil
import numpy as np

from NeuNorm.normalization import Normalization


class TestExportingPhase1(unittest.TestCase):
    
    def setUp(self):    
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, '../data/'))
        
    def test_error_raised_if_wrong_folder(self):
        '''assert error is raised when folder does not exist'''
        sample_path = self.data_path + '/tif/sample'
        ob_path = self.data_path + '/tif/ob'
        o_norm = Normalization()
        o_norm.load(folder=sample_path,auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        self.assertRaises(IOError, o_norm.export, folder='/unknown/', data_type='sample')

    def test_error_raised_if_data_type_is_not_valid(self):
        '''assert error is raised if data_type is wrong'''
        sample_path = self.data_path + '/tif/sample'
        ob_path = self.data_path + '/tif/ob'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        self.assertRaises(KeyError, o_norm.export, data_type='not_real_type')
        
    def test_do_nothing_if_nothing_to_export(self):
        '''assert do nothing if nothing to export'''
        sample_path = self.data_path + '/tif/sample'
        ob_path = self.data_path + '/tif/ob'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        self.assertFalse(o_norm.export(data_type='df'))
        
class TestExportingPhase2(unittest.TestCase):
    
    def setUp(self):    
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, '../data/'))
        self.export_folder = self.data_path + '/temporary_folder/'
        os.mkdir(self.export_folder)
        
    def tearDown(self):
        shutil.rmtree(self.export_folder)
        
    def test_export_create_the_right_file_name(self):
        '''assert export works for all data types for tif output'''

        sample_path = self.data_path + '/tif/sample'
        ob_path = self.data_path + '/tif/ob'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)

        # OB
        o_norm.export(folder=self.export_folder, data_type='ob') 
        _output_file_name_0 = o_norm._export_file_name[0]
        
        _file_name_0 = os.path.basename(o_norm.data['ob']['file_name'][0])
        _new_file_name = os.path.splitext(_file_name_0)[0] + '.tif'
        _expected_file_name_0 = os.path.join(self.export_folder, _new_file_name)

        self.assertTrue(_expected_file_name_0 == _output_file_name_0)
        
        # Normalized
        o_norm.normalization()
        o_norm.export(folder=self.export_folder, data_type='normalized')
        _output_file_name_0 = o_norm._export_file_name[0]
    
        _file_name_0 = os.path.basename(o_norm.data['sample']['file_name'][0])
        _new_file_name = 'normalized_' + os.path.splitext(_file_name_0)[0] + '.tif'
        _expected_file_name_0 = os.path.join(self.export_folder, _new_file_name)

        self.assertTrue(_expected_file_name_0 == _output_file_name_0)
        
    def test_export_works_for_tif(self):
        '''assert the file created is correct for tif images'''
        sample_path = self.data_path + '/tif/sample'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        _sample_0 = o_norm.data['sample']['data'][0]
        o_norm.export(folder=self.export_folder, data_type='sample')    
        
        o_norm_2 = Normalization()
        o_norm_2.load(folder=self.export_folder, auto_gamma_filter=False)
        _sample_reloaded = o_norm_2.data['sample']['data'][0]
        
        self.assertTrue((_sample_0 == _sample_reloaded).all())

    def test_export_works_for_tiff_metadata(self):
        '''assert file created using tif has the metadata as well'''
        sample_path = self.data_path + '/tif/sample'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.export(folder=self.export_folder, data_type='sample')

        o_norm_2 = Normalization()
        o_norm_2.load(folder=self.export_folder, auto_gamma_filter=False)

        for index in np.arange(len(o_norm.data['sample']['data'])):
            input_metadata = str(o_norm.data['sample']['metadata'][index])
            export_metadata = str(o_norm_2.data['sample']['metadata'][index])
            self.assertTrue((input_metadata == export_metadata))

    def test_export_works_for_fits(self):
        '''assert the file created is correct for fits images'''
        sample_path = self.data_path + '/fits/sample'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        _sample_0 = o_norm.data['sample']['data'][0]
        o_norm.export(folder=self.export_folder, data_type='sample', file_type='fits')    
        
        o_norm_2 = Normalization()
        o_norm_2.load(folder=self.export_folder, auto_gamma_filter=False)
        _sample_reloaded = o_norm_2.data['sample']['data'][0]
        
        self.assertTrue((_sample_0 == _sample_reloaded).all())

    def test_export_with_manually_loaded_data(self):
        '''assert the file is correctly exported when loaded manually'''
        sample_path = self.data_path + '/fits/sample'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)

        data =  o_norm.data['sample']['data'][0]
        file_name = os.path.join(self.data_path, '/fits/sample/image001.fits')
        o_norm_1 = Normalization()
        o_norm_1.load(data=data, auto_gamma_filter=False)
        o_norm_1.data['sample']['file_name'] = [file_name]
        _sample_0  = o_norm.data['sample']['data'][0]
        o_norm_1.export(folder=self.export_folder, data_type='sample', file_type='fits')

        # making sure the file exists first
        output_file = os.path.join(self.export_folder, 'image001.fits')
        self.assertTrue(os.path.exists(output_file))

        o_norm_2 = Normalization()
        o_norm_2.load(folder=self.export_folder, auto_gamma_filter=False)
        _sample_reloaded = o_norm_2.data['sample']['data'][0]

        self.assertTrue((_sample_0 == _sample_reloaded).all())

