import unittest
import numpy as np
import os

from NeuNorm.normalization import Normalization


class TestLoading(unittest.TestCase):
    
    def setUp(self):    
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, '../data/'))
        
    def test_dict_initialized(self):
        """assert image, ob and df dicts are correctly initialized"""
        o_norm = Normalization()
        data = o_norm.data
        dict_image = o_norm.dict_image
        self.assertEqual(None, dict_image['data'])
        self.assertEqual(None, dict_image['file_name'])
        self.assertEqual(None, data['sample']['data'])
        self.assertEqual(None, data['sample']['file_name'])
        
        dict_ob = o_norm.dict_ob
        self.assertEqual(None, dict_ob['data'])
        self.assertEqual(None, dict_ob['file_name'])
        self.assertEqual(None, data['ob']['data'])
        self.assertEqual(None, data['ob']['file_name'])

        dict_df = o_norm.dict_df
        self.assertEqual(None, dict_df['data'])
        self.assertEqual(None, dict_df['file_name'])
        self.assertEqual(None, data['df']['data'])
        self.assertEqual(None, data['df']['file_name'])

    def test_loading_bad_single_files(self):
        """assert error is raised when inexisting file is given"""
        bad_tiff_file_name = 'bad_tiff_file_name.tiff'
        o_norm = Normalization()
        self.assertRaises(OSError, o_norm.load, bad_tiff_file_name, '', 'sample')
        self.assertRaises(OSError, o_norm.load, bad_tiff_file_name, '', 'ob')
        self.assertRaises(OSError, o_norm.load, bad_tiff_file_name, '', 'df')
        bad_fits_file_name = 'bad_fits_file_name.fits'
        o_norm = Normalization()
        self.assertRaises(OSError, o_norm.load, bad_fits_file_name, '', 'sample')
        self.assertRaises(OSError, o_norm.load, bad_fits_file_name, '', 'ob')
        self.assertRaises(OSError, o_norm.load, bad_fits_file_name, '', 'df')
        bad_h5_file_name = 'bad_h5_file_name.h5'
        self.assertRaises(OSError, o_norm.load, bad_h5_file_name, '', 'sample')
        self.assertRaises(OSError, o_norm.load, bad_h5_file_name, '', 'ob')
        self.assertRaises(OSError, o_norm.load, bad_h5_file_name, '', 'df')

    def test_loading_hdf_raise_error(self):
        """assert hdf5 raise an error when trying to load - not implemented yet"""
        sample_hdf5_file = self.data_path + '/hdf5/dump_file.hdf5'
        o_norm = Normalization()
        self.assertRaises(NotImplementedError, o_norm.load, file=sample_hdf5_file)

    def test_loading_unsuported_file_format(self):
        """assert error is raised when trying to load unsuported file format"""
        sample_fake_file = self.data_path + '/different_format/not_supported_file.fake'
        o_norm = Normalization()
        self.assertRaises(OSError, o_norm.load, file=sample_fake_file)

    def test_loading_good_single_file(self):
        """assert sample, ob and df single file correctly loaded"""
        # tiff
        sample_tif_file = self.data_path + '/tif//sample/image001.tif'
        o_norm = Normalization()
        o_norm.load(file=sample_tif_file, data_type='sample', auto_gamma_filter=False)
        _expected_data = np.ones([5,5])
        _expected_data[0,0] = 5
        _expected_data[:,2] = 2
        _expected_data[:,3] = 3
        _expected_data[:,4] = 4
        _loaded_data = o_norm.data['sample']['data']
        self.assertTrue((_expected_data == _loaded_data).all())
        _expected_name = sample_tif_file
        _loaded_name = o_norm.data['sample']['file_name'][0]
        self.assertTrue(_expected_name == _loaded_name)

        # fits
        sample_fits_file = self.data_path + '/fits//sample/image001.fits'
        o_norm = Normalization()
        o_norm.load(file=sample_fits_file, data_type='sample', auto_gamma_filter=False)
        _expected_data = np.ones([5,5])
        _expected_data[0,0] = 5
        _expected_data[:,2] = 2
        _expected_data[:,3] = 3
        _expected_data[:,4] = 4
        _loaded_data = o_norm.data['sample']['data']
        self.assertTrue((_expected_data == _loaded_data).all())
        _expected_name = sample_fits_file
        _loaded_name = o_norm.data['sample']['file_name'][0]
        self.assertTrue(_expected_name == _loaded_name)
        
    def test_loading_good_several_single_files(self):
        """assert sample, ob and df multi files correctly loaded"""
        # tiff
        sample_tif_file_1 = self.data_path + '/tif//sample/image001.tif'
        sample_tif_file_2 = self.data_path + '/tif/sample/image002.tif'
        ob_tif_file_1 = self.data_path + '/tif/ob/ob001.tif'
        o_norm = Normalization()
        o_norm.load(file=sample_tif_file_1, data_type='sample', auto_gamma_filter=False)
        o_norm.load(file=sample_tif_file_2, data_type='sample', auto_gamma_filter=False)
        o_norm.load(file=ob_tif_file_1, data_type='ob', auto_gamma_filter=False)
        
        # sample 0
        _expected_data_1 = np.ones([5,5])
        _expected_data_1[0,0] = 5
        _expected_data_1[:,2] = 2
        _expected_data_1[:,3] = 3
        _expected_data_1[:,4] = 4
        _loaded_data_1 = o_norm.data['sample']['data'][0]
        self.assertTrue((_expected_data_1 == _loaded_data_1).all())
        _expected_name_1 = sample_tif_file_1
        _loaded_name_1 = o_norm.data['sample']['file_name'][0]
        self.assertTrue(_expected_name_1 == _loaded_name_1)
        
        # sample 1
        _expected_data_2 = np.ones([5,5])
        _expected_data_2[0,0] = 5
        _expected_data_2[:,2] = 2
        _expected_data_2[:,3] = 3
        _expected_data_2[:,4] = 4
        _loaded_data_2 = o_norm.data['sample']['data'][1]
        self.assertTrue((_expected_data_2 == _loaded_data_2).all())
        _expected_name_2 = sample_tif_file_2
        _loaded_name_2 = o_norm.data['sample']['file_name'][1]
        self.assertTrue(_expected_name_2 == _loaded_name_2)        
 
        # ob 0
        _expected_data_1 = np.ones([5,5])
        _expected_data_1[0,0] = 5
        _loaded_data_1 = o_norm.data['ob']['data'][0]
        self.assertTrue((_expected_data_1 == _loaded_data_1).all())
        
        # fits
        sample_fits_file_1 = self.data_path + '/fits//sample/image001.fits'
        sample_fits_file_2 = self.data_path + '/fits/sample/image002.fits'
        o_norm = Normalization()
        o_norm.load(file=sample_fits_file_1, data_type='sample', auto_gamma_filter=False)
        o_norm.load(file=sample_fits_file_2, data_type='sample', auto_gamma_filter=False)
    
        _expected_data_1 = np.ones([5,5])
        _expected_data_1[0,0] = 5
        _expected_data_1[:,2] = 2
        _expected_data_1[:,3] = 3
        _expected_data_1[:,4] = 4
        _loaded_data_1 = o_norm.data['sample']['data'][0]
        self.assertTrue((_expected_data_1 == _loaded_data_1).all())
        _expected_name_1 = sample_fits_file_1
        _loaded_name_1 = o_norm.data['sample']['file_name'][0]
        self.assertTrue(_expected_name_1 == _loaded_name_1)
    
        _expected_data_2 = np.ones([5,5])
        _expected_data_2[0,0] = 5
        _expected_data_2[:,2] = 2
        _expected_data_2[:,3] = 3
        _expected_data_2[:,4] = 4
        _loaded_data_2 = o_norm.data['sample']['data'][1]
        self.assertTrue((_expected_data_2 == _loaded_data_2).all())
        _expected_name_2 = sample_fits_file_2
        _loaded_name_2 = o_norm.data['sample']['file_name'][1]
        self.assertTrue(_expected_name_2 == _loaded_name_2)                 
    
    def test_all_images_names_retrieved_from_folder(self):
        """assert list_of images are correctly loaded when retrieved from folder"""
        # tif
        path = self.data_path + '/tif/sample'
        o_norm = Normalization()
        o_norm.load(folder=path, data_type='sample', auto_gamma_filter=False)
        list_of_files = ['image001.tif', 'image002.tif', 'image003.tif']
        list_of_files_expected = [os.path.join(path, _file) for _file in list_of_files]
        list_of_files_retrieved = o_norm.data['sample']['file_name']
        self.assertTrue(list_of_files_expected == list_of_files_retrieved)
        
        #fits
        path = self.data_path + '/fits/sample'
        o_norm = Normalization()
        o_norm.load(folder=path, data_type='sample', auto_gamma_filter=False)
        list_of_files = ['image001.fits', 'image002.fits', 'image003.fits']
        list_of_files_expected = [os.path.join(path, _file) for _file in list_of_files]
        list_of_files_retrieved = o_norm.data['sample']['file_name']
        self.assertTrue(list_of_files_expected == list_of_files_retrieved)
        
    def test_error_raised_when_data_size_do_not_match(self):
        """assert IOError raised when data of a same type do not match in size"""
        # sample
        image1 = self.data_path + '/tif/sample/image001.tif'
        image2 = self.data_path + '/different_format/image001_4_by_4.tif'
        o_norm = Normalization()
        o_norm.load(file=image1, auto_gamma_filter=False)
        self.assertRaises(IOError, o_norm.load, file=image2, auto_gamma_filter=False)

        # ob
        ob1 = self.data_path + '/tif/ob/ob001.tif'
        ob2 = self.data_path + '/different_format/ob001_4_by_4.tif'
        o_norm = Normalization()
        o_norm.load(file=ob1, auto_gamma_filter=False)
        self.assertRaises(IOError, o_norm.load, file=ob2, auto_gamma_filter=False)
        
        # df
        df1 = self.data_path + '/tif/df/df001.tif'
        df2 = self.data_path + '/different_format/df001_4_by_4.tif'
        o_norm = Normalization()
        o_norm.load(file=df1, auto_gamma_filter=False)
        self.assertRaises(IOError, o_norm.load, file=df2, auto_gamma_filter=False)

    def test_loading_new_data_not_allowed_if_algorithm_already_run(self):
        """assert error raises when loading new data on data already manipulated"""
        # tiff
        sample_tif_file = self.data_path + '/tif/sample/image001.tif'
        ob_tif_file = self.data_path +'/tif/ob/ob001.tif'
        o_norm = Normalization()
        o_norm.load(file=sample_tif_file, data_type='sample', auto_gamma_filter=False)
        o_norm.load(file=ob_tif_file, data_type='ob', auto_gamma_filter=False)
        o_norm.normalization()
        new_sample_tif_file = self.data_path + '/tif/sample/image002.tif'
        self.assertRaises(IOError, o_norm.load, file=new_sample_tif_file)

    def test_loading_tiff_metadata(self):
        """assert metadata of sample are correctly loaded"""
        sample_tif_file = self.data_path + '/tif/sample/image001.tif'
        o_norm = Normalization()
        o_norm.load(file=sample_tif_file, auto_gamma_filter=False)
        metadata = o_norm.data['sample']['metadata']
        metadata_1_expected = 'this is metadata of image001.tif'
        metadata_1_returned = metadata[0][1]
        if isinstance(metadata_1_returned, tuple):
            metadata_1_returned = metadata_1_returned[0]
        self.assertEqual(metadata_1_expected, metadata_1_returned)

class TestGammaFiltering(unittest.TestCase):

    def setUp(self):    
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, '../data/'))

    def test_manuel_gamma_filtered_raises_error_when_array_is_empty(self):
        """assert manual gamma filtering complains when no data provided"""
        path = self.data_path + '/tif/sample'
        o_norm = Normalization()
        o_norm.load(folder=path, data_type='sample', manual_gamma_filter=False, auto_gamma_filter=False)
        data_0 = o_norm.data['sample']['data']
        self.assertRaises(ValueError, o_norm._manual_gamma_filtering)

    def test_gamma_filtered_works(self):
        """assert gamma filtering works"""
        path = self.data_path + '/tif/sample_with_gamma/'

        # gamma filter is True
        o_norm = Normalization()
        o_norm.load(folder=path, data_type='sample', manual_gamma_filter=True, auto_gamma_filter=False)
        _expected_sample = np.ones((5, 5))
        _expected_sample[0, 0] = 0.375
        _expected_sample[:, 2] = 2
        _expected_sample[:, 3] = 3
        _expected_sample[:, 4] = 4
        _returned_sample = o_norm.data['sample']['data']

        self.assertTrue((_expected_sample == _returned_sample).all())
        
        # gamma filter is False
        o_norm = Normalization()
        o_norm.load(folder=path, data_type='sample', manual_gamma_filter=False, auto_gamma_filter=False)
        _expected_sample = np.ones((5, 5))
        _expected_sample[0, 0] = 1000
        _expected_sample[:, 2] = 2
        _expected_sample[:, 3] = 3
        _expected_sample[:, 4] = 4
        _returned_sample = o_norm.data['sample']['data']
        self.assertTrue((_expected_sample == _returned_sample).all())

    def test_auto_gamma_filtered_works(self):
        """assert auto gamma filter works"""
        file_name = self.data_path + '/different_format/image001_with_gamma.tif'
        o_norm = Normalization()
        o_norm.load(file=file_name)

        loaded_data = o_norm.data['sample']['data'][0]

        expected_data = np.zeros((10, 10))
        expected_data = np.asarray(expected_data, np.float32)


        self.assertTrue((expected_data ==  loaded_data).all())