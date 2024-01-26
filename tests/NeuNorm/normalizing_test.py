import pytest
import numpy as np
import os
from PIL import Image

from NeuNorm.normalization import Normalization
from NeuNorm.roi import ROI


class TestNormalization:

    def setup_method(self):
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, '../data/'))

    def test_loading_list_of_files(self):
        """assert initialization using list of files"""
        list_files = [self.data_path + '/tif/sample/image001.tif',
                      self.data_path + '/tif/sample/image002.tif',
                      self.data_path + '/tif/sample/image003.tif']
        o_norm = Normalization()
        o_norm.load(file=list_files, auto_gamma_filter=False)
        data_returned = o_norm.data['sample']['data']
        assert (3, 5, 5) == np.shape(data_returned)
        
    def test_initialization_using_array_with_data(self):
        """assert initialization using arrays with data"""
        sample_01 = self.data_path + '/tif/sample/image001.tif'
        sample_02 = self.data_path + '/tif/sample/image002.tif'
        data = []
        data.append(np.asarray(Image.open(sample_01)))
        data.append(np.asarray(Image.open(sample_02)))
        o_norm = Normalization()
        o_norm.load(data=data, auto_gamma_filter=False)

        data_returned = o_norm.data['sample']['data']
        assert (2, 5, 5) == np.shape(data_returned)
        
    def test_initialization_using_array_with_data_one_by_one(self):
        """assert initialization using arrays with data one by one"""
        o_norm = Normalization()

        sample_01 = self.data_path + '/tif/sample/image001.tif'
        _data = np.asarray(Image.open(sample_01))
        o_norm.load(data=_data, auto_gamma_filter=False)
        
        sample_02 = self.data_path + '/tif/sample/image002.tif'
        _data = np.asarray(Image.open(sample_02))
        o_norm.load(data=_data, auto_gamma_filter=False)

        data_returned = o_norm.data['sample']['data']
        assert (2, 5, 5) == np.shape(data_returned)

    def test_initialization_using_array_with_ob(self):
        """assert initialization using arrays with ob"""
        ob_01 = self.data_path + '/tif/ob/ob001.tif'
        ob_02 = self.data_path + '/tif/ob/ob002.tif'
        data = []
        data.append(np.asarray(Image.open(ob_01)))
        data.append(np.asarray(Image.open(ob_02)))
        o_norm = Normalization()
        o_norm.load(data=data, data_type='ob', auto_gamma_filter=False)

        data_returned = o_norm.data['ob']['data']
        assert (2, 5, 5) == np.shape(data_returned)
        
    def test_normalization_raises_error_if_no_ob_or_sample(self):
        """assert error raises when no ob or sample provided"""
        path = self.data_path + '/tif/sample'
        o_norm = Normalization()
        o_norm.load(folder=path, data_type='sample', auto_gamma_filter=False)
        with pytest.raises(IOError):
            o_norm.normalization()
        
        path = self.data_path + '/tif/ob'
        o_norm = Normalization()
        o_norm.load(folder=path, data_type='ob', auto_gamma_filter=False)
        with pytest.raises(IOError):
            o_norm.normalization()
        
        sample_path = self.data_path + '/tif/sample'
        ob_path = self.data_path + '/tif/ob'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, data_type='sample', auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        assert o_norm.normalization()
 
    def test_normalization_ran_only_once(self):
        """assert normalization is only once if force switch not turn on"""
        sample_tif_folder = self.data_path + '/tif/sample'
        ob_tif_folder = self.data_path + '/tif/ob'
    
        # testing sample with norm_roi
        o_norm = Normalization()
        o_norm.load(folder=sample_tif_folder, auto_gamma_filter=False)
        o_norm.load(folder=ob_tif_folder, data_type='ob', auto_gamma_filter=False)
        roi = ROI(x0=0, y0=0, x1=3, y1=2)
        o_norm.normalization(roi=roi)
        _returned_first_time = o_norm.data['sample']['data'][0]
        o_norm.normalization(roi=roi)
        _returned_second_time = o_norm.data['sample']['data'][0]
        assert (_returned_first_time == _returned_second_time).all()

    def test_normalization_ran_twice_with_force_flag(self):
        """assert normalization can be ran twice using force flag"""
        sample_tif_folder = self.data_path + '/tif/sample'
        ob_tif_folder = self.data_path + '/tif/ob'
    
        # testing sample with norm_roi
        o_norm = Normalization()
        o_norm.load(folder=sample_tif_folder, auto_gamma_filter=False)
        o_norm.load(folder=ob_tif_folder, data_type='ob', auto_gamma_filter=False)
        roi = ROI(x0=0, y0=0, x1=3, y1=2)
        o_norm.normalization(roi=roi)
        _returned_first_time = o_norm.data['sample']['data'][0]
        roi = ROI(x0=0, y0=0, x1=2, y1=3)
        o_norm.normalization(roi=roi, force=True)
        _returned_second_time = o_norm.data['sample']['data'][0]
        assert not ((_returned_first_time == _returned_second_time).all())
  
    def test_normalization_works_if_input_arrays_are_type_int(self):
        """assert normalization works when input arrays are type int"""
        o_norm = Normalization()
        
        sample_01 = self.data_path + '/tif/sample/image001.tif'
        _data = np.asarray(Image.open(sample_01), dtype=int)
        o_norm.load(data=_data, auto_gamma_filter=False)
        
        sample_02 = self.data_path + '/tif/sample/image002.tif'
        _data = np.asarray(Image.open(sample_01), dtype=int)
        o_norm.load(data=_data, auto_gamma_filter=False)

        ob_01 = self.data_path + '/tif/ob/ob001.tif'
        _data = np.asarray(Image.open(ob_01), dtype=int)
        _data[0,0] = 0
        o_norm.load(data=_data, data_type='ob', auto_gamma_filter=False)
    
        ob_02 = self.data_path + '/tif/ob/ob002.tif'
        _data = np.asarray(Image.open(ob_01), dtype=int)
        o_norm.load(data=_data, data_type='ob', auto_gamma_filter=False)

        o_norm.normalization()

    def test_normalization_works(self):
        """assert sample and ob normalization works with and without roi"""
        sample_tif_folder = self.data_path + '/tif/sample'
        ob_tif_folder = self.data_path + '/tif/ob'

        # testing full normalization
        o_norm = Normalization()
        o_norm.load(folder=sample_tif_folder, auto_gamma_filter=False)
        o_norm.load(folder=ob_tif_folder, data_type='ob', auto_gamma_filter=False)
        roi = ROI(x0=0, y0=0, x1=1, y1=4)
        _sample = o_norm.data['sample']['data'][0]
        _ob = o_norm.data['ob']['data'][0]
        _expected_num = _sample / np.mean(_sample[0:4, 0:2])
        _expected_den = _ob / np.mean(_ob[0:4, 0:2])
        _expected = _expected_num /_expected_den
        o_norm.normalization(roi=roi, use_only_sample=False)
        _returned = o_norm.get_normalized_data()[0]
        assert pytest.approx(_expected, 1e-4) == _returned

        # testing sample without norm_roi
        o_norm1 = Normalization()
        o_norm1.load(folder=sample_tif_folder, auto_gamma_filter=False)
        o_norm1.load(folder=ob_tif_folder, data_type='ob', auto_gamma_filter=False)
        _expected = o_norm1.data['sample']['data'][0]
        o_norm1.normalization()
        _returned = o_norm1.data['sample']['data'][0]
        assert (_expected == _returned).all()

        # testing ob with norm_roi
        o_norm = Normalization()
        o_norm.load(folder=sample_tif_folder, auto_gamma_filter=False)
        o_norm.load(folder=ob_tif_folder, data_type='ob', auto_gamma_filter=False)
        norm_roi = ROI(x0=0, y0=0, x1=1, y1=4)
        o_norm.normalization(roi=norm_roi)
        _ob = o_norm.data['ob']['data'][0]
        _expected = _ob / np.mean(_ob[0:5, 0:2])
        _returned = o_norm.data['ob']['data'][0]
        assert pytest.approx(_expected, 1e-4) == _returned

        # testing ob without norm_roi
        o_norm = Normalization()
        o_norm.load(folder=sample_tif_folder, auto_gamma_filter=False)
        o_norm.load(folder=ob_tif_folder, data_type='ob', auto_gamma_filter=False)
        _expected = o_norm.data['ob']['data'][0]
        o_norm.normalization()
        _returned = o_norm.data['ob']['data'][0]
        assert (_expected == _returned).all()
  
    def test_normalization_with_same_ob_and_sample_but_forced_mean_ob(self):
        """assert normalization with same ob and sample number of files force to use mean ob when flag used"""
        samples_path =  self.data_path + '/tif/sample/' # 3 files
        ob1 = self.data_path + '/tif/ob/ob001.tif'
        ob2 = self.data_path + '/tif/ob/ob002.tif'
        ob3 = self.data_path + '/tif/ob/ob003.tif'
        o_norm = Normalization()
        o_norm.load(folder=samples_path, auto_gamma_filter=False)
        o_norm.load(file=[ob1, ob2, ob3], data_type='ob', auto_gamma_filter=False)
        o_norm.normalization(force_mean_ob=True)
        expected_normalized_array = np.ones((5, 5))
        expected_normalized_array[:, 2] = 2
        expected_normalized_array[:, 3] = 3
        expected_normalized_array[:, 4] = 4
        assert (o_norm.data['normalized'][0] == expected_normalized_array).all()

    def test_normalization_with_same_ob_and_sample_but_forced_median_ob(self):
        """assert normalization with same ob and sample number of files force to use mean ob when flag used"""
        samples_path =  self.data_path + '/tif/sample/' # 3 files
        ob1 = self.data_path + '/tif/ob/ob001.tif'
        ob2 = self.data_path + '/tif/ob/ob002.tif'
        ob3 = self.data_path + '/tif/ob/ob003.tif'
        o_norm = Normalization()
        o_norm.load(folder=samples_path, auto_gamma_filter=False)
        o_norm.load(file=[ob1, ob2, ob3], data_type='ob', auto_gamma_filter=False)
        o_norm.normalization(force_mean_ob=True)
        expected_normalized_array = np.ones((5, 5))
        expected_normalized_array[:, 2] = 2
        expected_normalized_array[:, 3] = 3
        expected_normalized_array[:, 4] = 4
        assert (o_norm.data['normalized'][0] == expected_normalized_array).all()

    def test_normalization_with_fewer_ob_than_sample_works(self):
        """assert normalization works when number of ob and sample is different"""
        samples_path = self.data_path + '/tif/sample/'  # 3 files
        ob1 = self.data_path + '/tif/ob/ob001.tif'
        ob2 = self.data_path + '/tif/ob/ob002.tif' 
        df1 = self.data_path + '/tif/df/df001.tif'
        o_norm = Normalization()
        o_norm.load(folder=samples_path, auto_gamma_filter=False)
        o_norm.load(file=[ob1, ob2], data_type='ob', auto_gamma_filter=False)
        o_norm.load(file=df1, data_type='df', auto_gamma_filter=False)
        o_norm.df_correction()
        o_norm.normalization()
        expected_normalized_array = np.zeros((5, 5))
        expected_normalized_array[0, 0] = 1
        assert (o_norm.data['normalized'] == expected_normalized_array).all()

    def test_nbr_data_files_same_after_normalization_by_list_roi(self):
        """assert the number of data files is the same after normalization by a list of ROI"""
        samples_path =  self.data_path + '/tif/sample/' # 3 files
        ob1 = self.data_path + '/tif/ob/ob001.tif'
        ob2 = self.data_path + '/tif/ob/ob002.tif'
        o_norm = Normalization()
        o_norm.load(folder=samples_path, auto_gamma_filter=False)
        o_norm.load(file=[ob1, ob2], data_type='ob', auto_gamma_filter=False)
        _roi1 = ROI(x0=0,y0=0,x1=2,y1=2)
        _roi2 = ROI(x0=1,y0=1,x1=3,y1=3)
        _list_roi = [_roi1, _roi2]
        nbr_data_before = len(o_norm.data['sample']['data'])
        o_norm.normalization(roi=_list_roi)
        nbr_data_after = len(o_norm.data['sample']['data'])
        assert nbr_data_after == nbr_data_before

    def test_normalization_works_with_only_1_df(self):
        """assert using 1 df in normalization works"""
        samples_path =  self.data_path + '/tif/sample/' # 3 files
        ob1 = self.data_path + '/tif/ob/ob001.tif'
        ob2 = self.data_path + '/tif/ob/ob002.tif'
        df1 = self.data_path + '/tif/df/df001.tif'
        o_norm = Normalization()
        o_norm.load(folder=samples_path, auto_gamma_filter=False)
        o_norm.load(file=[ob1, ob2], data_type='ob', auto_gamma_filter=False)
        o_norm.load(file=df1, data_type='df', auto_gamma_filter=False)
        o_norm.df_correction()
        _roi1 = ROI(x0=0, y0=0, x1=2, y1=2)
        _roi2 = ROI(x0=1, y0=1, x1=3, y1=3)
        _list_roi = [_roi1, _roi2]
        nbr_data_before = len(o_norm.data['sample']['data'])
        o_norm.normalization(roi=_list_roi)
        nbr_data_after = len(o_norm.data['sample']['data'])
        assert nbr_data_after == nbr_data_before

    def test_normalization_works_with_2_dfs(self):
        """assert using 2 df in normalization works"""
        samples_path = self.data_path + '/tif/sample/'  # 3 files
        ob1 = self.data_path + '/tif/ob/ob001.tif'
        ob2 = self.data_path + '/tif/ob/ob002.tif'
        df1 = self.data_path + '/tif/df/df001.tif'
        df2 = self.data_path + '/tif/df/df002.tif'
        o_norm = Normalization()
        o_norm.load(folder=samples_path, auto_gamma_filter=False)
        o_norm.load(file=[ob1, ob2], data_type='ob', auto_gamma_filter=False)
        o_norm.load(file=[df1, df2], data_type='df', auto_gamma_filter=False)
        o_norm.df_correction()
        _roi1 = ROI(x0=0, y0=0, x1=2, y1=2)
        _roi2 = ROI(x0=1, y0=1, x1=3, y1=3)
        _list_roi = [_roi1, _roi2]
        nbr_data_before = len(o_norm.data['sample']['data'])
        o_norm.normalization(roi=_list_roi)
        nbr_data_after = len(o_norm.data['sample']['data'])
        assert nbr_data_after == nbr_data_before

    def test_normalization_works_with_1_roi_given_as_a_list(self):
        """Make sure the normalization works when 2 ROI are used"""
        sample = self.data_path + '/fits/test_roi/sample.fits'
        ob = self.data_path + '/fits/test_roi/ob.fits'
        _roi = ROI(x0=0, y0=0, x1=1, y1=2)
        list_roi = [_roi]
        o_norm = Normalization()
        o_norm.load(file=sample, auto_gamma_filter=False)
        o_norm.load(file=ob, data_type='ob', auto_gamma_filter=False)
        o_norm.normalization(roi=list_roi)
        normalized_data = o_norm.get_normalized_data()[0]

        expected_normalized_data = np.ones([5, 4])
        expected_normalized_data[0:2, 2:4] = .5
        expected_normalized_data[3:5, 0:2] = 1.11111111

        height, width = np.shape(expected_normalized_data)
        for _h in np.arange(height):
            for _w in np.arange(width):
                assert expected_normalized_data[_h, _w] == pytest.approx(normalized_data[_h, _w], 1e-5)


class TestDFCorrection:

    def setup_method(self):
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, '../data/'))  
        
    def test_df_correction_when_no_df(self):
        """assert sample and ob are inchanged if df is empty"""

        # sample
        path = self.data_path + '/tif/sample'
        o_norm = Normalization()
        o_norm.load(folder=path, data_type='sample', auto_gamma_filter=False)
        data_before = o_norm.data['sample']['data'][0]
        o_norm.df_correction()
        data_after = o_norm.data['sample']['data'][0]
        assert (data_before == data_after).all()
        
        #ob
        path = self.data_path + '/tif/ob'
        o_norm = Normalization()
        o_norm.load(folder=path, data_type='ob', auto_gamma_filter=False)
        data_before = o_norm.data['ob']['data'][0]
        o_norm.df_correction()
        data_after = o_norm.data['ob']['data'][0]
        assert (data_before == data_after).all()
        
    def test_df_fails_when_not_identical_data_shape(self):
        o_norm = Normalization()
        sample_1 = np.ones([5,5])
        df_1 = np.ones([6,6])
        o_norm.data['sample']['data'] = sample_1
        o_norm.data['df']['data'] = df_1
        with pytest.raises(IOError):
            o_norm.df_correction()
        
        o_norm = Normalization()
        ob_1 = np.ones([6,6])
        o_norm.data['ob']['data'] = sample_1
        o_norm.data['df']['data'] = ob_1
        with pytest.raises(IOError):
            o_norm.df_correction()

    def test_df_averaging_only_run_the_first_time(self):
        """assert the average_df is only run the first time the df_correction is run"""
        sample_path = self.data_path + '/tif/sample/'
        ob_path = self.data_path + '/tif/ob/'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        df_file_1 = self.data_path + '/tif/df/df002.tif'
        df_file_2 = self.data_path + '/tif/df/df003.tif'
        o_norm.load(file=df_file_1, data_type='df', auto_gamma_filter=False)
        o_norm.load(file=df_file_2, data_type='df', auto_gamma_filter=False)
    
        df_average_data = o_norm.data['df']['data_average']
        assert not df_average_data
    
        #sample
        o_norm.df_correction()
        df_average_data = o_norm.data['df']['data_average']
        assert df_average_data.size != 0
    
        #ob
        o_norm.df_correction()
        expected_df_average = df_average_data
        df_average = o_norm.data['df']['data_average']
        assert (expected_df_average == df_average).all()

    def test_df_correction(self):
        """assert df corrction works"""
        sample_path = self.data_path + '/tif/sample/'
        ob_path = self.data_path + '/tif/ob/'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        df_file_1 = self.data_path + '/tif/df/df002.tif'
        df_file_2 = self.data_path + '/tif/df/df003.tif'
        o_norm.load(file=df_file_1, data_type='df', auto_gamma_filter=False)
        o_norm.load(file=df_file_2, data_type='df', auto_gamma_filter=False)
        
        #sample
        o_norm.df_correction()
        _expected_data = np.zeros([5,5])
        _expected_data[:,2] = 1
        _expected_data[:,3] = 2
        _expected_data[:,4] = 3       
        _sample_data = o_norm.data['sample']['data'][0]
        assert (_expected_data == o_norm.data['sample']['data'][0]).all()
        
        #ob
        _expected_data = np.zeros([5,5])
        _ob_data = o_norm.data['ob']['data'][0]
        assert (_expected_data == _ob_data).all()

    def test_df_correction_locked_when_run_twice_without_force_flag(self):
        """assert df corrction run only one time if force flag is False"""
        sample_path = self.data_path + '/tif/sample/'
        ob_path = self.data_path + '/tif/ob/'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        df_file_1 = self.data_path + '/tif/df/df002.tif'
        df_file_2 = self.data_path + '/tif/df/df003.tif'
        o_norm.load(file=df_file_1, data_type='df', auto_gamma_filter=False)
        o_norm.load(file=df_file_2, data_type='df', auto_gamma_filter=False)
        
        # first iteration
        o_norm.df_correction()
        _sample_first_run = o_norm.data['sample']['data'][0]
        _ob_first_run = o_norm.data['ob']['data'][0]
        
        # second iteration
        o_norm.df_correction()
        _sample_second_run = o_norm.data['sample']['data'][0]
        _ob_second_run = o_norm.data['ob']['data'][0]
         
        assert (_sample_first_run == _sample_second_run).all()
        assert (_ob_first_run == _ob_second_run).all()
         
    def test_df_correction_run_twice_with_force_flag(self):
        """assert df corrction run more than once with force flag"""
        sample_path = self.data_path + '/tif/sample/'
        ob_path = self.data_path + '/tif/ob/'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        df_file_1 = self.data_path + '/tif/df/df002.tif'
        df_file_2 = self.data_path + '/tif/df/df003.tif'
        o_norm.load(file=df_file_1, data_type='df', auto_gamma_filter=False)
        o_norm.load(file=df_file_2, data_type='df', auto_gamma_filter=False)
        
        # first iteration
        o_norm.df_correction()
        _sample_first_run = o_norm.data['sample']['data'][0]
        _ob_first_run = o_norm.data['ob']['data'][0]
        _average_df = o_norm.data['df']['data_average']
        
        # second iteration
        o_norm.df_correction(force=True)
        _sample_second_run = o_norm.data['sample']['data'][0]
        _ob_second_run = o_norm.data['ob']['data'][0]

        # expected
        _expected_sample_after_second_run = _sample_first_run - _average_df
        _expected_ob_after_second_run = _ob_first_run - _average_df
         
        assert (_sample_second_run == _expected_sample_after_second_run).all()
        assert (_ob_second_run == _expected_ob_after_second_run).all()
         
        
class TestApplyingROI:

    def setup_method(self):
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, '../data/'))       
        
    def test_roi_type_in_normalization(self):
        """assert error is raised when type of norm roi are not ROI in normalization"""
        sample_tif_file = self.data_path + '/tif/sample/image001.tif'
        ob_tif_file = self.data_path + '/tif/ob/ob001.tif'
        o_norm = Normalization()
        o_norm.load(file=sample_tif_file, data_type='sample', auto_gamma_filter=False)
        o_norm.load(file=ob_tif_file, data_type='ob', auto_gamma_filter=False)
        roi = {'x0':0, 'y0':0, 'x1':2, 'y1':2}
        with pytest.raises(ValueError):
            o_norm.normalization(roi=roi)
        
    def test_roi_fit_images(self):
        """assert norm roi do fit the images"""
        sample_tif_file = self.data_path + '/tif/sample/image001.tif'
        ob_tif_file = self.data_path + '/tif/ob/ob001.tif'
        
        # x0 < 0 or x1 > image_width
        o_norm = Normalization()
        o_norm.load(file=sample_tif_file, data_type='sample', auto_gamma_filter=False)
        o_norm.load(file=ob_tif_file, data_type='ob', auto_gamma_filter=False)
        roi = ROI(x0=0, y0=0, x1=20, y1=4)
        with pytest.raises(ValueError):
            o_norm.normalization(roi=roi)
       
        o_norm = Normalization()
        o_norm.load(file=sample_tif_file, data_type='sample', auto_gamma_filter=False)
        o_norm.load(file=ob_tif_file, data_type='ob', auto_gamma_filter=False)
        roi = ROI(x0=-1, y0=0, x1=4, y1=4)
        with pytest.raises(ValueError):
            o_norm.normalization(roi=roi)
        
        # y0 < 0 or y1 > image_height
        o_norm = Normalization()
        o_norm.load(file=sample_tif_file, data_type='sample', auto_gamma_filter=False)
        o_norm.load(file=ob_tif_file, data_type='ob', auto_gamma_filter=False)
        roi = ROI(x0=0, y0=-1, x1=4, y1=4)
        with pytest.raises(ValueError):
            o_norm.normalization(roi=roi)

        # y1>image_height
        o_norm = Normalization()
        o_norm.load(file=sample_tif_file, data_type='sample', auto_gamma_filter=False)
        o_norm.load(file=ob_tif_file, data_type='ob', auto_gamma_filter=False)
        roi = ROI(x0=0, y0=0, x1=4, y1=20)
        with pytest.raises(ValueError):
            o_norm.normalization(roi=roi)

    def test_error_raised_when_data_shape_of_different_type_do_not_match(self):
        """assert shape of data must match to allow normalization"""
        
        # sample and ob
        image1 = self.data_path + '/tif/sample/image001.tif'
        ob1 = self.data_path + '/different_format/ob001_4_by_4.tif'
        o_norm = Normalization()
        o_norm.load(file=image1, auto_gamma_filter=False)
        o_norm.load(file=ob1, data_type='ob', auto_gamma_filter=False)
        with pytest.raises(ValueError):
            o_norm.normalization()

        # sample, ob and df
        image1 = self.data_path + '/tif/sample/image001.tif'
        ob1 = self.data_path + '/tif/ob/ob001.tif'
        df1 = self.data_path + '/different_format/df001_4_by_4.tif'
        o_norm = Normalization()
        o_norm.load(file=image1, auto_gamma_filter=False)
        o_norm.load(file=ob1, data_type='ob', auto_gamma_filter=False)
        o_norm.load(file=df1, data_type='df', auto_gamma_filter=False)
        with pytest.raises(ValueError):
            o_norm.normalization()

    def test_full_normalization_sample_with_several_roi(self):
        """assert the full normalization works with several roi selected"""
        sample_path = self.data_path + '/tif/sample'
        ob_path = self.data_path + '/tif/ob'
        df_path = self.data_path + '/tif/df'

        # without DF
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        # o_norm.load(folder=df_path, data_type='df', auto_gamma_filter=False)
        _roi_1 = ROI(x0=0, y0=0, x1=1, y1=1)
        _roi_2 = ROI(x0=0, y0=0, x1=1, y1=1)
        o_norm.normalization(roi=[_roi_1, _roi_2])
        _norm_returned = o_norm.data['normalized'][0]
        _norm_expected = np.ones((5, 5))
        _norm_expected[:, 2] = 2
        _norm_expected[:, 3] = 3
        _norm_expected[:, 4] = 4

        assert _norm_expected[0, 0] == pytest.approx(_norm_returned[0, 0], 1e-8)

        # with DF
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        o_norm.load(folder=df_path, data_type='df', auto_gamma_filter=False)
        _roi_1 = ROI(x0=0, y0=0, x1=1, y1=1)
        _roi_2 = ROI(x0=0, y0=0, x1=1, y1=1)
        o_norm.normalization(roi=[_roi_1, _roi_2])
        _norm_returned = o_norm.data['normalized'][0]
        _norm_expected = np.ones((5, 5))
        _norm_expected[:, 2] = 2
        _norm_expected[:, 3] = 3
        _norm_expected[:, 4] = 4

        assert _norm_expected[0, 0] == pytest.approx(_norm_returned[0, 0], 1e-8)

        sample_path = self.data_path + '/tif/special_sample'
        ob_path = self.data_path + '/tif/special_ob'

        # without DF
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        # o_norm.load(folder=df_path, data_type='df', auto_gamma_filter=False)
        _roi_1 = ROI(x0=0, y0=0, x1=0, y1=0)
        _roi_2 = ROI(x0=4, y0=4, x1=4, y1=4)
        o_norm.normalization(roi=[_roi_1, _roi_2])
        _norm_returned = o_norm.data['normalized'][0]
        _norm_expected = np.ones((5, 5))
        _norm_expected[:, 2] = 2
        _norm_expected[:, 3] = 3
        _norm_expected[:, 4] = 4

        assert _norm_expected[0, 0] == pytest.approx(_norm_returned[0, 0], 1e-8)

    def test_full_normalization_sample_with_one_roi(self):
        """assert the full normalization works with several roi selected"""
        sample_path = self.data_path + '/tif/sample'
        ob_path = self.data_path + '/tif/ob'
        df_path = self.data_path + '/tif/df'

        # without DF
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        _roi_1 = ROI(x0=0, y0=0, x1=1, y1=1)
        o_norm.normalization(roi=[_roi_1])
        _norm_returned = o_norm.data['normalized'][0]
        _norm_expected = np.ones((5, 5))
        _norm_expected[:, 2] = 2
        _norm_expected[:, 3] = 3
        _norm_expected[:, 4] = 4

        nbr_col, nbr_row = np.shape(_norm_expected)
        for _col in np.arange(nbr_col):
            for _row in np.arange(nbr_row):
                assert _norm_expected[_col, _row] == _norm_returned[_col, _row]

        # with DF
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        o_norm.load(folder=df_path, data_type='df', auto_gamma_filter=False)
        _roi_1 = ROI(x0=0, y0=0, x1=1, y1=1)
        o_norm.normalization(roi=[_roi_1])
        _norm_returned = o_norm.data['normalized'][0]
        _norm_expected = np.ones((5, 5))
        _norm_expected[:, 2] = 2
        _norm_expected[:, 3] = 3
        _norm_expected[:, 4] = 4

        nbr_col, nbr_row = np.shape(_norm_expected)
        for _col in np.arange(nbr_col):
            for _row in np.arange(nbr_row):
                assert _norm_expected[_col, _row] == _norm_returned[_col, _row]

    def test_full_normalization_sample_divide_by_ob_works(self):
        """assert the full normalization works (when sample is divided by ob)"""

        # without normalization roi
        sample_path = self.data_path + '/tif/sample'
        ob_path = self.data_path + '/tif/ob'
        df_path = self.data_path + '/tif/df'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        o_norm.load(folder=df_path, data_type='df', auto_gamma_filter=False)
        o_norm.normalization()
        _norm_expected = np.ones((5, 5))
        _norm_expected[:, 2] = 2
        _norm_expected[:, 3] = 3
        _norm_expected[:, 4] = 4
        _norm_returned = o_norm.data['normalized']
        assert (_norm_expected == _norm_returned).all()
        
        # with normalization roi
        sample_path = self.data_path + '/tif/sample'
        ob_path = self.data_path + '/tif/ob'
        df_path = self.data_path + '/tif/df'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        o_norm.load(folder=df_path, data_type='df', auto_gamma_filter=False)
        _roi = ROI(x0=0, y0=1, x1=1, y1=4)
        o_norm.normalization(roi=_roi)
        _norm_expected = np.ones((5, 5))
        # _norm_expected.fill(1)
        _norm_expected[:, 2] = 2
        _norm_expected[:, 3] = 3
        _norm_expected[:, 4] = 4
        _norm_returned = o_norm.data['normalized'][0]
        assert (_norm_expected == _norm_returned).all()
        
    def test_various_data_type_correctly_returned(self):
        """assert normalized, sample, ob and df data are correctly returned"""
        sample_path = self.data_path + '/tif/sample'
        ob_path = self.data_path + '/tif/ob'
        df_path = self.data_path + '/tif/df'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, auto_gamma_filter=False)
        o_norm.load(folder=ob_path, data_type='ob', auto_gamma_filter=False)
        o_norm.load(folder=df_path, data_type='df', auto_gamma_filter=False)
        
        # sample
        _data_expected = o_norm.data['sample']['data'][0]
        _data_returned = o_norm.get_sample_data()[0]
        assert (_data_expected == _data_returned).all()
        
        # ob
        _ob_expected = o_norm.data['ob']['data']
        _ob_returned = o_norm.get_ob_data()[0]
        assert (_ob_expected == _ob_returned).all()
        
        # df
        _df_expected = o_norm.data['df']['data']
        _df_returned = o_norm.get_df_data()
        assert _df_expected == _df_returned
        
        # normalized is empty before normalization
        assert o_norm.get_normalized_data() is None
        
        # run normalization
        o_norm.normalization()
        
        _norm_expected = o_norm.data['normalized'][0]
        _norm_returned = o_norm.get_normalized_data()[0]
        assert (_norm_expected == _norm_returned).all()

    def test_normalization_using_roi_of_sample_only(self):
        """testing features that will normalized the data according to a ROI of the sample
        this feature require at least 1 ROI. the average counts of the ROI for each image will be used as
        "open beam" counts and each pixel will be divided by this value
        """
        sample_file = self.data_path + '/tif/special_sample/image_0001_roi_no_ob.tiff'
        o_norm = Normalization()
        o_norm.load(file=sample_file, auto_gamma_filter=False)

        with pytest.raises(ValueError):
            o_norm.normalization(use_only_sample=True)
        del o_norm

        o_norm = Normalization()
        o_norm.load(file=sample_file, auto_gamma_filter=False)
        _roi = ROI(x0=0, y0=0, x1=0, y1=0)
        o_norm.normalization(roi=_roi, use_only_sample=True)

        _norm_expected = np.ones((5, 5))
        _norm_expected[1:, 0] = 1./10
        _norm_expected[:, 1] = 2./10
        _norm_expected[:, 2] = 3./10
        _norm_expected[:, 3] = 4./10
        _norm_expected[:-1, 4] = 5./10
        _norm_returned = o_norm.get_normalized_data()[0]

        nbr_col, nbr_row = np.shape(_norm_expected)
        for _col in np.arange(nbr_col):
            for _row in np.arange(nbr_row):
                assert _norm_expected[_col, _row] == pytest.approx(_norm_returned[_col, _row], 1e-5)
