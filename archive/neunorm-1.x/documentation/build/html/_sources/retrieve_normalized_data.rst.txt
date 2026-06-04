************************
Retrieve Normalized Data
************************

The sample/OB normalized data can be recovered this way

>>> normalized_data = neunorm.data['normalized']

You can retrieve the data using either this way

>>> sample = o_norm.data['sample']['data']
>>> ob = o_norm.data['ob']['data']
>>> df = o_norm.data['df']['data']
>>> norm = o_norm.data['normalization']

or

>>> sample = o_norm.get_sample_data()
>>> ob = o_norm.get_ob_data()
>>> df = o_norm.get_df_data()
>>> normalized = o_norm.get_normalized_data()
