*****************************
Using library from a Notebook
*****************************

If you run the library from a notebook, you have the option to display a progress bar showing you the progress 
of the loading or normalization processes (NB: progress bar will not show up when loading one file at a time)

>>> sample_folder = '/Users/me/sample/'
>>> o_norm = Normalization()
>>> o_norm.load(folder=sample_folder, notebook=True)

.. image:: _static/progress_bar_loading.png
    :align: center
    :alt: typical attenuation plot


or during normalization

>>> o_norm.normalization(notebook=True)

.. image:: _static/progress_bar_normalization.png
    :align: center
    :alt: typical attenuation plot
