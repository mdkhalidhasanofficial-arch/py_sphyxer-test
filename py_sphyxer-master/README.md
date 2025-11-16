### py_sphyxer

#### demo repo for sphinx documentation setup


#### how to make sphinx documentation
sequence of commands to generate spinx docs ...

1. $ pip install sphinx
  

2. ensure project directory structure is similar to this:

    py_sphyxer
    - data
    - docs
    - reports
    - src
      - \_\_init\_\_.py
      - data
        - \_\_init\_\_.py
        - read_data.py
      - ticker
        - \_\_init\_\_.py
        - tkr.py
        - tkr_utils.py
      - models
        - \_\_init\_\_.py
        - base_model.py
        - arima.py
      - utils.py
    - test
  

3. commands (from /docs directory)\


    user:py_sphyxers/docs$ sphinx-quickstart
    - separate source and build directories?: y

  

4. from project main directory\
`user:py_sphyxers$ sphinx-apidoc -o docs src`
  

5. update /docs/source/conf.py\


    extensions = [  
                'sphinx.ext.autodoc',  
                'sphinx.ext.napoleon',  
                'sphinx.ext.viewcode'  
                ]

    html_theme = 'sphinx_rtd_theme'

6. add elements to index.rst file, e.g.,:\
`   introduction
   usage
   examples
   src`
  

7. from directory: project/docs ...\
`user:py_sphyxers/docs$ make html`
  

8. to re-build after repo changes:\
`user:py_sphyxers/docs$ make clean
user:py_sphyxers/docs$ make html`


how to make html viewable on github:
???


which files are necessary / unnecesaary to commit to the repo?
- docs/build/html - yes
- docs/build/doctrees - ?
- docs/source - ?
- docs/make.bat, Makefile, src.rst, src.data.rst, etc - ?