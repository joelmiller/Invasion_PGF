#!/usr/bin/env python

r'''
Setup script for Invasion_PGF

to install from this script, run

    python setup.py install

Alternately, you can install with pip.
    
    pip install Invasion_PGF
    
If this is a "release candidate" (has an "rc" in the version name below), then
pip will download the previous version - see the download_url below.
'''

from distutils.core import setup

setup(name='Invasion_PGF',
      packages = ['Invasion_PGF'], 
      version='0.90.0',  #http://semver.org/
      description = 'Methods based on probability generating functions (PGFs) for studying early phase of invasive processes',
      author = 'Joel C. Miller',
      author_email = 'joel.c.miller.research@gmail.com',
      url = 'https://github.com/joelmiller/Invasion_PGF',
      download_url = 'https://github.com/joelmiller/Invasion_PGF/archive/0.90.0.tar.gz',
      keywords = ['Invasive processes', 'Infectious disease'],
      dependency_links = ['https://github.com/WarrenWeckesser/odeintw/tarball/master#egg=package-0.0.1']
      install_requires = [
          'numpy'
          ],
      )
