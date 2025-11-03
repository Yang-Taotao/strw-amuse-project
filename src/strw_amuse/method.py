"""
Methods for STRW-AMUSE project.
"""
import amuse
import amuse._version

def amuse_version_check() -> None:
    '''Display the version of AMUSE.'''
    print(f'AMUSE on v: {amuse._version.__version__}')
    return None
