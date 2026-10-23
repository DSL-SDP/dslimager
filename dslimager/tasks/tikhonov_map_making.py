"""Tikhonov-regularized map-making task.

Solve the linear visibility model ``V = B T + n`` for the sky map ``T`` using
Tikhonov regularization,

.. math::

    T = (B^\\dagger B + \\epsilon I)^{-1} B^\\dagger V,

which is evaluated through the pre-computed thin SVD ``B = U diag(s) Vh`` as

.. math::

    T = Vh^\\dagger diag(s / (s^2 + \\epsilon)) U^\\dagger V.

The regularization parameter is given relative to the largest singular value,
i.e. ``eps_abs = eps * s_max^2``, so that it is independent of the units and
scaling of ``B``.  The solution is written as a RING-ordered HEALPix map.

Inheritance diagram
-------------------

.. inheritance-diagram:: TikhonovMapMaking
   :parts: 2

"""

import os

import h5py
import healpy as hp
import numpy as np
from caput import mpiutil
from dslpipe.pipeline.pipeline import OneAndOne
from dslpipe.utils.logging import get_logger

logger = get_logger(__name__)

# Data shipped with the package, used as defaults.
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


class TikhonovMapMaking(OneAndOne):
    """Tikhonov-regularized map-making from visibilities.

    The input file (``input_files``) is the visibility/response-matrix file
    containing ``V_data`` and the metadata attributes; the SVD file
    (``svd_file``) supplies ``U``, ``s`` and ``Vh``.
    """

    params_init = {
                    'eps': 3e-9, # Tikhonov parameter, relative to smax^2
                    'input_files': os.path.join(_DATA_DIR, 'B_matrix_nside32.hdf5'),
                    'svd_file': os.path.join(_DATA_DIR, 'svd_components.hdf5'),
                  }

    prefix = 'mm_'

    def read_input(self):
        """Read the visibility data and geometry from disk."""

        if len(self.input_files) == 0:
            raise ValueError('No input file is given for the map-making task')

        file_name = self.input_files[0]
        if mpiutil.rank0:
            logger.info(f'Reading visibilities from {file_name}')

        with h5py.File(file_name, 'r') as f:
            visibilities = f['V_data'][:]
            nside = int(f.attrs['nside_q'])
            lambda_m = f.attrs.get('lambda_m', None)

        return {'visibilities': visibilities, 'nside': nside,
                'lambda_m': lambda_m}

    def process(self, input):
        """Compute the Tikhonov-regularized map."""

        visibilities = input['visibilities']
        nside = input['nside']

        eps_rel = float(self.params['eps'])
        if eps_rel <= 0:
            raise ValueError(f'eps must be positive, got {eps_rel}')

        if mpiutil.rank0:
            logger.info(f'Tikhonov map-making with eps = {eps_rel:.3e} (relative to smax^2)')

        # Load the SVD of B.
        svd_file = self.params['svd_file']
        if not os.path.exists(svd_file):
            raise FileNotFoundError(f'SVD file not found: {svd_file}')
        if mpiutil.rank0:
            logger.info(f'Reading SVD components from {svd_file}')
        with h5py.File(svd_file, 'r') as f:
            u = f['U'][:]
            s = f['s'][:]
            vh = f['Vh'][:]

        if mpiutil.rank0:
            logger.info(f'Singular values: smax={s[0]:.4e}, smin={s[-1]:.4e}, '
                        f'cond={s[0] / s[-1]:.3e}')

        # T = Vh^H diag(s / (s^2 + eps)) U^H V.
        c = u.conj().T @ visibilities
        eps_abs = eps_rel * s[0]**2
        filt = s / (s**2 + eps_abs)
        t_map = vh.conj().T @ (filt * c)

        # Residual of the fitted model, computed without keeping B in memory.
        t_model = u @ (s * (vh @ t_map))
        residual = np.linalg.norm(t_model - visibilities) / np.linalg.norm(visibilities)
        imag_ratio = np.abs(t_map.imag).max() / np.abs(t_map.real).max()

        if mpiutil.rank0:
            logger.info(f'Absolute eps={eps_abs:.3e}, residual={residual:.3e}, '
                        f'|Im|/|Re|={imag_ratio:.1e}')

        # B columns are NEST ordered; output maps are RING ordered.
        t_ring = hp.reorder(t_map.real, n2r=True)

        return {'T_recon': t_ring, 'singular_values': s, 'nside': nside,
                'lambda_m': input['lambda_m'], 'eps_rel': eps_rel,
                'eps_abs': eps_abs, 'residual': residual,
                'imag_ratio': imag_ratio}

    def write_output(self, output):
        """Write the reconstructed map and metadata to disk."""

        if mpiutil.rank0:
            file_name = self.output_files[0]
            logger.info(f'Writing reconstructed map to {file_name}')

            with h5py.File(file_name, 'w') as f:
                f.create_dataset('T_recon', data=output['T_recon'])
                f.create_dataset('singular_values', data=output['singular_values'])

                f.attrs['nside'] = output['nside']
                f.attrs['order'] = 'RING'
                f.attrs['unit'] = 'K'
                f.attrs['method'] = ('Tikhonov: T=(B^H B + eps I)^-1 B^H V '
                                     'via SVD filter s/(s^2+eps)')
                f.attrs['eps'] = output['eps_rel']
                f.attrs['eps_abs'] = output['eps_abs']
                f.attrs['residual'] = output['residual']
                f.attrs['imag_ratio'] = output['imag_ratio']
                if output['lambda_m'] is not None:
                    f.attrs['lambda_m'] = output['lambda_m']
                f.attrs['history'] = self.history
