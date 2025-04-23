import os
import numpy as np
import pandas as pd
import desispec.io
from desispec.coaddition import coadd_cameras
from astropy.io import fits
import h5py
import random

# Combine QSO + LAE (ELG) to lenses

np.random.seed(123)  # Set seed for NumPy's random generator
random.seed(123)  # Set seed for Python's random module


# Function to extract data from HDF5 files
def extract_data_from_hdf5(file_path):
    with h5py.File(file_path, "r") as hdf5_file:
        classifier = hdf5_file["classifier"][:].astype(str)
        name = hdf5_file["name"][:].astype(str)
        redshift = hdf5_file["redshift"][:]
        intflux = hdf5_file["intflux"][:]
        spectra = hdf5_file["spectrum"][:]
    return classifier, name, redshift, intflux, spectra


# Function to process and modify QSO fluxes
def process_qso_fluxes(fits_directory, hdf5_directory, output_directory, lens_percentage=.1, num_files=None):
    # List all HDF5 files in the directory and sort them by name
    hdf5_files = sorted([f for f in os.listdir(hdf5_directory) if f.endswith(".h5")])

    # Initialize arrays to store data
    all_classifierLAE, all_namesLAE, all_redshiftsLAE, all_intfluxLAE, all_spectraLAE = [], [], [], [], []

    # Iterate over HDF5 files in the directory
    for filename in hdf5_files:
        file_path = os.path.join(hdf5_directory, filename)
        print(f"add LAE spectra {filename}")

        # Extract data from the HDF5 file without filtering
        classifier, name, redshift, intflux, spectra = extract_data_from_hdf5(file_path)

        # Append the data to the respective arrays
        all_classifierLAE.extend(classifier)
        all_namesLAE.extend(name)
        all_redshiftsLAE.extend(redshift)
        all_intfluxLAE.extend(intflux)
        all_spectraLAE.extend(spectra)

    all_classifierLAE, all_namesLAE, all_redshiftsLAE, all_intfluxLAE, all_spectraLAE = np.array(all_classifierLAE), np.array(all_namesLAE), np.array(all_redshiftsLAE), np.array(all_intfluxLAE), np.array(all_spectraLAE)

    print(f"Total LAE entries: {len(all_redshiftsLAE)}")

    # List all QSO FITS files in the directory
    fits_files = sorted([f for f in os.listdir(fits_directory) if f.endswith(".fits")])

    if num_files is not None:
        fits_files = fits_files[:num_files]

    # Initialize arrays to store aggregated QSO data
    all_qso_fluxes = []
    all_targetids_qso = []
    all_redshifts_qso = []
    all_ivar = []
    file_qso_indices = []

    # Iterate over FITS files to aggregate data
    for fits_file in fits_files:
        file_path = os.path.join(fits_directory, fits_file)

        # Read the stacked spectra
        qsos = desispec.io.read_spectra(file_path)

        # Access the extra catalog data
        extra_catalog = qsos.extra_catalog

        # Extract TARGETID and Z arrays
        targetids_qso = extra_catalog['TARGETID']
        redshifts_qso = extra_catalog['Z']

        # Combine camera spectra for QSOs
        qso_fluxes = []
        qso_ivar = []
        for i2 in range(len(qsos)):
            coadd = coadd_cameras(qsos[i2])
            qso_fluxes.append(coadd.flux['brz'][0])
            qso_ivar.append(coadd.ivar['brz'][0])

        # Append data to the aggregated arrays
        all_qso_fluxes.extend(qso_fluxes)
        all_ivar.extend(qso_ivar)
        all_targetids_qso.extend(targetids_qso)
        all_redshifts_qso.extend(redshifts_qso)
        file_qso_indices.append(len(all_qso_fluxes))

    all_qso_fluxes = np.array(all_qso_fluxes)
    all_ivar = np.array(all_ivar)
    all_targetids_qso = np.array(all_targetids_qso)
    all_redshifts_qso = np.array(all_redshifts_qso)

    print(f"Total QSO entries: {len(all_redshifts_qso)}")

    # Initialize a labels array with zeros
    num_qsos = len(all_qso_fluxes)
    labels = np.zeros(num_qsos, dtype=int)
    LAE_redshifts_for_qsos = np.zeros(num_qsos)
    LAE_intflux_for_qsos = np.zeros(num_qsos)
    LAE_names_for_qsos = np.array([""] * num_qsos, dtype=object)

    # Handling 100% lensing when lens_percentage is 1.0
    if lens_percentage == 1.0:
        valid_qso_indices = np.where(all_redshifts_qso <= 3.0)[0]
        print(f"Number of valid QSOs for lensing: {len(valid_qso_indices)}")
    else:
        # Randomly sample a percentage of the QSOs, regardless of redshift
        num_modify = int(lens_percentage * num_qsos)
        valid_qso_indices = random.sample(range(num_qsos), num_modify)

    # List to keep track of successfully modified indices
    modified_indices = []

    print(np.max(valid_qso_indices))
    print(len(valid_qso_indices))

    # Superimpose LAE fluxes onto QSO fluxes based on the redshift condition
    for i in valid_qso_indices:
        #if i >= len(all_redshiftsLAE):
        #    print(f"QSO index {i} exceeds LAE redshift data size, skipping.")
        #    continue

        elg_candidates = np.where(all_redshiftsLAE > all_redshifts_qso[i])[0]
        if elg_candidates.size > 0:
            elg_idx = random.choice(elg_candidates)
            elg_flux = all_spectraLAE[elg_idx]

            # Generate adjusted_factor from a normal distribution
            while True:
                adjusted_factor = np.random.normal(4, 2)
                if adjusted_factor >= 2 and adjusted_factor <= 4:
                    break

            all_qso_fluxes[i] += elg_flux * adjusted_factor
            labels[i] = 1  # Set the label to 1 to indicate modification
            LAE_redshifts_for_qsos[i] = all_redshiftsLAE[elg_idx]
            LAE_intflux_for_qsos[i] = all_intfluxLAE[elg_idx]
            LAE_names_for_qsos[i] = all_namesLAE[elg_idx]
            modified_indices.append(i)

    print(f"Number of modified QSOs: {len(modified_indices)}")

    # Output the modified data into separate files based on the original distribution
    start_idx = 0
    for idx, fits_file in enumerate(fits_files):
        end_idx = file_qso_indices[idx]

        # Create a new HDUList object for the modified data
        hdu_list = fits.HDUList()

        # Create a Primary HDU
        primary_hdu = fits.PrimaryHDU()
        hdu_list.append(primary_hdu)

        # Create a BinaryTableHDU for the modified fluxes
        col1 = fits.Column(name='TARGETID', format='K', array=all_targetids_qso[start_idx:end_idx])
        col2 = fits.Column(name='Z', format='D', array=all_redshifts_qso[start_idx:end_idx])
        col3 = fits.Column(name='FLUX', format='QD()', array=all_qso_fluxes[start_idx:end_idx])
        col4 = fits.Column(name='ivar', format='QD()', array=all_ivar[start_idx:end_idx])
        col5 = fits.Column(name='LABEL', format='I', array=labels[start_idx:end_idx])  # Add labels column
        col6 = fits.Column(name='ELG_Z', format='D', array=LAE_redshifts_for_qsos[start_idx:end_idx])  # Add ELG redshift column
        col7 = fits.Column(name='ELG_NAME', format='A40', array=LAE_names_for_qsos[start_idx:end_idx])  # Add ELG name column
        col8 = fits.Column(name='LAE_intflux', format='D', array=LAE_intflux_for_qsos[start_idx:end_idx])  # Add LAE intflux
        cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8])
        hdu = fits.BinTableHDU.from_columns(cols)

        hdu_list.append(hdu)

        # Write to a new FITS file
        output_file_path = os.path.join(output_directory, f"modified_minsignal1_{fits_file}")
        hdu_list.writeto(output_file_path, overwrite=True)

        start_idx = end_idx

    # Output the results
    print("Original QSO fluxes shape:", all_qso_fluxes.shape)
    print("Labels array shape:", labels.shape)
    print("Number of modified fluxes:", np.sum(labels))

    # Ensure the changes are reflected in a new array for further analysis
    resulting_fluxes = all_qso_fluxes


# Example usage
fits_directoryA = os.path.expandvars('$SCRATCH/MainQSO/A')
hdf5_directoryA = os.path.expandvars('$SCRATCH/augLAE_minsignal1_hdf5')
output_directoryA = os.path.expandvars('$SCRATCH/modifiedMainQSO_minsignal1/A')
# I directly indicate what percentage of the files I want to be lensed and select the number of files
# i want to use from the MainQSO directory (whereever the QSO files are located
process_qso_fluxes(fits_directoryA, hdf5_directoryA, output_directoryA, lens_percentage=.1, num_files=None)

# Example usage
fits_directoryB = os.path.expandvars('$SCRATCH/MainQSO/B')
hdf5_directoryB = os.path.expandvars('$SCRATCH/augLAE_minsignal1_hdf5')
output_directoryB = os.path.expandvars('$SCRATCH/modifiedMainQSO_minsignal1/B')
# I directly indicate what percentage of the files I want to be lensed and select the number of files
# i want to use from the MainQSO directory (whereever the QSO files are located
process_qso_fluxes(fits_directoryB, hdf5_directoryB, output_directoryB, lens_percentage=.1, num_files=None)
