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
random.seed(123)     # Set seed for Python's random module

# Function to extract data from HDF5 files
def extract_data_from_hdf5(file_path):
    classifier, name, spectra, redshift = [], [], [], []
    with h5py.File(file_path, "r") as hdf5_file:
        classifier = hdf5_file["classifier"][:]
        name = hdf5_file["name"][:]
        spectra = hdf5_file["spectra"][:]
        redshift = hdf5_file["redshift"][:]
    return classifier, [n.decode() for n in name], spectra, redshift

# Function to process and modify QSO fluxes
def process_qso_fluxes(fits_directory, pkl_directory, pkl_names, output_directory, lens_percentage=1.0, num_files=None):
    # List all pickle files in the directory
    pkl_files = [os.path.join(pkl_directory, f) for f in pkl_names]

    # Store LAE data
    all_LAE = pd.concat([pd.read_pickle(f) for f in pkl_files], ignore_index=True)
    print(len(all_LAE))


    # List all QSO FITS files in the directory
    fits_files = sorted([f for f in os.listdir(fits_directory) if f.endswith(".fits")])

    if num_files is not None:
        fits_files = fits_files[:num_files]

    # Initialize arrays to store aggregated QSO data
    all_qso_fluxes = []
    all_targetids_qso = []
    all_redshifts_qso = []

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
        for i in range(len(qsos)):
            coadd = coadd_cameras(qsos[i])
            qso_fluxes.append(coadd.flux['brz'][0])

        # Append data to the aggregated arrays
        all_qso_fluxes.extend(qso_fluxes)
        all_targetids_qso.extend(targetids_qso)
        all_redshifts_qso.extend(redshifts_qso)
        file_qso_indices.append(len(all_qso_fluxes))

    all_qso_fluxes = np.array(all_qso_fluxes)
    all_targetids_qso = np.array(all_targetids_qso)
    all_redshifts_qso = np.array(all_redshifts_qso)
    
    # Initialize a labels array with zeros
    num_qsos = len(all_qso_fluxes)
    labels = np.zeros(num_qsos, dtype=int)
    LAE_redshifts_for_qsos = np.zeros(num_qsos)
    LAE_names_for_qsos = np.array([""] * num_qsos, dtype=object)

    # Handling 100% lensing when lens_percentage is 1.0
    if lens_percentage == 1.0:
        valid_qso_indices = np.where(all_redshifts_qso <= 3.0)[0]
    else:
        # Randomly sample a percentage of the QSOs, regardless of redshift
        num_modify = int(lens_percentage * num_qsos)
        valid_qso_indices = random.sample(range(num_qsos), num_modify)

    # List to keep track of successfully modified indices
    modified_indices = []

    # Superimpose LAE fluxes onto QSO fluxes based on the redshift condition
    for i in valid_qso_indices:
        LAE_candidates = np.where(all_LAE["redshift"] > all_redshifts_qso[i])[0]
        if LAE_candidates.size > 0:
            LAE_idx = random.choice(LAE_candidates)
            LAE_flux = all_LAE["spectrum"][LAE_idx]

            # Generate adjusted_factor from a normal distribution
            while True:
                adjusted_factor = np.random.normal(4, 2)
                if adjusted_factor >= 2 and adjusted_factor <= 4:
                    break

            all_qso_fluxes[i] += LAE_flux * adjusted_factor
            labels[i] = 1  # Set the label to 1 to indicate modification
            LAE_redshifts_for_qsos[i] = all_LAE["redshift"][LAE_idx]
            LAE_names_for_qsos[i] = all_LAE["name"][LAE_idx]
            modified_indices.append(i)

    print(f"Number of modified QSOs: {len(modified_indices)}")  # Output the modified data into separate files based on original distribution
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
        col3 = fits.Column(name='FLUX', format='PD()', array=all_qso_fluxes[start_idx:end_idx])
        col4 = fits.Column(name='LABEL', format='I', array=labels[start_idx:end_idx])  # Add labels column
        col5 = fits.Column(name='ELG_Z', format='D', array=LAE_redshifts_for_qsos[start_idx:end_idx])  # Add LAE redshift column
        col6 = fits.Column(name='ELG_NAME', format='A20', array=LAE_names_for_qsos[start_idx:end_idx])  # Add LAE name column
        cols = fits.ColDefs([col1, col2, col3, col4, col5, col6])
        hdu = fits.BinTableHDU.from_columns(cols)

        hdu_list.append(hdu)

        # Write to a new FITS file
        output_file_path = os.path.join(output_directory, f"modified_{fits_file}")
        hdu_list.writeto(output_file_path, overwrite=True)

        start_idx = end_idx

    # Output the results
    print("Original QSO fluxes shape:", all_qso_fluxes.shape)
    print("Labels array shape:", labels.shape)
    print("Number of modified fluxes:", np.sum(labels))

    # Ensure the changes are reflected in a new array for further analysis
    resulting_fluxes = all_qso_fluxes



# Example usage
fits_directory = os.path.expandvars('$SCRATCH/MainQSO/')
pkl_directory = 'fastspec_LAE1'
output_directory = 'MainQSO/Redshift/'

pkl_names = ["augLAE_SPHINX.pkl", "augLAE_ZoomSim.pkl"]

process_qso_fluxes(fits_directory, pkl_directory, pkl_names, output_directory, lens_percentage=1.0, num_files=None)
