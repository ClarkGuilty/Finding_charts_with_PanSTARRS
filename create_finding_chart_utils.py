#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 01:02:57 2026

@author: Javier Acevedo Barroso and Daniel Stern
"""

import argparse
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from astropy.utils.exceptions import AstropyWarning
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from datetime import datetime
from matplotlib import gridspec
from os.path import join
import os
import warnings

warnings.simplefilter("ignore", AstropyWarning)


# magnitude_column = 'rMeanKronMag'
magnitude_column = 'rPSFMag'
orig_figures_path = "Figures"
orig_outputs_path = 'Outputs'


def _normalize_column(col):
    col = str(col).strip().lower()
    if col == "ra":
        return "RA"
    if col == "dec":
        return "Dec"
    if col == "name":
        return "name"
    return col

def read_catalog(path):
    # Peek at the first row without assuming a header.
    first_row = pd.read_csv(path, header=None, nrows=1)
    first_row = [str(x).strip().lower() for x in first_row.iloc[0]]

    has_header = "ra" in first_row and "dec" in first_row

    if has_header:
        df = pd.read_csv(path)
        df.columns = [_normalize_column(col) for col in df.columns]
    else:
        df = pd.read_csv(path, header=None)
        if df.shape[1] == 3:
            df.columns = ["name", "RA", "Dec"]
        elif df.shape[1] == 2:
            df.columns = ["RA", "Dec"]
        else:
            raise ValueError(f"Expected 2 or 3 columns, got {df.shape[1]} columns")
    return df


def _guess_unit(value, coord_type):
    """
    Guess the angular unit for `value` based on its type/content.
    """
    if isinstance(value, str):
        lower = value.lower()
        has_sexagesimal = ':' in value or any(c in lower for c in 'hms°d')
        if coord_type == 'ra':
            return u.hourangle if has_sexagesimal else u.deg
        else:
            # declination is almost always in degrees
            return u.deg
    elif isinstance(value, (int, float)):
        return u.deg
    elif isinstance(value, u.Quantity):
        return value.unit
    else:
        return u.deg  # fallback

def parse_any_coord(ra, dec, frame='icrs', **kwargs):
    """
    Parse RA/Dec in any of the commonly used formats and return a SkyCoord.
    """
    if isinstance(ra, SkyCoord):
        if dec is not None:
            raise ValueError("If `ra` is already a SkyCoord, `dec` must be None.")
        return ra

    ra_unit = _guess_unit(ra, 'ra')
    dec_unit = _guess_unit(dec, 'dec')

    ra_angle = Angle(ra, unit=ra_unit)
    dec_angle = Angle(dec, unit=dec_unit)

    return SkyCoord(ra=ra_angle, dec=dec_angle, frame=frame, **kwargs)

def decompose_flags(flags: int):
    """
    Decompose an integer into its individual binary flags.
    Returns a list of the powers of two that make up the flag.
    """
    if flags < 0:
        raise ValueError("Flags must be a non-negative integer")

    result = []
    bit_position = 0
    while flags:
        if flags & 1:
            result.append(1 << bit_position)
        flags >>= 1
        bit_position += 1
    return result


def to_sexagesimal(ra_deg, dec_deg):
    c = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree)
    return c.to_string('hmsdms', sep=':', precision=2)

def get_brightest_sources(coords, radius_arcmin=1, count=3):
    """Queries the Pan-STARRS catalog for the brightest sources."""
    # service_url = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.csv"
    service_url = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/stack.csv"
    params = {
        "ra": coords.ra.deg,
        "dec": coords.dec.deg,
        "radius": radius_arcmin/60,
        "columns": ['objID','raMean','decMean',magnitude_column, 'qualityFlag'],
        "nObserve": ">5" # filter for quality (detected more than 5 times)
    }
    response = requests.get(service_url, params=params)
    # if response.status_code != 200 or "ra" not in response.text:
    #     return None
    df = pd.read_csv(StringIO(response.text))
    df = df[df[magnitude_column] > 0]
    brightest = df.sort_values(by=magnitude_column).head(count).rename(
        columns={'raMean':'ra','decMean':'dec'})
    return brightest[['ra', 'dec', magnitude_column]]

def download_panstarrs_r_band(coords, size_arcmin=2.0, output_file="ps1_image.fits"):
    """Downloads a Pan-STARRS r-band FITS image."""
    ra = coords.ra.deg
    dec = coords.dec.deg
    size_pixels = int((size_arcmin * 60) / 0.25)
    service_url = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    params = {"ra": ra, "dec": dec, "filters": "r"}
    res = requests.get(service_url, params=params)

    lines = res.text.split('\n')
    if len(lines) < 2: return False
    
    header = lines[0].split()
    data = lines[1].split()
    remote_filename = data[header.index("filename")]
    
    cut_url = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
    cut_params = {"ra": ra, "dec": dec, "size": size_pixels, "format": "fits", "red": remote_filename}
    img_res = requests.get(cut_url, params=cut_params)
    
    if img_res.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(img_res.content)
        return True
    return False



def create_finding_chart(fits_file, target_coords, catalog_df,
                         current_figures_path,
                         target_name=None,
                         output_format="png",
                         dpi=150,
                         appendix="finding_chart"):
    """Creates a PNG finding chart with a side panel for coordinate offsets."""
    with fits.open(fits_file) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)

    output_filename = f"{appendix}.{output_format}"
    if target_name is not None:
        output_filename = target_name+"_"+output_filename
    
    target_ra = target_coords.ra.deg
    target_dec = target_coords.dec.deg
    norm = ImageNormalize(data, interval=ZScaleInterval())

    # Create figure with extra width for the side panel
    fw, fh = plt.rcParams['figure.figsize']
    fig = plt.figure(figsize=(1.5 * fw, 1*fw)) 
    
    # Create a grid: 1 row, 2 columns. 
    gs = gridspec.GridSpec(1, 2, width_ratios=[2,1], wspace=0.1)

    # Left subplot: The Image
    ax = fig.add_subplot(gs[0], projection=wcs)
    ax.imshow(data, cmap='gist_yarg', norm=norm, origin='lower')

    # Right subplot: The Text Panel
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off') # Hide axes for the text panel
    
    # Initialize text string for the side panel
    offset_list_text = "  Offsets (arcsec)\nfrom star to target\n" + "-"*20 + "\n"

    # 1. Mark the Central Target
    tx, ty = wcs.all_world2pix(target_ra, target_dec, 1)
    ax.scatter(tx, ty, s=200, facecolor='red', marker='+',
               label='Target', lw=2)
    
    # 2. Mark sources and calculate offsets
    for i, (idx, row) in enumerate(catalog_df.iterrows()):
        sx, sy = wcs.all_world2pix(row.ra, row.dec, 1)
        
        # Calculate offsets
        # Note: dra = -(RA_star - RA_target) * cos(Dec)
        dra_as = -(row.ra - target_ra) * np.cos(np.radians(target_dec)) * 3600
        ddec_as = -(row.dec - target_dec) * 3600
        
                
        # Plot on image
        label_text = f"S{i+1}: r={row[magnitude_column]:.1f}"

        ax.scatter(sx, sy, s=150, edgecolor='cyan', facecolor='none', lw=1.5)
        ax.text(sx-35, sy + 20, label_text, color='yellow', fontsize='medium', 
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

        # Add to side panel text
        # Using :+ .1f to force showing the plus/minus sign
        offset_list_text += f"S{i+1}: dRA {dra_as:+.1f} dDec {ddec_as:+.1f}\n"

    # Place the compiled text in the right panel
    ax_text.text(0.0, 0.8, offset_list_text, transform=ax_text.transAxes, 
             verticalalignment='top', horizontalalignment='left',
             family='monospace', fontsize=10)

    radec_hex = to_sexagesimal(target_ra, target_dec)
    ax.set_title(f"{target_name}\nPS1 r-band Finding Chart\n {target_ra:.5f}, {target_dec:.5f}\n{radec_hex}\n")
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    
    plt.savefig(join(current_figures_path, output_filename), 
                bbox_inches='tight', dpi=dpi)
    print(f"Finding chart saved to: {output_filename}")
    # plt.show()
    

def make_finding_chart_from_coordinates(coords,
                                        target_name,
                                        search_radius, #in arcmins
                                        current_figures_path,
                                        current_outputs_path,
                                        fits_name=None,
                                        n_objects=5, #number of objects
                                        min_distance_to_target=0.5, #arcsec
                                        min_distance_between_offsets=1, #arcsec
                                        output_format="png",
                                        dpi=150,
                                        output_individual_catalogues=False,
                                        ):
    
    results_list = []
    
    if fits_name is None:
        fits_name = f"{target_name}.fits"

    if download_panstarrs_r_band(coords,
                                 size_arcmin=search_radius,
                                 output_file=fits_name):
        
        # Query more stars than needed to account for filtering
        brightest_df = get_brightest_sources(coords, 
                                             radius_arcmin=search_radius/2,
                                             count=n_objects + 15)
        
        if brightest_df is None or brightest_df.empty:
            return []

        # 1. Convert found stars to SkyCoord
        candidate_coords = SkyCoord(ra=brightest_df['ra'].values,
                                    dec=brightest_df['dec'].values, 
                                    unit='deg')
        
        # 2. Filter out objects too close to the TARGET
        dists_to_target = coords.separation(candidate_coords).arcsec
        mask = dists_to_target > min_distance_to_target
        brightest_df = brightest_df[mask].reset_index(drop=True)
        candidate_coords = candidate_coords[mask]

        # 3. Filter out objects too close to EACH OTHER (Internal duplicates)
        # We keep only the first occurrence within a 0.5" radius
        if len(brightest_df) > 0:
            final_indices = []
            skipped_indices = set()
            
            for i in range(len(candidate_coords)):
                if i in skipped_indices:
                    continue
                
                # Compare this star to all subsequent stars
                sep = candidate_coords[i].separation(candidate_coords[i+1:]).arcsec
                # Find indices of neighbors within 0.5 arcsec
                duplicates = np.where(sep < min_distance_between_offsets)[0] + (i + 1)
                skipped_indices.update(duplicates)
                final_indices.append(i)
            
            brightest_df = brightest_df.iloc[final_indices].head(n_objects)
        
        # 4. Process the remaining unique stars
        target_ra = coords.ra.deg
        target_dec = coords.dec.deg

        # Helper to format angles
        def format_coords(ra_deg, dec_deg):
            # ra_str as HH:MM:SS.SS, dec_str as +DD:MM:SS.SS
            ra_str = Angle(ra_deg, unit='deg').to_string(unit='hour', sep=':', precision=2, pad=True)
            dec_str = Angle(dec_deg, unit='deg').to_string(unit='deg', sep=':', precision=1, pad=True, alwayssign=True)
            return ra_str, dec_str

        t_ra_fmt, t_dec_fmt = format_coords(target_ra, target_dec)
        
        row_data = {
            "NAME": target_name,
            "RA": t_ra_fmt,
            "DECL": t_dec_fmt,
            "OFFSET_RA": 0,
            "OFFSET_DEC": 0,
            "COMMENT": "",
            "PRIORITY": "", "BINSPAT": "2", "BINSPECT": "2",
            "SLITANGLE": "PA", "SLITWIDTH": "1.5", "AIRMASS_MAX": "2.5",
            "WRANGE_LOW": "650", "WRANGE_HIGH": "680", "CHANNEL": "R",
            "MAGNITUDE": "21.0",
            "MAGFILTER": "z", "EXPTIME": "SNR"
        }
        results_list.append(row_data)

        for i, (idx, row) in enumerate(brightest_df.iterrows()):
            dra_as = -(row['ra'] - target_ra) * np.cos(np.radians(target_dec)) * 3600
            ddec_as = -(row['dec'] - target_dec) * 3600
            
            t_ra_fmt_star, t_dec_fmt_star = format_coords(row['ra'], row['dec'])
            
            row_data = {
                "NAME": f"{target_name}_os_{i+1}",
                "RA": t_ra_fmt_star,
                "DECL": t_dec_fmt_star,
                "OFFSET_RA": round(dra_as, 2),
                "OFFSET_DEC": round(ddec_as, 2),
                "COMMENT": f"Ref Star {i+1} mag {row[magnitude_column]:.1f}",
                "PRIORITY": "", "BINSPAT": "2", "BINSPECT": "2",
                "SLITANGLE": "PA", "SLITWIDTH": "1.5", "AIRMASS_MAX": "2.5",
                "WRANGE_LOW": "650", "WRANGE_HIGH": "680", "CHANNEL": "R",
                "MAGNITUDE": 21.0,
                "MAGFILTER": "z", "EXPTIME": "SNR"
            }
            results_list.append(row_data)

        if output_individual_catalogues:
            with open(join(current_outputs_path, target_name+'.txt'), 'w') as f:
                print("RA,Dec,rPSFMag,dRa,dDec", file=f)
                for item in results_list:
                    print(f"{item['RA']},{item['DECL']},{item['MAGNITUDE']},{item['OFFSET_RA']},{item['OFFSET_DEC']}", file=f)
        
        
        if not brightest_df.empty:
            create_finding_chart(fits_name,
                                 coords,
                                 brightest_df,
                                 current_figures_path,
                                 target_name=target_name,
                                 output_format=output_format,
                                 dpi=dpi
                                 )
                
        if os.path.exists(fits_name):
            os.remove(fits_name)   
            
        return results_list
    else:
        print(f"Download failed for {target_name} (PanSTARRS footprint check).")
        return []
        
def name_from_radec(coords, survey=""):
    ra, dec = coords.to_string('hmsdms').split()
    return f'{survey}J'+ra[:2]+ra[3:5]+ra[6:11]+dec[:3].replace('-','-')+dec[4:6]+dec[7:11] # J031718.38-413525.9 No Latex

#%%

def existing_file(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"{path} does not exist or is not a file.")
    return path

def positive_float(value):
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid float.")
    if f <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be greater than zero.")
    return f

def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate finding charts for a list of targets or a single object."
    )

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "-f", "--file",
        type=existing_file,
        help="Path to a text file containing names and RA/Dec coordinates."
    )
    target_group.add_argument(
        "--radec",
        nargs=2,
        metavar=("RA", "DEC"),
        help="RA and Dec for a single target (provide both values).\n"
        "Use angles in decimal deg, hexagesimal, or hmsdms."
    )

    parser.add_argument(
        "--name",
        help="Optional name for the single target (used only with --radec)."
    )
    parser.add_argument(
        "--nstars",
        type=int,
        default=5,
        help="Number of stars to include in the finding chart (default: 5)."
    )
    parser.add_argument(
        "--chart-size",
        type=positive_float,
        default=2.9,
        # metavar="ARCMIN",
        help=(
            "Size of the finding chart in arcminutes. "
            "This also serves as the search DIAMETER used to locate bright "
            "reference stars for offset calculations (default: 2.9')."
        )
    )
    parser.add_argument(
        "--min-offset-radius",
        type=positive_float,
        default=2,
        # metavar="ARCSEC",
        help=(
            "Minimum distance from the target coordinate (in arcminutes) below which "
            "bright-object matches are ignored. This prevents matching the object itself "
            "(default: 0.5'')."
        )
    )
    parser.add_argument(
    "--format",
    choices=("pdf", "png"),
    default="pdf",
    help="Output format for the finding chart (default: pdf)."
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for PNG output (ignored for PDF, default: 150)."
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this execution run (used as the folder name)."
    )
    
    parser.add_argument(
        "--output-individual-catalogues", # Renamed from --output-catalog
        action="store_true",
        help=(
            "Output a separate .txt catalogue for each target containing "
            "reference star offsets."
        )
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Determine sub-directory name
    if args.run_name:
        run_folder = args.run_name
    else:
        run_folder = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        
    # Construct and create the specific paths for this run
    current_figures_path = join(orig_figures_path, run_folder)
    current_outputs_path = join(orig_outputs_path, run_folder)
    
    os.makedirs(current_figures_path, exist_ok=True)
    os.makedirs(current_outputs_path, exist_ok=True)

    print(f"Number of stars: {args.nstars}")
    print(f"Chart size / search diameter: {args.chart_size} arcmin")
    print(f"Minimum offset radius: {args.min_offset_radius} arcsec\n")

    if args.file is None:
        row_dict = {'RA' : args.radec[0], 'Dec' : args.radec[1]}
        if args.name: row_dict['name'] = args.name
        df = pd.DataFrame([row_dict])
    else:
        print(f"Processing targets from file: {args.file}")
        df = read_catalog(args.file)
    
    all_summary_data = [] # Global collection

    for index, row in df.iterrows():
        coords = parse_any_coord(row['RA'], row['Dec'])
        target_name = row['name'] if 'name' in row.keys() else name_from_radec(coords)
        fits_name = f"{target_name}.fits"

        try:
            target_data_list = make_finding_chart_from_coordinates(
                                            coords,
                                            target_name,
                                            args.chart_size,
                                            current_figures_path,
                                            current_outputs_path,
                                            fits_name=fits_name,
                                            n_objects=args.nstars,
                                            min_distance_to_target=args.min_offset_radius,
                                            output_format=args.format,
                                            dpi=args.dpi,
                                            output_individual_catalogues=args.output_individual_catalogues
                                            )
            if target_data_list:
                all_summary_data.extend(target_data_list)
                
        except pd.errors.EmptyDataError as E:
            print(f"Failed to find a match for: {target_name}")

    # Create the final dataframe
    if all_summary_data:
        final_df = pd.DataFrame(all_summary_data)
        output_csv = join(current_outputs_path, "summary.csv")
        final_df.to_csv(output_csv, index=False)
        print(f"Summary catalogue saved to: {output_csv}")

if __name__ == "__main__":
    main()

