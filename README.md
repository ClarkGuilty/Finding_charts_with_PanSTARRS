# Finding_charts_with_PanSTARRS

Simple scripts to make finding charts with offsets to nearby bright sources using PanSTARRS imaging and catalogs.

It also includes utilities to prepare observations with NGPS at Palomar Observatory.


## Installation

Having a dedicated environment is the most reliable way to use the program. Some options for this are: `Conda`, `Micromamba`, `venv`, or `uv`. Include the following packages in your environment:

```
requests
pandas
numpy
matplotlib
astropy
```

### Installation using `conda`/`micromamba`
```
conda create -n finding_charts python requests pandas numpy matplotlib astropy
```
Then, make sure you have activated your environment bedfore running the python script.
```
conda activate finding_charts
```

### Installation using pip
It is recommended that always use pip inside an enviroment. Also, some systems use `pip3` and `python3`, so adapt accordingly.
```
pip install requests pandas numpy matplotlib astropy
```

## Usage

```
python create_finding_chart_utils.py [-h] (-f FILE | --radec RA DEC)
                                     [--name NAME] [--nstars NSTARS]
                                     [--chart-size CHART_SIZE]
                                     [--min-offset-radius MIN_OFFSET_RADIUS]
                                     [--format {pdf,png}] [--dpi DPI]

Generate finding charts for a list of targets or a single object.

options:
  -h, --help            show this help message and exit
  -f, --file FILE       Path to a text file containing names and RA/Dec
                        coordinates.
  --radec RA DEC        RA and Dec for a single target (provide both values).
                        Use angles in decimal deg, hexagesimal, or hmsdms.
  --name NAME           Optional name for the single target (used only with
                        --radec).
  --nstars NSTARS       Number of stars to include in the finding chart
                        (default: 5).
  --chart-size CHART_SIZE
                        Size of the finding chart in arcminutes. This also
                        serves as the search DIAMETER used to locate bright
                        reference stars for offset calculations (default:
                        2.9').
  --min-offset-radius MIN_OFFSET_RADIUS
                        Minimum distance from the target coordinate (in
                        arcminutes) below which bright-object matches are
                        ignored. This prevents matching the object itself
                        (default: 0.5'').
  --format {pdf,png}    Output format for the finding chart (default: pdf).
  --dpi DPI             DPI for PNG output (ignored for PDF, default: 150).
```



## Examples



### Making a finding chart for a concrete RA-Dec:
```bash
python create_finding_chart_utils.py --radec 15.0311 -5.2373 --name example_1
```

This will query PanSTARRS and download a fits image centered around the target coordinates. If the area is covered in PanSTARRS footprint, then it will also download the catalogue of sources in a cone of radius `CHART_SIZE`/2 (1.45' by default) and mark the five brightest sources. Finally, it will save the finding chart in Figures/example_1.pdf.

To increase the number of sources highlighted, use the `--nstars` argument:
```bash
python create_finding_chart_utils.py --radec 15.0311 -5.2373 --name example_2 --nstars 15
```

If there are no bright enough sources in your FoV, you can increase the search radius (and with it, the size of your finding chart) with `CHART_SIZE`:
```sh
python create_finding_chart_utils.py --radec 15.0311 -5.2373 --name example_3 --chart-size 5
```

### Making finding charts for a list of coordinates:

Your input catalog must have 2 or 3 columns: `Name`, `RA`, and `Dec`. The `Name` column is optional. Examples of complying catalogs include:

#### Two columns, no header.
```
15.0311,-5.2373
12.8435,-1.9270
187.2334,73.9249
```

#### Two columns, with header.
```
RA,Dec
15.0311,-5.2373
12.8435,-1.9270
187.2334,73.9249
```

#### Three columns, no header.
```
Target_1,15.0311,-5.2373
Target_2,12.8435,-1.9270
Target_3,187.2334,73.9249
```

#### Three columns, with header.
```
Name,RA,Dec
Target_1,15.0311,-5.2373
Target_2,12.8435,-1.9270
Target_3,187.2334,73.9249
```

#### Finding chart from catalog -- Default parameters
Assuming your catalog file is called `catalog.txt`, then:
```sh
python create_finding_chart_utils.py -f catalog.txt
```

#### Finding chart from catalog -- Custom parameters
You can still use all parameters except for `--radec` and `name`:
```sh
python create_finding_chart_utils.py -f catalog.txt --nstars 15 --chart-size 5 --format png --dpi 200
```

