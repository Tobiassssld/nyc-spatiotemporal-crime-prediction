# Data

This directory contains the processed data required to run the crime-prediction experiments.

The modelling pipeline can be executed directly with the files included here. The original raw datasets are not redistributed in this repository because of their size and differing source licenses.

## Processed Data

The repository includes two processed data components:

```text
data/
├── FEATURE_MATRIX.csv
├── nyc_grid_2km_active.shp
├── nyc_grid_2km_active.shx
├── nyc_grid_2km_active.dbf
├── nyc_grid_2km_active.prj
└── nyc_grid_2km_active.cpg
```

### `FEATURE_MATRIX.csv`

This is the model-ready daily feature matrix used by `run_experiments.py`.

The dataset covers:

- **365 days**, from 2012-07-01 to 2013-06-30;
- **250 active spatial regions** in New York City;
- **91,250 region-day observations**;
- one target variable and eight predictor variables.

Each row represents one spatial region on one day.

The main columns are:

| Column | Description |
| --- | --- |
| `region_id` | Identifier of the active spatial grid cell |
| `date` | Calendar date |
| `complaint_count` | Daily NYPD complaint count; prediction target |
| `sas_count` | Daily NYPD stop, question, and frisk event count |
| `311_count` | Daily NYC 311 service-request count |
| `checkin_count` | Daily Foursquare check-in count |
| `taxi_count` | Daily taxi pickup count |
| `PRCP` | Daily precipitation |
| `SNOW` | Daily snowfall |
| `TMIN` | Daily minimum temperature |
| `TMAX` | Daily maximum temperature |

Count-based features are spatially aggregated to the active grid cells. Weather variables are city-level daily measurements and are therefore shared across regions for a given date.

The modelling code reshapes this table into:

```text
X: (365, 250, 8)
Y: (365, 250)
```

where `X` contains the eight predictor variables and `Y` contains daily complaint counts.

### `nyc_grid_2km_active.*`

These files together form the processed shapefile representing the active NYC spatial grid.

The grid was constructed using approximately **2 km × 2 km cells** covering New York City. Cells without crime complaints during the study period were removed, leaving 250 active regions.

The geometry is used to construct the spatial-neighbor graph for the TCP spatial regularization terms. Two regions are treated as neighbors when their polygons touch.

A shapefile consists of multiple associated files. Keep the complete set together:

```text
nyc_grid_2km_active.shp
nyc_grid_2km_active.shx
nyc_grid_2km_active.dbf
nyc_grid_2km_active.prj
nyc_grid_2km_active.cpg
```

At minimum, the `.shp`, `.shx`, and `.dbf` files are required for the shapefile to load correctly. The projection and encoding files should also be retained.

## Raw Data Sources

The processed feature matrix was constructed from several public urban datasets.

The raw datasets themselves are not included in this repository.

### NYPD Complaint Data

Daily crime complaints are used as the prediction target.

Source:

[NYPD Complaint Data Historic](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i/about_data)

### NYPD Stop, Question and Frisk Data

Stop, question, and frisk records are aggregated by region and date to create `sas_count`.

Source:

[NYPD Stop, Question and Frisk Data](https://www.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page)

### NYC 311 Service Requests

311 records are aggregated by region and date to create `311_count`.

Source:

[NYC 311 Service Requests](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/about_data)

### Foursquare Check-ins

Foursquare venue check-ins are spatially assigned to grid cells and aggregated by date to create `checkin_count`.

Source:

[Foursquare Check-in Dataset](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)

### NYC Taxi Trips

Taxi pickup locations are spatially aggregated by grid cell and date to create `taxi_count`.

Source:

[NYC Taxi Trip Data](https://databank.illinois.edu/datasets/IDB-9610843)

### Weather

Daily weather observations from Central Park provide:

- precipitation (`PRCP`);
- snowfall (`SNOW`);
- minimum temperature (`TMIN`);
- maximum temperature (`TMAX`).

Source:

[NYC Central Park Weather](https://www.kaggle.com/datasets/danbraswell/new-york-city-weather-18692022)

### NYC Borough Boundaries

NYC administrative boundaries are used when constructing and clipping the spatial grid.

Source:

[NYC Borough Boundaries](https://www.nyc.gov/content/planning/pages/resources/datasets/borough-boundaries)

## Rebuilding the Processed Data

The processed files included in this directory are sufficient to run the modelling experiments:

```bash
python run_experiments.py --mode smoke
```

or:

```bash
python run_experiments.py --mode full
```

Rebuilding the feature matrix from the original source datasets is optional.

The preprocessing workflow is provided in:

```text
notebooks/build_feature_matrix.ipynb
```

Raw source files should be placed under:

```text
raw_data/
```

following the paths referenced in the notebook.

The notebook performs the main preprocessing steps:

1. load and clean the source datasets;
2. convert point-based records to spatial geometries;
3. construct and clip the NYC grid;
4. spatially join events to grid cells;
5. aggregate observations by region and date;
6. restrict the dataset to active regions;
7. align all features to the same daily region-date index;
8. merge daily weather variables;
9. fill missing count-based observations where appropriate;
10. export the final `FEATURE_MATRIX.csv` and active-grid shapefile.

The `raw_data/` directory is intentionally excluded from version control.

## Reproducibility

For a quick end-to-end validation using the processed data:

```bash
python run_experiments.py --mode smoke --device cpu --seed 42
```

Smoke mode uses a reduced search space and fewer training epochs. It verifies the complete data-loading, training, forecasting, evaluation, and visualization pipeline, but its prediction scores should not be compared with the reported full-experiment results.

The complete experimental configuration can be run with:

```bash
python run_experiments.py --mode full
```

See the repository-level [`README.md`](../README.md) for details on the model, evaluation protocol, and reported results.