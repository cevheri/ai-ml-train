# Astronomical Data Analysis with Python Using Astropy and Astroquery

## Introduction

In this article, we will introduce Astropy and Astroquery, two powerful Python libraries for astronomical data analysis. We will show you how to use these libraries with step-by-step installation, usage and sample code. We will also provide some useful tips and tricks to help you get started with astronomical data analysis using Python.

## Install the required libraries

### Juypter Notebook

Install classic Jupyter Notebook pip:
```bash
pip install notebook
```

Run JupyterNotebook:
```bash
jupyter notebook
```

Jupyter Official Document: [Installing Jupiter Notebook](!https://jupyter.org/install)

Or you can use [Anaconda](!https://www.anaconda.com/products/distribution)

### Astropy

Astropy is a Python package that offers tools and functions that are useful for various tasks in astronomy and astrophysics, including and not limited to planning an observation, reducing the data from the observation, analysing it, and other numerical and modeling tasks.

To install Astropy, you can use pip:
```bash
pip install astropy
```

Astropy Official Document: [Astropy](!https://docs.astropy.org/en/stable/)

### Astroquery
Astroquery is a set of tools for querying astronomical web forms and databases.

There are two other packages with complimentary functionality as Astroquery: pyvo is an Astropy affiliated package, and Simple-Cone-Search-Creator to generate a cone search service complying with the IVOA standard. They are more oriented to general virtual observatory discovery and queries, whereas Astroquery has web service specific interfaces.

To install Astroquery, you can use pip:
```bash
pip install astroquery
```

Astroquery Official Document: [Astroquery](!https://astroquery.readthedocs.io/en/latest/)


## Usage

### Astropy
Astropy provides a wide range of tools for astronomical data analysis. Here are some of the most commonly used features:

Astropy provides a wide range of tools for astronomical data analysis. Here are some of the most commonly used features:

- Units and constants: Astropy provides a wide range of units and constants for astronomical calculations. You can use these units and constants to perform calculations with physical quantities.
- Time and dates: Astropy provides tools for working with time and dates in astronomy. You can use these tools to convert between different time scales, calculate the phase of the moon, and perform other time-related calculations.
- Coordinates: Astropy provides tools for working with astronomical coordinates. You can use these tools to convert between different coordinate systems, calculate the separation between two objects, and perform other coordinate-related calculations.
- Tables: Astropy provides tools for working with tabular data in astronomy. You can use these tools to read and write data tables, perform calculations on tables, and plot tables.
- FITS files: Astropy provides tools for working with FITS files, a common file format used in astronomy. You can use these tools to read and write FITS files, access the data in FITS files, and perform other operations on FITS files.
- ...and much more! :)


```python
# units and constants
from astropy import constants as const
from astropy import units as u

print(const.G)
print(const.c)
```

      Name   = Gravitational constant
      Value  = 6.6743e-11
      Uncertainty  = 1.5e-15
      Unit  = m3 / (kg s2)
      Reference = CODATA 2018
      Name   = Speed of light in vacuum
      Value  = 299792458.0
      Uncertainty  = 0.0
      Unit  = m / s
      Reference = CODATA 2018


### Astroquery
Astroquery provides a wide range of tools for querying astronomical web forms and databases. Here are some of the most commonly used features:

- Querying web forms: Astroquery provides tools for querying astronomical web forms. You can use these tools to search for astronomical objects, download data from astronomical databases, and perform other operations on the web.
- Querying databases: Astroquery provides tools for querying astronomical databases. You can use these tools to search for astronomical objects, download data from astronomical databases, and perform other operations on the data.
- Querying catalogs: Astroquery provides tools for querying astronomical catalogs. You can use these tools to search for astronomical objects in catalogs, download data from catalogs, and perform other operations on the data.
- Querying images: Astroquery provides tools for querying astronomical images. You can use these tools to search for astronomical images, download images from astronomical databases, and perform other operations on the images.
- ...and much more! :)

### Gaia data query from ESA(European Space Agency) database

* [GAIA DR3]('https://www.cosmos.esa.int/web/gaia/dr3)
 
<img src="https://www.cosmos.esa.int/documents/29201/1666086/GDR2_fluxRGB_cartesian_1000x500.png/823c19eb-2f5e-517a-86a4-c8e8f80b100f?t=1525786477803">



```python
# Import the necessary libraries

from astroquery.gaia import Gaia # Import the Gaia module from the astro
tables = Gaia.load_tables(only_names=True) # Load the available Gaia tables from the ESA database
print(tables[0:2])
```

    INFO: Retrieving tables... [astroquery.utils.tap.core]
    INFO: Parsing tables... [astroquery.utils.tap.core]
    INFO: Done. [astroquery.utils.tap.core]
    [<astroquery.utils.tap.model.taptable.TapTableMeta object at 0x7fe563a08e10>, <astroquery.utils.tap.model.taptable.TapTableMeta object at 0x7fe563a08e90>]


#### Read Gaia Data Release 3 (Gaia DR3) table



```python
# Print the tables that contain 'gaiadr3' in their names
gaiadr3 = [table.get_qualified_name() for table in tables if 'gaiadr3' in table.get_qualified_name()]
print(gaiadr3[0:2])

```

    ['gaiadr3.gaiadr3.gaia_source', 'gaiadr3.gaiadr3.gaia_source_lite']



```python
# Query the Gaia DR3 table : gaia_source 
query = """SELECT TOP 10 * FROM gaiadr3.gaia_source"""
job = Gaia.launch_job(query)
result = job.get_results()
print(result)
```

        solution_id             DESIGNATION          ... libname_gspphot
                                                     ...                
    ------------------- ---------------------------- ... ---------------
    1636148068921376768 Gaia DR3 1247114191857269120 ...         PHOENIX
    1636148068921376768 Gaia DR3 1247114264872561920 ...           MARCS
    1636148068921376768 Gaia DR3 1247114294936486272 ...                
    1636148068921376768 Gaia DR3 1247114329296223872 ...                
    1636148068921376768 Gaia DR3 1247114329298857728 ...                
    1636148068921376768 Gaia DR3 1247114329298859008 ...                
    1636148068921376768 Gaia DR3 1247114333591419008 ...           MARCS
    1636148068921376768 Gaia DR3 1247114333592038144 ...                
    1636148068921376768 Gaia DR3 1247114363655964928 ...           MARCS
    1636148068921376768 Gaia DR3 1247114363655966336 ...           MARCS



```python
# Query the Gaia DR3 table : gaiaedr3.gaia_source from parallax > 10
query = """SELECT * FROM gaiadr3.gaia_source WHERE parallax > 50"""
job = Gaia.launch_job(query)
result = job.get_results()
print(result.columns)
result[0:2]
```

    <TableColumns names=('solution_id','DESIGNATION','SOURCE_ID','random_index','ref_epoch','ra','ra_error','dec','dec_error','parallax','parallax_error','parallax_over_error','pm','pmra','pmra_error','pmdec','pmdec_error','ra_dec_corr','ra_parallax_corr','ra_pmra_corr','ra_pmdec_corr','dec_parallax_corr','dec_pmra_corr','dec_pmdec_corr','parallax_pmra_corr','parallax_pmdec_corr','pmra_pmdec_corr','astrometric_n_obs_al','astrometric_n_obs_ac','astrometric_n_good_obs_al','astrometric_n_bad_obs_al','astrometric_gof_al','astrometric_chi2_al','astrometric_excess_noise','astrometric_excess_noise_sig','astrometric_params_solved','astrometric_primary_flag','nu_eff_used_in_astrometry','pseudocolour','pseudocolour_error','ra_pseudocolour_corr','dec_pseudocolour_corr','parallax_pseudocolour_corr','pmra_pseudocolour_corr','pmdec_pseudocolour_corr','astrometric_matched_transits','visibility_periods_used','astrometric_sigma5d_max','matched_transits','new_matched_transits','matched_transits_removed','ipd_gof_harmonic_amplitude','ipd_gof_harmonic_phase','ipd_frac_multi_peak','ipd_frac_odd_win','ruwe','scan_direction_strength_k1','scan_direction_strength_k2','scan_direction_strength_k3','scan_direction_strength_k4','scan_direction_mean_k1','scan_direction_mean_k2','scan_direction_mean_k3','scan_direction_mean_k4','duplicated_source','phot_g_n_obs','phot_g_mean_flux','phot_g_mean_flux_error','phot_g_mean_flux_over_error','phot_g_mean_mag','phot_bp_n_obs','phot_bp_mean_flux','phot_bp_mean_flux_error','phot_bp_mean_flux_over_error','phot_bp_mean_mag','phot_rp_n_obs','phot_rp_mean_flux','phot_rp_mean_flux_error','phot_rp_mean_flux_over_error','phot_rp_mean_mag','phot_bp_rp_excess_factor','phot_bp_n_contaminated_transits','phot_bp_n_blended_transits','phot_rp_n_contaminated_transits','phot_rp_n_blended_transits','phot_proc_mode','bp_rp','bp_g','g_rp','radial_velocity','radial_velocity_error','rv_method_used','rv_nb_transits','rv_nb_deblended_transits','rv_visibility_periods_used','rv_expected_sig_to_noise','rv_renormalised_gof','rv_chisq_pvalue','rv_time_duration','rv_amplitude_robust','rv_template_teff','rv_template_logg','rv_template_fe_h','rv_atm_param_origin','vbroad','vbroad_error','vbroad_nb_transits','grvs_mag','grvs_mag_error','grvs_mag_nb_transits','rvs_spec_sig_to_noise','phot_variable_flag','l','b','ecl_lon','ecl_lat','in_qso_candidates','in_galaxy_candidates','non_single_star','has_xp_continuous','has_xp_sampled','has_rvs','has_epoch_photometry','has_epoch_rv','has_mcmc_gspphot','has_mcmc_msc','in_andromeda_survey','classprob_dsc_combmod_quasar','classprob_dsc_combmod_galaxy','classprob_dsc_combmod_star','teff_gspphot','teff_gspphot_lower','teff_gspphot_upper','logg_gspphot','logg_gspphot_lower','logg_gspphot_upper','mh_gspphot','mh_gspphot_lower','mh_gspphot_upper','distance_gspphot','distance_gspphot_lower','distance_gspphot_upper','azero_gspphot','azero_gspphot_lower','azero_gspphot_upper','ag_gspphot','ag_gspphot_lower','ag_gspphot_upper','ebpminrp_gspphot','ebpminrp_gspphot_lower','ebpminrp_gspphot_upper','libname_gspphot')>





<div><i>Table length=2</i>
<table id="table140623195812112" class="table-striped table-bordered table-condensed">
<thead><tr><th>solution_id</th><th>DESIGNATION</th><th>SOURCE_ID</th><th>random_index</th><th>ref_epoch</th><th>ra</th><th>ra_error</th><th>dec</th><th>dec_error</th><th>parallax</th><th>parallax_error</th><th>parallax_over_error</th><th>pm</th><th>pmra</th><th>pmra_error</th><th>pmdec</th><th>pmdec_error</th><th>ra_dec_corr</th><th>ra_parallax_corr</th><th>ra_pmra_corr</th><th>ra_pmdec_corr</th><th>dec_parallax_corr</th><th>dec_pmra_corr</th><th>dec_pmdec_corr</th><th>parallax_pmra_corr</th><th>parallax_pmdec_corr</th><th>pmra_pmdec_corr</th><th>astrometric_n_obs_al</th><th>astrometric_n_obs_ac</th><th>astrometric_n_good_obs_al</th><th>astrometric_n_bad_obs_al</th><th>astrometric_gof_al</th><th>astrometric_chi2_al</th><th>astrometric_excess_noise</th><th>astrometric_excess_noise_sig</th><th>astrometric_params_solved</th><th>astrometric_primary_flag</th><th>nu_eff_used_in_astrometry</th><th>pseudocolour</th><th>pseudocolour_error</th><th>ra_pseudocolour_corr</th><th>dec_pseudocolour_corr</th><th>parallax_pseudocolour_corr</th><th>pmra_pseudocolour_corr</th><th>pmdec_pseudocolour_corr</th><th>astrometric_matched_transits</th><th>visibility_periods_used</th><th>astrometric_sigma5d_max</th><th>matched_transits</th><th>new_matched_transits</th><th>matched_transits_removed</th><th>ipd_gof_harmonic_amplitude</th><th>ipd_gof_harmonic_phase</th><th>ipd_frac_multi_peak</th><th>ipd_frac_odd_win</th><th>ruwe</th><th>scan_direction_strength_k1</th><th>scan_direction_strength_k2</th><th>scan_direction_strength_k3</th><th>scan_direction_strength_k4</th><th>scan_direction_mean_k1</th><th>scan_direction_mean_k2</th><th>scan_direction_mean_k3</th><th>scan_direction_mean_k4</th><th>duplicated_source</th><th>phot_g_n_obs</th><th>phot_g_mean_flux</th><th>phot_g_mean_flux_error</th><th>phot_g_mean_flux_over_error</th><th>phot_g_mean_mag</th><th>phot_bp_n_obs</th><th>phot_bp_mean_flux</th><th>phot_bp_mean_flux_error</th><th>phot_bp_mean_flux_over_error</th><th>phot_bp_mean_mag</th><th>phot_rp_n_obs</th><th>phot_rp_mean_flux</th><th>phot_rp_mean_flux_error</th><th>phot_rp_mean_flux_over_error</th><th>phot_rp_mean_mag</th><th>phot_bp_rp_excess_factor</th><th>phot_bp_n_contaminated_transits</th><th>phot_bp_n_blended_transits</th><th>phot_rp_n_contaminated_transits</th><th>phot_rp_n_blended_transits</th><th>phot_proc_mode</th><th>bp_rp</th><th>bp_g</th><th>g_rp</th><th>radial_velocity</th><th>radial_velocity_error</th><th>rv_method_used</th><th>rv_nb_transits</th><th>rv_nb_deblended_transits</th><th>rv_visibility_periods_used</th><th>rv_expected_sig_to_noise</th><th>rv_renormalised_gof</th><th>rv_chisq_pvalue</th><th>rv_time_duration</th><th>rv_amplitude_robust</th><th>rv_template_teff</th><th>rv_template_logg</th><th>rv_template_fe_h</th><th>rv_atm_param_origin</th><th>vbroad</th><th>vbroad_error</th><th>vbroad_nb_transits</th><th>grvs_mag</th><th>grvs_mag_error</th><th>grvs_mag_nb_transits</th><th>rvs_spec_sig_to_noise</th><th>phot_variable_flag</th><th>l</th><th>b</th><th>ecl_lon</th><th>ecl_lat</th><th>in_qso_candidates</th><th>in_galaxy_candidates</th><th>non_single_star</th><th>has_xp_continuous</th><th>has_xp_sampled</th><th>has_rvs</th><th>has_epoch_photometry</th><th>has_epoch_rv</th><th>has_mcmc_gspphot</th><th>has_mcmc_msc</th><th>in_andromeda_survey</th><th>classprob_dsc_combmod_quasar</th><th>classprob_dsc_combmod_galaxy</th><th>classprob_dsc_combmod_star</th><th>teff_gspphot</th><th>teff_gspphot_lower</th><th>teff_gspphot_upper</th><th>logg_gspphot</th><th>logg_gspphot_lower</th><th>logg_gspphot_upper</th><th>mh_gspphot</th><th>mh_gspphot_lower</th><th>mh_gspphot_upper</th><th>distance_gspphot</th><th>distance_gspphot_lower</th><th>distance_gspphot_upper</th><th>azero_gspphot</th><th>azero_gspphot_lower</th><th>azero_gspphot_upper</th><th>ag_gspphot</th><th>ag_gspphot_lower</th><th>ag_gspphot_upper</th><th>ebpminrp_gspphot</th><th>ebpminrp_gspphot_lower</th><th>ebpminrp_gspphot_upper</th><th>libname_gspphot</th></tr></thead>
<thead><tr><th></th><th></th><th></th><th></th><th>yr</th><th>deg</th><th>mas</th><th>deg</th><th>mas</th><th>mas</th><th>mas</th><th></th><th>mas / yr</th><th>mas / yr</th><th>mas / yr</th><th>mas / yr</th><th>mas / yr</th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th>mas</th><th></th><th></th><th></th><th>1 / um</th><th>1 / um</th><th>1 / um</th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th>mas</th><th></th><th></th><th></th><th></th><th>deg</th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th>deg</th><th>deg</th><th>deg</th><th>deg</th><th></th><th></th><th>electron / s</th><th>electron / s</th><th></th><th>mag</th><th></th><th>electron / s</th><th>electron / s</th><th></th><th>mag</th><th></th><th>electron / s</th><th>electron / s</th><th></th><th>mag</th><th></th><th></th><th></th><th></th><th></th><th></th><th>mag</th><th>mag</th><th>mag</th><th>km / s</th><th>km / s</th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th>d</th><th>km / s</th><th>K</th><th>log(cm.s**-2)</th><th>dex</th><th></th><th>km / s</th><th>km / s</th><th></th><th>mag</th><th>mag</th><th></th><th></th><th></th><th>deg</th><th>deg</th><th>deg</th><th>deg</th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th>K</th><th>K</th><th>K</th><th>log(cm.s**-2)</th><th>log(cm.s**-2)</th><th>log(cm.s**-2)</th><th>dex</th><th>dex</th><th>dex</th><th>pc</th><th>pc</th><th>pc</th><th>mag</th><th>mag</th><th>mag</th><th>mag</th><th>mag</th><th>mag</th><th>mag</th><th>mag</th><th>mag</th><th></th></tr></thead>
<thead><tr><th>int64</th><th>object</th><th>int64</th><th>int64</th><th>float64</th><th>float64</th><th>float32</th><th>float64</th><th>float32</th><th>float64</th><th>float32</th><th>float32</th><th>float32</th><th>float64</th><th>float32</th><th>float64</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>int16</th><th>int16</th><th>int16</th><th>int16</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>int16</th><th>bool</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>int16</th><th>int16</th><th>float32</th><th>int16</th><th>int16</th><th>int16</th><th>float32</th><th>float32</th><th>int16</th><th>int16</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>bool</th><th>int16</th><th>float64</th><th>float32</th><th>float32</th><th>float32</th><th>int16</th><th>float64</th><th>float32</th><th>float32</th><th>float32</th><th>int16</th><th>float64</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>int16</th><th>int16</th><th>int16</th><th>int16</th><th>int16</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>int16</th><th>int16</th><th>int16</th><th>int16</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>int16</th><th>float32</th><th>float32</th><th>int16</th><th>float32</th><th>float32</th><th>int16</th><th>float32</th><th>object</th><th>float64</th><th>float64</th><th>float64</th><th>float64</th><th>bool</th><th>bool</th><th>int16</th><th>bool</th><th>bool</th><th>bool</th><th>bool</th><th>bool</th><th>bool</th><th>bool</th><th>bool</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>object</th></tr></thead>
<tr><td>1636148068921376768</td><td>Gaia DR3 458018340407727104</td><td>458018340407727104</td><td>831950515</td><td>2016.0</td><td>37.87447894404984</td><td>0.015150387</td><td>57.37870558540406</td><td>0.01780867</td><td>50.00541479563848</td><td>0.023927981</td><td>2089.83</td><td>1115.7747</td><td>1115.7192342216508</td><td>0.020135028</td><td>11.126863327716851</td><td>0.021904316</td><td>0.15360849</td><td>0.1357884</td><td>-0.19699116</td><td>-0.34400034</td><td>0.039431885</td><td>-0.3336701</td><td>-0.42438364</td><td>0.050697062</td><td>0.017145477</td><td>-0.024663059</td><td>516</td><td>516</td><td>512</td><td>4</td><td>14.778097</td><td>3696.4827</td><td>0.20194183</td><td>78.39548</td><td>31</td><td>False</td><td>1.2682395</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>59</td><td>21</td><td>0.03221005</td><td>67</td><td>37</td><td>0</td><td>0.010275437</td><td>3.9688818</td><td>2</td><td>0</td><td>1.4975666</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>False</td><td>515</td><td>293388.13180825405</td><td>118.78259</td><td>2469.959</td><td>12.018761</td><td>65</td><td>53947.04189213484</td><td>66.93241</td><td>805.9928</td><td>13.508623</td><td>63</td><td>372479.76596460125</td><td>167.4906</td><td>2223.8845</td><td>10.820139</td><td>1.4534563</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2.6884842</td><td>1.4898624</td><td>1.1986217</td><td>1.5972129</td><td>0.29962748</td><td>1</td><td>26</td><td>7</td><td>14</td><td>79.03226</td><td>-0.6232322</td><td>0.468219</td><td>932.6956</td><td>3.5594358</td><td>3500.0</td><td>4.0</td><td>0.0</td><td>440</td><td>9.075809</td><td>3.1903214</td><td>18</td><td>10.253924</td><td>0.0065138764</td><td>19</td><td>82.37111</td><td>NOT_AVAILABLE</td><td>136.12588680712437</td><td>-2.907308665094729</td><td>56.325888086901905</td><td>39.874060729788916</td><td>False</td><td>False</td><td>0</td><td>True</td><td>True</td><td>True</td><td>False</td><td>False</td><td>False</td><td>True</td><td>False</td><td>1.0202512e-11</td><td>5.0961034e-11</td><td>1.0000273e-06</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td></td></tr>
<tr><td>1636148068921376768</td><td>Gaia DR3 2673992663636347520</td><td>2673992663636347520</td><td>80684491</td><td>2016.0</td><td>327.8635523883638</td><td>0.036881894</td><td>-1.4538921085791252</td><td>0.034450177</td><td>50.0364884312193</td><td>0.040332425</td><td>1240.602</td><td>214.22371</td><td>213.42468947925627</td><td>0.047064923</td><td>18.485081759202735</td><td>0.042042043</td><td>0.25638336</td><td>0.18722051</td><td>-0.07092278</td><td>-0.27526748</td><td>0.122030206</td><td>-0.21398196</td><td>-0.5571672</td><td>-0.22492963</td><td>-0.29449576</td><td>0.2013386</td><td>202</td><td>0</td><td>201</td><td>1</td><td>0.7136408</td><td>430.1806</td><td>0.22516978</td><td>11.070151</td><td>31</td><td>False</td><td>1.1933515</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>23</td><td>14</td><td>0.07165638</td><td>25</td><td>11</td><td>0</td><td>0.22495106</td><td>162.48</td><td>0</td><td>0</td><td>1.0345395</td><td>0.20867826</td><td>0.25083607</td><td>0.16893198</td><td>0.72327703</td><td>-50.602245</td><td>-15.347554</td><td>-46.703465</td><td>23.75228</td><td>False</td><td>209</td><td>21768.572889626033</td><td>14.7179</td><td>1479.0542</td><td>14.842792</td><td>21</td><td>1852.5878200730342</td><td>11.262805</td><td>164.48724</td><td>17.169096</td><td>20</td><td>32569.87753348949</td><td>45.66329</td><td>713.2618</td><td>13.465856</td><td>1.5812918</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3.7032404</td><td>2.3263044</td><td>1.376936</td><td>-1.328453</td><td>4.116577</td><td>2</td><td>10</td><td>0</td><td>7</td><td>6.2367735</td><td>--</td><td>--</td><td>589.65485</td><td>--</td><td>3300.0</td><td>3.5</td><td>0.0</td><td>111</td><td>--</td><td>--</td><td>--</td><td>12.915806</td><td>0.041672494</td><td>10</td><td>--</td><td>NOT_AVAILABLE</td><td>55.913788696899076</td><td>-39.84613855914817</td><td>329.53292199901597</td><td>10.850407313856962</td><td>False</td><td>False</td><td>0</td><td>True</td><td>True</td><td>False</td><td>False</td><td>False</td><td>False</td><td>True</td><td>False</td><td>1.0204125e-13</td><td>5.09691e-13</td><td>0.99999815</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td></td></tr>
</table></div>




```python
## Data Visualization
import matplotlib.pyplot as plt

bp_rp = result['phot_bp_mean_mag'] - result['phot_rp_mean_mag']
fig, ax = plt.subplots(1,1,figsize=(7,7))
ax.hexbin(bp_rp, result['phot_g_mean_mag'],bins='log', mincnt=1,  color= 'red')
ax.set_xlabel(r"$G_{BP} - G_{RP}$")
ax.set_ylabel(r"G")
ax.set_title("Hertzsprung Russell Diagram Before Filtering from Gaia3")
ax.invert_yaxis()
plt.show()
```


    
![png](output_15_0.png)
    



```python
## Search article from NASA ADS (Astrophysics Data System)
from astroquery.simbad import Simbad

result_table = Simbad.query_object("M31")
# print result table with all columns
print(result_table.columns)
# print the result table with the selected columns
print(result_table['COO_BIBCODE'])
```

    <TableColumns names=('MAIN_ID','RA','DEC','RA_PREC','DEC_PREC','COO_ERR_MAJA','COO_ERR_MINA','COO_ERR_ANGLE','COO_QUAL','COO_WAVELENGTH','COO_BIBCODE','SCRIPT_NUMBER_ID')>
        COO_BIBCODE    
    -------------------
    2006AJ....131.1163S


## Conclusion
In this article, we introduced Astropy and Astroquery, two powerful Python libraries for astronomical data analysis. We showed you how to use these libraries with step-by-step installation, usage and sample code. We also provided some useful tips and tricks to help you get started with astronomical data analysis using Python.


Full code is available on [GitHub](!https://github.com/cevheri/intro-to-astro)

## References

- [Astropy Source Code](!https://github.com/astropy/astropy)
- [Astropy Tutorial](!https://github.com/astropy/astropy-tutorials)
- [Astropy](!https://docs.astropy.org/en/stable/)
- [Astropy Learn](!https://learn.astropy.org/)
- [Astroquery](!https://astroquery.readthedocs.io/en/latest/)
- [ESA Gaia](!https://www.cosmos.esa.int/web/gaia/dr3)
- [NASA ADS](!https://ui.adsabs.harvard.edu/)
- [Astroquery Gaia](!https://astroquery.readthedocs.io/en/latest/gaia/gaia.html)
- [Astropy intro](!https://philuttley.github.io/prog4aa_lesson2/09-astropyintro/index.html)

---
### Introduction to Astronomy Research Course

- [Intro-2-Astro](!https://github.com/howardisaacson/Intro-to-Astro2024)
 

<img src="https://raw.githubusercontent.com/howardisaacson/Intro-to-Astro2024/main/Web_Banner.gif">




```python
## Deploy to Medium
# !pip install jupyter-to-medium
!jupyter bundlerextension enable --py jupyter_to_medium._bundler --sys-prefix
```

    usage: jupyter [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                   [--paths] [--json] [--debug]
                   [subcommand]
    
    Jupyter: Interactive Computing
    
    positional arguments:
      subcommand     the subcommand to launch
    
    options:
      -h, --help     show this help message and exit
      --version      show the versions of core jupyter packages and exit
      --config-dir   show Jupyter config dir
      --data-dir     show Jupyter data dir
      --runtime-dir  show Jupyter runtime dir
      --paths        show all Jupyter paths. Add --json for machine-readable
                     format.
      --json         output paths as machine-readable json
      --debug        output debug information about paths
    
    Available subcommands: console dejavu events execute kernel kernelspec lab
    labextension labhub migrate nbconvert notebook qtconsole run server
    troubleshoot trust
    
    Jupyter command `jupyter-bundlerextension` not found.

