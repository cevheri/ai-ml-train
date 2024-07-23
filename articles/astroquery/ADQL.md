# Gaia dataset and queries with ADQL (Astronomical Data Query Language)

## Overview
ADQL is a specialized variant of the SQL query language adapted for accessing the astronomical datasets of the virtual observatory, via the Table access protocol (TAP). ADQL is dedigned to handle large datasets distributed over several locations, while not retrieving data that is not needed.

## Language
ADQL is a query language that allows data to be retrieved via a single command, the select statement, which is designed to perform as the select statement in the SQL language. ADQL has extensions designed to improve handling of astronomical data such as spherical co-ordinates that are not handled by standard SQL.

## Example

```sql
SELECT source_id, ra, dec
FROM gaiadr1.tgas_source
WHERE phot_g_mean_flux > 13
```

##  Basic Queries with Gaia Database wit GUI
The Gaia archive can be found here: https://gea.esac.esa.int/archive/. Click on search tab. Search tab has tree options: Basic, Advanced(ADQL) and Query Results.
<img src="gaia-archive-search-01.png"> 



## Basic Query with ADQL on jupyter notebook
NiceToHave - Astropy and Astroquery with jupyter notebook: https://cevheri.medium.com/astronomical-data-analysis-with-python-using-astropy-and-astroquery-ff7857588c5f

### Install the necessary libraries


```python
!pip install astroquery
```

### Import the necessary libraries


```python
from astroquery.gaia import Gaia
import pandas as pd
```

#### ADQL Query for 10.000 closest stars


```python
print("Querying Gaia database...")
query = """
SELECT TOP 10000
       source_id,
       parallax,
       parallax_error,
       phot_g_mean_mag,
       bp_rp,
       ra,
       ra_error
 FROM gaiadr3.gaia_source
WHERE parallax > 0
ORDER BY parallax DESC
"""
print("ADQL Query: ", query)
path = 'closest_10000.csv'
job = Gaia.launch_job(query, output_file=path, output_format='csv', dump_to_file=True)
print(f"Job finished. Data saved to {path}. Job Info: {job}")
```

    Querying Gaia database...
    ADQL Query:  
    SELECT TOP 10000
           source_id,
           parallax,
           parallax_error,
           phot_g_mean_mag,
           bp_rp,
           ra,
           ra_error
     FROM gaiadr3.gaia_source
    WHERE parallax > 0
    ORDER BY parallax DESC
    
    Job finished. Data saved to closest_10000.csv. Job Info: Jobid: None
    Phase: COMPLETED
    Owner: None
    Output file: closest_10000.csv
    Results: None


#### Explanation of the query
* **source_id**: Unique identifier for each star.
* **parallax**: The apparent shift of the star due to Earth's orbit around the Sun, used to measure distance.
* **parallax_error**: The error in the parallax measurement.
* **phot_g_mean_mag**: The mean magnitude in the G band.
* **bp_rp**: The color index, representing the difference between the blue and red photometric bands.
* **ra**: Right Ascension, the celestial equivalent of longitude.
* **ra_error**: The error in the right ascension measurement.

#### EDA - Exploratory Data Analysis

##### Load data


```python
df = pd.read_csv(path)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source_id</th>
      <th>parallax</th>
      <th>parallax_error</th>
      <th>phot_g_mean_mag</th>
      <th>bp_rp</th>
      <th>ra</th>
      <th>ra_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5853498713190525696</td>
      <td>768.066539</td>
      <td>0.049873</td>
      <td>8.984749</td>
      <td>3.804580</td>
      <td>217.392321</td>
      <td>0.023999</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4472832130942575872</td>
      <td>546.975940</td>
      <td>0.040116</td>
      <td>8.193974</td>
      <td>2.833697</td>
      <td>269.448503</td>
      <td>0.026239</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3864972938605115520</td>
      <td>415.179416</td>
      <td>0.068371</td>
      <td>11.038391</td>
      <td>4.184836</td>
      <td>164.103190</td>
      <td>0.066837</td>
    </tr>
    <tr>
      <th>3</th>
      <td>762815470562110464</td>
      <td>392.752945</td>
      <td>0.032067</td>
      <td>6.551172</td>
      <td>2.215609</td>
      <td>165.830960</td>
      <td>0.024126</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2947050466531873024</td>
      <td>374.489589</td>
      <td>0.231335</td>
      <td>8.524133</td>
      <td>-0.278427</td>
      <td>101.286626</td>
      <td>0.164834</td>
    </tr>
  </tbody>
</table>
</div>



##### Data Description


```python
df.shape
```




    (10000, 7)



##### Data info


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 7 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   source_id        10000 non-null  int64  
     1   parallax         10000 non-null  float64
     2   parallax_error   10000 non-null  float64
     3   phot_g_mean_mag  9985 non-null   float64
     4   bp_rp            9632 non-null   float64
     5   ra               10000 non-null  float64
     6   ra_error         10000 non-null  float64
    dtypes: float64(6), int64(1)
    memory usage: 547.0 KB


#### Data Visualization

##### Import the necessary libraries


```python
import matplotlib.pyplot as plt
```

##### Visualize the data


```python
plt.figure(figsize=(10, 6))
plt.scatter(df['ra'], df['parallax'], s=0.1)
plt.xlabel('RA')
plt.ylabel('Parallax')
plt.title('Parallax vs RA')
plt.show()

```


    
![png](output_25_0.png)
    


#### Explanation this plot
This plot represents the relationship between the right ascension (RA) and parallax of stars from the Gaia DR3 dataset.


#### 1000 closest stars - plot

##### Import the necessary libraries


```python
import matplotlib.pyplot as plt
import numpy as np
```

###### calculate the distance
* Calculate distance in parsecs
* Calculate absolute g-band photometric magnitude
* Calculate absolute magnitude



```python
df['distance'] = 1 / (df['parallax'] * 1e-3)
df['abs_g'] = df['phot_g_mean_mag'] + 5 + 5 * np.log10(df['distance'])
df['abs_mag'] = df['phot_g_mean_mag'] + 5 * (np.log10(df['distance']) - 1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source_id</th>
      <th>parallax</th>
      <th>parallax_error</th>
      <th>phot_g_mean_mag</th>
      <th>bp_rp</th>
      <th>ra</th>
      <th>ra_error</th>
      <th>distance</th>
      <th>abs_g</th>
      <th>abs_mag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5853498713190525696</td>
      <td>768.066539</td>
      <td>0.049873</td>
      <td>8.984749</td>
      <td>3.804580</td>
      <td>217.392321</td>
      <td>0.023999</td>
      <td>1.301971</td>
      <td>14.557755</td>
      <td>4.557755</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4472832130942575872</td>
      <td>546.975940</td>
      <td>0.040116</td>
      <td>8.193974</td>
      <td>2.833697</td>
      <td>269.448503</td>
      <td>0.026239</td>
      <td>1.828234</td>
      <td>14.504133</td>
      <td>4.504133</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3864972938605115520</td>
      <td>415.179416</td>
      <td>0.068371</td>
      <td>11.038391</td>
      <td>4.184836</td>
      <td>164.103190</td>
      <td>0.066837</td>
      <td>2.408597</td>
      <td>17.947212</td>
      <td>7.947212</td>
    </tr>
    <tr>
      <th>3</th>
      <td>762815470562110464</td>
      <td>392.752945</td>
      <td>0.032067</td>
      <td>6.551172</td>
      <td>2.215609</td>
      <td>165.830960</td>
      <td>0.024126</td>
      <td>2.546130</td>
      <td>13.580575</td>
      <td>3.580575</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2947050466531873024</td>
      <td>374.489589</td>
      <td>0.231335</td>
      <td>8.524133</td>
      <td>-0.278427</td>
      <td>101.286626</td>
      <td>0.164834</td>
      <td>2.670301</td>
      <td>15.656934</td>
      <td>5.656934</td>
    </tr>
  </tbody>
</table>
</div>



##### Plot bp_rp vs abs_mag


```python
plt.figure(figsize=(10, 6))
plt.scatter(df['bp_rp'], df['abs_mag'], s=0.1)
plt.xlabel('bp_rp')
plt.ylabel('Absolute Magnitude')
plt.title('Color Index vs Absolute Magnitude')
plt.show()
```


    
![png](output_33_0.png)
    


#### Explanation of the plot
This plot represents the relationship between the color index (bp_rp) and the absolute magnitude of stars from the Gaia DR3 dataset.

##### Plot parallax vs distance


```python
plt.figure(figsize=(10, 6))
plt.scatter(df['parallax'], df['distance'], s=0.1)
plt.xlabel('Parallax')
plt.ylabel('Distance (parsecs)')
plt.title('Parallax vs Distance')
plt.show()
```


    
![png](output_36_0.png)
    


#### Explanation of the plot
This plot represents the relationship between the parallax and distance of stars from the Gaia DR3 dataset.

##### Plot parallax vs abs_mag



```python
plt.figure(figsize=(10, 6))
plt.scatter(df['parallax'], df['abs_mag'], s=0.1)
plt.xlabel('Parallax')
plt.ylabel('Absolute Magnitude')
plt.title('Parallax vs Absolute Magnitude')
plt.show()

```


    
![png](output_39_0.png)
    


#### Explanation of the plot
This plot represents the relationship between the parallax and absolute magnitude of stars from the Gaia DR3 dataset.

## Conclusion
In this notebook, we have used the Gaia DR3 dataset to query the 10,000 closest stars to Earth using ADQL. We have performed exploratory data analysis (EDA) on the retrieved data, including visualizations of the relationships between various parameters such as right ascension, parallax, color index, absolute magnitude, and distance. These visualizations provide insights into the properties of stars in our cosmic neighborhood, and demonstrate the power of ADQL queries for accessing and analyzing astronomical data.

## References
1. ADQL Wiki: https://en.wikipedia.org/wiki/Astronomical_Data_Query_Language
2. Gaia Archive: https://gea.esac.esa.int/archive/
2. Gaia DR3: https://www.cosmos.esa.int/web/gaia/dr3
3. Gaia TAP + Astroquery: https://astroquery.readthedocs.io/en/latest/gaia/gaia.html
4. ADQL: https://www.ivoa.net/documents/ADQL/20180112/PR-ADQL-2.1-20180112.html
5. Intro2Astro - Gaia Introduction: https://github.com/howardisaacson/Intro-to-Astro2024/blob/main/Week4_TESS_Gaia/GaiaTutorialAssignment.ipynb

## Further Reading
* https://www.gaia.ac.uk/data/gaia-data-release-1/adql-cookbook
* https://en.wikipedia.org/wiki/Transiting_Exoplanet_Survey_Satellite
* https://www.karar.com/yazarlar/zafer-acar/tess-ile-yeni-dunyalar-kesfetmek-6853#google_vignette
* https://www.esa.int/

## Astropy and Astroquery with jupyter notebook
https://cevheri.medium.com/astronomical-data-analysis-with-python-using-astropy-and-astroquery-ff7857588c5f

