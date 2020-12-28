Repo with code, data, and results for the paper "Real-Time Estimation of Disease Activity in Emerging Outbreaks using Internet 
Search Information." Structure as follows:

- data: Digitized epidemiological reports and Google trends time-series for each outbreak, labeled by country. In each case, the file 
{country}reports.csv contains the digitized reports by release date and the file {country}predictors.csv contains each of the Google 
trends time-series for related keywords.
- results: CSV files containing the time-series predictions of each model for each outbreak (using evaluation method #2 in the paper -- not
paying attention to report release dates and just assuming a standard reporting delay). One file for each outbreak for each of 1 and 2 
weeks assumed reporting delay; each file contains the ground truth data & predictions by each of the three models.
- OutbreakMethods.ipynb: All code
- figures: .png files for figures in paper
