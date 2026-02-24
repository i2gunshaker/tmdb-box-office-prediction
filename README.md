
# Predicting Box Office Success: A Regression Analysis

This repository contains the code and final report for **SDS 301: Modern Regression Analysis**. In this project, our team built a multiple linear regression model to predict worldwide box office revenue using pre-release movie characteristics.

## 📌 Project Overview
The global film industry is high-risk, with extreme variance in financial outcomes. This project aims to forecast commercial performance using the **TMDB Box Office Prediction dataset**. We investigated factors such as production budget, cast/crew size, genres, and release timing to identify key drivers of financial success.

## 📂 Repository Structure
* `docs/SDS_301_final_project_report.pdf`: The comprehensive final project report detailing our Exploratory Data Analysis (EDA), methodology, model diagnostics (heteroskedasticity, omitted variable bias), and conclusions.
* `scripts/R_code.r`: The complete R script containing:
  * Data engineering and cleaning
  * Exploratory Data Analysis (EDA)
  * Modeling, robust Standard Errors (HC3), and Cross-Validation
  * Diagnostic plotting

## 💾 Data Source & Reproducibility
Due to file size limits, the dataset is not hosted in this repository. 

To run the code yourself:
1. Download the `train.csv` file from the [TMDB Box Office Prediction Kaggle Competition](https://www.kaggle.com/competitions/tmdb-box-office-prediction/data?select=train.csv).
2. Create a folder named `data/` in the main directory of this repository.
3. Place the downloaded `train.csv` file into the `data/` folder.
4. Run the `R_code.r` script.

*Note: We exclusively utilized the training set ($n=3,000$) as it contains the target variable (revenue) necessary to train and evaluate our regression models.*

## 🛠️ Tools & Technologies
* **Language:** R
* **Libraries:** `tidyverse`, `lubridate`, `stringr`, `ggplot2`
* **Techniques:** Multiple Linear Regression, Feature Engineering, Cross-Validation, Heteroskedasticity Diagnostics (Breusch–Pagan test)

## 📊 Key Findings
* **Budget is the strongest predictor:** A film's production budget has the most significant association with revenue, though it operates on a logarithmic scale.
* **Model Performance:** Our final model explains approximately 52% of the variance in log-revenue ($R^2 \approx 0.52$). 
* **Prediction Variance:** Diagnostic plots revealed a "fan shape" in residuals, indicating that blockbuster revenues are fundamentally harder to predict precisely than smaller-scale films.

## 👥 Authors
* Aitugan Shagyr
* Makhabbat Batyrova
* Aisultan Zhakupbayev
* Raiymbek Arysbek
