# ==============================================================================
# SDS 301 Final Project — TMDB Box Office Prediction
# File contains multiple blocks:
#   1) Data engineering (build cleaned train/test CSVs)
#   2) EDA (plots + numeric summaries)
#   3) Modeling/diagnostics/selection (m1/m2/m3 + robust SE + CV + exports)
#
# Note: Some later blocks assume objects created earlier (e.g., df_plot, df_m, m1).
# If you run top-to-bottom in one go, run the EDA block first (creates df_plot),
# then the modeling block (creates df_m and models), then diagnostics/Section 5.
# ==============================================================================


# ==============================================================================
# SDS 301 Final Project - Data Engineering Script
# Purpose: Load, clean, and feature-engineer the TMDB Box Office dataset.
# ==============================================================================

# --- STEP 1: SETUP & LIBRARIES ---
# tidyverse = data manipulation; lubridate = dates; stringr = regex/string features
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(lubridate)) install.packages("lubridate")
if(!require(stringr)) install.packages("stringr")

library(tidyverse)
library(lubridate)
library(stringr)

# Set working directory (update to your folder)
setwd("/Users/i2gunshaker/Downloads/")

# --- STEP 2: LOAD & COMBINE DATA ---
# Train + test are processed together so feature extraction and cleaning are identical.
print("Loading data...")
train <- read_csv("train.csv")
test <- read_csv("test.csv")

# Label rows so we can split back after processing
train$set <- "train"
test$set <- "test"

# Bind rows (test has revenue = NA by design)
full_data <- bind_rows(train, test)

# --- STEP 3: FEATURE ENGINEERING ---
# Several columns are stored as JSON-like strings with single quotes.
# To keep the workflow stable, we extract simple signals using regex counts/first match.
print("Engineering features...")

full_data <- full_data %>%
  mutate(
    # 1) Counts as proxies for "scale/complexity" of the movie metadata
    # Count occurrences of repeated keys inside the string representation
    cast_count = str_count(cast, "'cast_id':"),
    crew_count = str_count(crew, "'credit_id':"),
    production_count = str_count(production_companies, "'id':"),
    genre_count = str_count(genres, "'id':"),
    
    # 2) Primary genre = first genre name listed
    # Extract the first text that appears after "name': '"
    primary_genre = str_extract(genres, "(?<=name': ').*?(?=')"),
    
    # 3) Date processing
    # Parse release_date strings like "2/20/15" into Date
    release_date_parsed = mdy(release_date),
    
    # Basic calendar features used in regression
    release_year = year(release_date_parsed),
    release_month = month(release_date_parsed)
  )

# --- STEP 4: DATA CLEANING & IMPUTATION ---

# Fix 1: Two-digit year mapping
# Some parsed years can land in the future; those are treated as 1900s entries.
full_data <- full_data %>%
  mutate(
    release_year = ifelse(release_year > 2025, release_year - 100, release_year)
  )

# Fix 2: Missing categorical values
# Keep a valid level so factor conversion/model.matrix won't drop rows.
full_data$primary_genre[is.na(full_data$primary_genre)] <- "Unknown"

# Fix 3: Missing numerical values
# Runtime: mean imputation (simple baseline; acceptable for first-pass modeling).
full_data$runtime[is.na(full_data$runtime)] <- mean(full_data$runtime, na.rm=TRUE)

# Counts: treat missing as 0 (empty/missing string implies no listed entries)
full_data$production_count[is.na(full_data$production_count)] <- 0
full_data$crew_count[is.na(full_data$crew_count)] <- 0

# --- STEP 5: DROP RAW TEXT COLUMNS ---
# After extracting numeric/compact features, remove high-cardinality text fields.
cols_to_drop <- c(
  "genres", "cast", "crew", "production_companies",
  "production_countries", "Keywords", "spoken_languages",
  "belongs_to_collection", "tagline", "homepage", "imdb_id",
  "poster_path", "overview", "status", "original_title"
)

# Keep 'original_language' or 'title' if you plan to use them; otherwise dropping is fine.
clean_data <- full_data %>% select(-any_of(cols_to_drop))

# --- STEP 6: SAVE FINAL FILES ---
# Split back into Train and Test
final_train_clean <- clean_data %>% filter(set == "train") %>% select(-set)
final_test_clean <- clean_data %>% filter(set == "test") %>% select(-set)

# Export cleaned train (used for EDA/modeling)
write_csv(final_train_clean, "final_train_clean2.csv")

# --- FINAL VERIFICATION ---
print("Processing Complete.")
print("Final Train Dimensions:")
print(dim(final_train_clean))
print("Columns Available for Analysis:")
print(colnames(final_train_clean))



# ==============================================================================
# SDS 301 — EDA (Complete: numeric summaries + distributions + relationships)
# TMDB Box Office Prediction (cleaned train)
# Outputs:
#   - figures/*.png
#   - outputs/eda_numeric_summary.csv
# ==============================================================================

# ----------------------------
# 0) Setup
# ----------------------------
options(stringsAsFactors = FALSE)
set.seed(301)

# ----------------------------
# 1) Libraries
# ----------------------------
pkgs <- c("tidyverse", "corrplot", "scales", "patchwork", "viridis", "readr")

to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install) > 0) install.packages(to_install)

library(tidyverse)
library(corrplot)
library(scales)
library(patchwork)
library(viridis)
library(readr)

# ----------------------------
# 2) Paths (edit if needed)
# ----------------------------

data_path <- "/Users/i2gunshaker/Downloads/final_train_clean2.csv"

if (!file.exists(data_path)) {
  stop(
    paste0(
      "File not found at: ", data_path, "\n",
      "Fix 'data_path' at the top of this script."
    )
  )
}

dir.create("figures", showWarnings = FALSE, recursive = TRUE)
dir.create("outputs", showWarnings = FALSE, recursive = TRUE)

# ----------------------------
# 3) Load data (robust parsing)
# ----------------------------
df <- readr::read_csv(data_path, show_col_types = FALSE)

# ----------------------------
# 4) Ensure required columns exist + derive release_month if needed
# ----------------------------
required_cols <- c(
  "revenue", "budget", "popularity", "runtime",
  "cast_count", "crew_count", "release_year",
  "primary_genre"
)

missing_cols <- setdiff(required_cols, names(df))

# If release_month was not saved, derive it from release_date (if available)
if (!("release_month" %in% names(df))) {
  if ("release_date" %in% names(df)) {
    df <- df %>%
      mutate(release_date = as.Date(release_date)) %>%
      mutate(release_month = as.integer(format(release_date, "%m")))
  } else {
    missing_cols <- union(missing_cols, "release_month (or release_date)")
  }
}

if (length(missing_cols) > 0) {
  stop(
    paste0(
      "Missing required columns:\n- ",
      paste(missing_cols, collapse = "\n- "),
      "\n\nFix your CSV column names or update 'required_cols'."
    )
  )
}

# ----------------------------
# 5) Force numeric parsing (prevents coercion surprises)
# ----------------------------
num_cols <- c(
  "revenue", "budget", "popularity", "runtime",
  "cast_count", "crew_count", "release_year", "release_month"
)

df <- df %>%
  mutate(across(any_of(num_cols), ~ readr::parse_number(as.character(.)))) %>%
  mutate(primary_genre = as.character(primary_genre))

# ----------------------------
# 6) Global plotting theme (keeps visuals consistent)
# ----------------------------
theme_set(
  theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(color = "gray40", size = 10),
      axis.title = element_text(face = "bold", size = 10),
      panel.grid.minor = element_blank(),
      legend.position = "bottom"
    )
)

# Helper: convert log1p scale tick positions back to $ labels
log_dollar_labels <- function(x) {
  label_dollar(scale_cut = cut_short_scale())(expm1(x))
}

# ----------------------------
# 7) EDA feature engineering (derived variables used only for plots/models)
# ----------------------------
df_plot <- df %>%
  mutate(
    # log1p stabilizes heavy right tails and keeps zeros defined
    log_revenue = log1p(revenue),
    
    # Budget is frequently missing/zero; treat budget<=0 as "unreported"
    budget_missing = if_else(is.na(budget) | budget <= 0, 1L, 0L),
    
    # Keep missing budgets as NA on the log scale (avoid pretending they are $0)
    log_budget_reported = if_else(budget_missing == 0L, log1p(budget), NA_real_),
    
    budget_status = if_else(budget_missing == 0L, "Budget > 0", "Budget = 0 / Missing"),
    
    # Month factor for seasonality plots
    month_label = factor(release_month, levels = 1:12, labels = month.abb)
  )

# ==============================================================================
# OUTPUT 1: Summary statistics (all numeric variables + missingness)
# ==============================================================================
numeric_vars <- names(df_plot)[sapply(df_plot, is.numeric)]

num_summary <- purrr::map_dfr(numeric_vars, function(v) {
  x <- df_plot[[v]]
  tibble(
    variable = v,
    n = sum(!is.na(x)),
    missing_n = sum(is.na(x)),
    missing_pct = mean(is.na(x)) * 100,
    mean = mean(x, na.rm = TRUE),
    sd = sd(x, na.rm = TRUE),
    median = median(x, na.rm = TRUE),
    iqr = IQR(x, na.rm = TRUE),
    min = suppressWarnings(min(x, na.rm = TRUE)),
    max = suppressWarnings(max(x, na.rm = TRUE))
  )
}) %>% arrange(desc(missing_pct))

# Budget-specific check: % of rows with budget <= 0 (even if not NA)
budget_zero_pct <- mean(!is.na(df_plot$budget) & df_plot$budget <= 0) * 100

readr::write_csv(num_summary, "outputs/eda_numeric_summary.csv")
cat("\nSaved: outputs/eda_numeric_summary.csv\n")
cat(sprintf("Budget <= 0 (treated as missing) percentage: %.1f%%\n\n", budget_zero_pct))

# ==============================================================================
# OUTPUT 2: Missingness overview (bar chart)
# ==============================================================================
miss_tbl <- df_plot %>%
  summarise(across(everything(), ~ mean(is.na(.)) * 100)) %>%
  pivot_longer(everything(), names_to = "var", values_to = "missing_pct") %>%
  arrange(desc(missing_pct))

pMISS <- ggplot(miss_tbl, aes(x = reorder(var, missing_pct), y = missing_pct)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Missingness by variable",
    subtitle = "Percent of NA values (budget missingness is also tracked via budget<=0)",
    x = NULL, y = "% missing"
  )

ggsave("figures/plot_missingness.png", pMISS, width = 9, height = 6, dpi = 300)
print(pMISS)

# ==============================================================================
# PLOT A: Revenue distribution (original vs log1p)
# ==============================================================================
pA1 <- ggplot(df_plot, aes(x = revenue)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white", alpha = 0.9) +
  scale_x_continuous(labels = label_dollar(scale_cut = cut_short_scale())) +
  coord_cartesian(xlim = c(0, quantile(df_plot$revenue, 0.99, na.rm = TRUE))) +
  labs(
    title = "Revenue distribution (original scale)",
    subtitle = "Zoomed to 99th percentile to show the main mass",
    x = "Revenue ($)", y = "Count"
  )

pA2 <- ggplot(df_plot, aes(x = log_revenue)) +
  geom_histogram(bins = 50, fill = "darkgreen", color = "white", alpha = 0.9) +
  scale_x_continuous(
    breaks = log1p(c(1e5, 1e6, 1e7, 1e8, 1e9)),
    labels = c("$100k", "$1M", "$10M", "$100M", "$1B")
  ) +
  labs(
    title = "Revenue distribution (log1p scale)",
    subtitle = "More symmetric after transformation",
    x = "log(1 + revenue)", y = "Count"
  )

pA <- pA1 + pA2 + patchwork::plot_layout(ncol = 2)
ggsave("figures/plotA_revenue_distributions.png", pA, width = 12, height = 5, dpi = 300)
print(pA)

# ==============================================================================
# PLOT B: Budget vs revenue (density heatmap) — reported budgets only
# ==============================================================================
df_valid <- df_plot %>% filter(!is.na(log_budget_reported), revenue > 0)

r2_val <- round(summary(lm(log_revenue ~ log_budget_reported, data = df_valid))$r.squared, 2)

pB <- ggplot(df_valid, aes(x = log_budget_reported, y = log_revenue)) +
  geom_bin2d(bins = 60) +
  scale_fill_viridis_c(option = "magma", name = "Count") +
  geom_smooth(method = "lm", color = "cyan", linewidth = 1) +
  scale_x_continuous(
    breaks = log1p(c(1e5, 1e6, 1e7, 1e8, 2e8)),
    labels = log_dollar_labels
  ) +
  scale_y_continuous(
    breaks = log1p(c(1e5, 1e6, 1e7, 1e8, 1e9)),
    labels = log_dollar_labels
  ) +
  guides(x = guide_axis(n.dodge = 2), y = guide_axis(n.dodge = 2)) +
  labs(
    title = "Budget vs revenue (density) — reported budgets only",
    subtitle = paste0("Linear fit on log1p scale (R² = ", r2_val, ")"),
    x = "Budget ($) [log1p scale with $ ticks]",
    y = "Revenue ($) [log1p scale with $ ticks]"
  )

ggsave("figures/plotB_budget_vs_revenue_density.png", pB, width = 10, height = 7, dpi = 300)
print(pB)

# ==============================================================================
# PLOT B2: Log-revenue by budget availability
# ==============================================================================
pB2 <- ggplot(df_plot, aes(x = budget_status, y = log_revenue, fill = budget_status)) +
  geom_boxplot(outlier.alpha = 0.2, show.legend = FALSE) +
  scale_fill_manual(values = c("gray70", "cyan")) +
  scale_y_continuous(
    breaks = log1p(c(1e5, 1e6, 1e7, 1e8, 1e9)),
    labels = c("$100k", "$1M", "$10M", "$100M", "$1B")
  ) +
  labs(
    title = "Log-revenue by budget status",
    subtitle = "Movies with missing/zero budgets tend to earn less",
    x = NULL, y = "log(1 + revenue) [with $ ticks]"
  ) +
  theme(axis.text.x = element_text(face = "bold", size = 11))

ggsave("figures/plotB2_revenue_by_budget_status.png", pB2, width = 9, height = 6, dpi = 300)
print(pB2)

# ==============================================================================
# PLOT C: Seasonality (geometric mean + median + counts)
# ==============================================================================
season_stats <- df_plot %>%
  filter(!is.na(month_label), revenue > 0) %>%
  group_by(month_label) %>%
  summarise(
    n = n(),
    geo_mean = exp(mean(log1p(revenue))) - 1,
    median = median(revenue),
    .groups = "drop"
  )

pC <- ggplot(season_stats, aes(x = month_label, y = geo_mean)) +
  geom_col(fill = "cornflowerblue", alpha = 0.8) +
  geom_point(aes(y = median), color = "darkblue", size = 2) +
  geom_text(aes(label = n), vjust = -0.5, size = 3, color = "gray30") +
  scale_y_continuous(labels = label_dollar(scale_cut = cut_short_scale())) +
  labs(
    title = "Seasonality: revenue by month",
    subtitle = "Bars = geometric mean • Dots = median • Labels = count",
    x = NULL, y = "Revenue ($)"
  )

ggsave("figures/plotC_seasonality_by_month.png", pC, width = 10, height = 6, dpi = 300)
print(pC)

# ==============================================================================
# EXTRA 1: Distributions for key numeric predictors
# ==============================================================================
num_vars_for_hist <- c("popularity", "runtime", "cast_count", "crew_count", "release_year")

pUNI <- df_plot %>%
  pivot_longer(all_of(num_vars_for_hist), names_to = "var", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 40, color = "white") +
  facet_wrap(~ var, scales = "free") +
  labs(
    title = "Distributions of numeric predictors",
    subtitle = "Skew/outliers/range checks",
    x = NULL, y = "Count"
  )

ggsave("figures/extra_univariate_numeric_predictors.png", pUNI, width = 12, height = 7, dpi = 300)
print(pUNI)

# ==============================================================================
# EXTRA 2: Primary genre frequency + log-revenue by genre (top 10)
# ==============================================================================
top_genres <- df_plot %>%
  count(primary_genre, sort = TRUE) %>%
  filter(!is.na(primary_genre)) %>%
  slice_head(n = 10)

pG1 <- ggplot(top_genres, aes(x = reorder(primary_genre, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(title = "Top 10 primary genres", x = NULL, y = "Count")

df_genre <- df_plot %>%
  mutate(primary_genre_top = if_else(primary_genre %in% top_genres$primary_genre,
                                     primary_genre, "Other"))

pG2 <- ggplot(df_genre, aes(x = primary_genre_top, y = log_revenue)) +
  geom_boxplot(outlier.alpha = 0.2) +
  coord_flip() +
  labs(
    title = "Log-revenue by primary genre",
    subtitle = "Top 10 genres shown; remaining grouped as Other",
    x = NULL, y = "log(1 + revenue)"
  )

pG <- pG1 + pG2 + patchwork::plot_layout(ncol = 2)
ggsave("figures/extra_genre_frequency_and_revenue.png", pG, width = 12, height = 6, dpi = 300)
print(pG)

# ==============================================================================
# EXTRA 3: Log-revenue vs predictors (scatter + smooth, faceted)
# ==============================================================================
preds <- c("log_budget_reported", "popularity", "runtime", "cast_count", "crew_count", "release_year")

pSCAT <- df_plot %>%
  pivot_longer(all_of(preds), names_to = "pred", values_to = "x") %>%
  ggplot(aes(x = x, y = log_revenue)) +
  geom_point(alpha = 0.15) +
  geom_smooth(se = FALSE) +
  facet_wrap(~ pred, scales = "free_x") +
  labs(
    title = "Log-revenue vs predictors",
    subtitle = "Nonlinearity/outliers/diminishing returns checks",
    x = NULL, y = "log(1 + revenue)"
  )

ggsave("figures/extra_logrevenue_vs_predictors.png", pSCAT, width = 12, height = 7, dpi = 300)
print(pSCAT)

# ==============================================================================
# PLOT D: Correlation matrix (reported budgets only; avoids zero-budget artifacts)
# ==============================================================================
cor_data <- df_plot %>%
  select(log_revenue, log_budget_reported, popularity, runtime, cast_count, crew_count, release_year) %>%
  na.omit()

# p-value matrix for correlation significance filtering
cor_mtest <- function(mat) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat <- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      p.mat[i, j] <- p.mat[j, i] <- cor.test(mat[, i], mat[, j])$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}

M <- cor(cor_data)
p_mat <- cor_mtest(cor_data)

png("figures/plotD_correlation_matrix.png", width = 1200, height = 900, res = 150)
corrplot(
  M, method = "color", type = "upper", order = "hclust",
  addCoef.col = "black", tl.col = "black", tl.srt = 45,
  p.mat = p_mat, sig.level = 0.05, insig = "blank",
  diag = FALSE, col = COL2("RdBu", 10),
  title = "Correlation matrix (reported budgets only)",
  mar = c(0, 0, 2, 0)
)
dev.off()

cat("\nSaved plots to: figures/\n")
cat("Saved numeric summary table to: outputs/eda_numeric_summary.csv\n")



# ==============================================================================
# DIAGNOSTICS FOR MODEL M1 (Baseline: Budget Only)
# NOTE: This block assumes:
#   - df_m exists (built in the modeling block below)
#   - m1 exists (fit in the modeling block below)
# If running from a fresh session, create df_m + m1 first, then run this section.
# ==============================================================================

# 0. Load Required Libraries
# ------------------------------------------------------------------------------
library(lmtest)
library(sandwich)
library(car)

# 1. Model Performance Check (R-squared)
# ------------------------------------------------------------------------------
# Baseline
m1 <- lm(log_rev ~ log_budget_use + budget_missing, data = df_m)

s_m1 <- summary(m1)

cat("\n--- MODEL PERFORMANCE (R-SQUARED) ---\n")
cat("R-squared:         ", round(s_m1$r.squared, 4), "\n")
cat("Adjusted R-squared:", round(s_m1$adj.r.squared, 4), "\n")
cat("Residual Std Error:", round(s_m1$sigma, 4), "\n")

# 2. Visual Residual Checks (standard lm diagnostic panel)
# ------------------------------------------------------------------------------
par(mfrow = c(2, 2))
plot(m1)
par(mfrow = c(1, 1))

# 3. Heteroscedasticity Test & Robust Standard Errors
# ------------------------------------------------------------------------------
cat("\n--- HETEROSCEDASTICITY CHECKS ---\n")
print(bptest(m1))

cat("\nRobust Coefficients (HC3) - Safe for reporting:\n")
print(coeftest(m1, vcov = vcovHC(m1, type = "HC3")))

# 4. Multicollinearity (VIF)
# ------------------------------------------------------------------------------
cat("\n--- MULTICOLLINEARITY (VIF) ---\n")
print(vif(m1))

# 5. Outliers & Influential Points
# ------------------------------------------------------------------------------
cat("\n--- OUTLIER DETECTION ---\n")
n <- nrow(df_m)

cook_m1 <- cooks.distance(m1)
influential <- which(cook_m1 > 4/n)
cat("Count of influential points (Cook's D > 4/n):", length(influential), "\n")
print(influential)

rstud_m1 <- rstudent(m1)
extreme <- which(abs(rstud_m1) > 3)
cat("\nCount of extreme residuals (|Studentized| > 3):", length(extreme), "\n")
print(extreme)

# 6. Specification Test (nested ANOVA)
# ------------------------------------------------------------------------------
cat("\n--- SPECIFICATION CHECK (Does adding Runtime help?) ---\n")
m1_run1 <- update(m1, . ~ . + poly(runtime, 2, raw = TRUE))
print(anova(m1, m1_run1))



# ==============================================================================
# MODELING BLOCK: define df_m + fit m1/m2 (used by later diagnostics)
# ==============================================================================

#m2
library(tidyverse)
library(car)
library(lmtest)
library(sandwich)

# Modeling dataset (log transforms + centered year + factor month + lump genres)
df_m <- df_plot %>%
  mutate(
    log_rev = log1p(revenue),
    budget_missing = if_else(is.na(budget) | budget <= 0, 1L, 0L),
    log_budget_use = if_else(budget_missing == 1L, 0, log1p(budget)),
    log_pop = log1p(popularity),
    log_cast = log1p(cast_count),
    log_crew = log1p(crew_count),
    release_year_c = release_year - mean(release_year, na.rm = TRUE),
    release_month_f = factor(release_month, levels = 1:12, labels = month.abb),
    # group rare genres to keep the design matrix stable
    primary_genre_top = fct_lump_n(factor(primary_genre), n = 10, other_level = "Other")
  )

# Baseline
m1 <- lm(log_rev ~ log_budget_use + budget_missing, data = df_m)

# Add main numeric predictors
m2 <- lm(log_rev ~ log_budget_use + budget_missing +
           log_pop + runtime + log_cast + log_crew,
         data = df_m)

# Residual diagnostic panel for m2
par(mfrow = c(2,2))
plot(m2, sub.caption = "")
par(mfrow = c(1,1))

# Heteroscedasticity check + robust SE
lmtest::bptest(m2)
coeftest(m2, vcov = vcovHC(m2, type = "HC3"))

# Multicollinearity
car::vif(m2)

# Influence / outliers
n <- nrow(df_m)
cook <- cooks.distance(m2)
which(cook > 4/n)

rstud <- rstudent(m2)
which(abs(rstud) > 3)



############################################################
# Model 3 (m3) — Diagnostics + model selection (concise notes)
# Assumes df_m is ready and contains all variables below.
############################################################

# Output folder (new name to avoid conflicts)
out_dir <- "figs_m3"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# 1) Fit Model 3 (full EDA-driven specification)
m3 <- lm(
  log_rev ~ log_budget_use + budget_missing +
    log_pop + runtime + log_cast + log_crew +
    release_year_c + release_month_f + primary_genre_top,
  data = df_m
)

# 2) Big-four diagnostics (linearity, normality, variance, leverage)
png(file.path(out_dir, "m3_big4.png"), width = 1200, height = 900, res = 150)
par(mfrow = c(2,2)); plot(m3); par(mfrow = c(1,1))
dev.off()

# 3) Test nonlinearity using nested ANOVA (quadratic year, then quadratic runtime)
m3_y2   <- update(m3, . ~ . + poly(release_year_c, 2, raw = TRUE))
m3_run2 <- update(m3_y2, . ~ . + poly(runtime, 2, raw = TRUE))
anova(m3, m3_y2, m3_run2)

# 4) Optional: spline version (flexible functional forms for key continuous predictors)
library(splines)
m3_spl <- update(
  m3,
  . ~ . - log_budget_use - log_pop - log_cast - log_crew +
    ns(log_budget_use, df=4) + ns(log_pop, df=4) +
    ns(log_cast, df=3) + ns(log_crew, df=3)
)
anova(m3, m3_spl)

# 5) Robust inference (HC3) since variance typically increases with fitted revenue
library(lmtest)
library(sandwich)
coeftest(m3, vcov = vcovHC(m3, type = "HC3"))
coeftest(m3_spl, vcov = vcovHC(m3_spl, type = "HC3"))

# 6) Influence check (Cook’s D > 4/n) + sensitivity refit
n <- nrow(model.frame(m3))
cd <- cooks.distance(m3)
idx <- which(cd > 4/n)
idx

m3_sens <- update(m3, data = df_m[-idx, ])
summary(m3)$coef
summary(m3_sens)$coef

# 7) WLS (optional sensitivity): downweight high fitted values
w <- 1 / (fitted(m3)^2)
m3_wls <- lm(formula(m3), data = df_m, weights = w)
summary(m3_wls)

# 8) “Clean” quadratic replacement (swap linear term for polynomial basis)
m3_y2_clean <- update(m3, . ~ . - release_year_c + poly(release_year_c, 2, raw = TRUE))
m3_run2_clean <- update(m3_y2_clean, . ~ . - runtime + poly(runtime, 2, raw = TRUE))

anova(m3, m3_y2_clean, m3_run2_clean)
coeftest(m3_run2_clean, vcov = vcovHC(m3_run2_clean, type = "HC3"))

# Compare fit/complexity
AIC(m3, m3_run2_clean)
BIC(m3, m3_run2_clean)
summary(m3)$adj.r.squared
summary(m3_run2_clean)$adj.r.squared

# 9) WLS for final model (optional): same weight idea
w2 <- 1 / (fitted(m3_run2_clean)^2)
m3_run2_wls <- lm(formula(m3_run2_clean), data = df_m, weights = w2)

png(file.path(out_dir, "m3_run2_wls_big4.png"), width = 1200, height = 900, res = 150)
par(mfrow = c(2,2)); plot(m3_run2_wls); par(mfrow = c(1,1))
dev.off()

# Alternative weight (upweights high fitted) — just a check
w_alt <- fitted(m3_run2_clean)^2
m3_run2_wls_alt <- lm(formula(m3_run2_clean), data = df_m, weights = w_alt)

png(file.path(out_dir, "m3_run2_wls_alt_big4.png"), width = 1200, height = 900, res = 150)
par(mfrow = c(2,2)); plot(m3_run2_wls_alt); par(mfrow = c(1,1))
dev.off()

# 10) Top-10 Cook’s D for final model + sensitivity refit
cd2 <- cooks.distance(m3_run2_clean)
ord <- order(cd2, decreasing = TRUE)
top <- ord[1:10]
top
cd2[top]

m_sens <- update(m3_run2_clean, data = df_m[-top, ])
coeftest(m3_run2_clean, vcov = vcovHC(m3_run2_clean, type="HC3"))
coeftest(m_sens, vcov = vcovHC(m_sens, type="HC3"))

# 11) Save final big-four diagnostics + BP test (heteroskedasticity)
png(file.path(out_dir, "m3_run2_clean_big4.png"), width = 1200, height = 900, res = 150)
par(mfrow = c(2,2)); plot(m3_run2_clean); par(mfrow = c(1,1))
dev.off()

bptest(m3_run2_clean)



# ============================================================
# SDS 301 — Section 5: Final Model + CV + Exports (new folders)
# Assumes df_m is already in memory (cleaned train + engineered features).
# ============================================================

# ---------- 0) Packages ----------
pkgs <- c("lmtest", "sandwich", "splines", "ggplot2")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) install.packages(to_install)

library(lmtest)
library(sandwich)
library(splines)
library(ggplot2)

# ---------- 1) Output folders (separate from EDA figures to stay organized) ----------
OUT_DIR <- "sec5_outputs"
FIG_DIR <- file.path(OUT_DIR, "sec5_figs")
TAB_DIR <- file.path(OUT_DIR, "sec5_tables")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TAB_DIR, showWarnings = FALSE, recursive = TRUE)

# ---------- 2) Quick checks ----------
stopifnot(exists("df_m"))
stopifnot(is.data.frame(df_m))

# Keep factor levels stable (important for predict() and CV splits)
if ("release_month_f" %in% names(df_m)) df_m$release_month_f <- factor(df_m$release_month_f)
if ("primary_genre_top" %in% names(df_m)) df_m$primary_genre_top <- factor(df_m$primary_genre_top)

# ---------- 3) Fit models ----------
# m3 = full linear model
m3 <- lm(
  log_rev ~ log_budget_use + budget_missing + log_pop + runtime +
    log_cast + log_crew + release_year_c + release_month_f + primary_genre_top,
  data = df_m
)

# add quadratic year (replace linear year with polynomial basis)
m3_y2_clean <- update(m3, . ~ . - release_year_c + poly(release_year_c, 2, raw = TRUE))

# add quadratic runtime (replace linear runtime with polynomial basis)
m3_run2_clean <- update(m3_y2_clean, . ~ . - runtime + poly(runtime, 2, raw = TRUE))

# spline alternative (more flexible; used as a comparison baseline)
m3_spl2 <- update(
  m3,
  . ~ . -
    log_budget_use - log_pop - log_cast - log_crew +
    ns(log_budget_use, df = 4) +
    ns(log_pop, df = 4) +
    ns(log_cast, df = 3) +
    ns(log_crew, df = 3)
)

# ---------- 4) HC3 table: estimate, robust SE, t, p, CI ----------
robust_hc3_table <- function(model, level = 0.95) {
  V <- vcovHC(model, type = "HC3")
  b <- coef(model)
  se <- sqrt(diag(V))
  tval <- b / se
  
  # residual df for CI + p-values
  df_res <- df.residual(model)
  alpha <- 1 - level
  crit <- qt(1 - alpha/2, df_res)
  
  data.frame(
    term = names(b),
    estimate = as.numeric(b),
    robust_se = as.numeric(se),
    t_value = as.numeric(tval),
    p_value = 2 * pt(abs(tval), df = df_res, lower.tail = FALSE),
    ci_low  = as.numeric(b - crit * se),
    ci_high = as.numeric(b + crit * se),
    row.names = NULL
  )
}

final_hc3 <- robust_hc3_table(m3_run2_clean)
write.csv(final_hc3, file.path(TAB_DIR, "final_model_HC3_table.csv"), row.names = FALSE)

# ---------- 5) Sensitivity: top-10 Cook’s distance ----------
cd <- cooks.distance(m3_run2_clean)
ord <- order(cd, decreasing = TRUE)
top10 <- ord[1:10]

top10_df <- data.frame(index = top10, cooks_distance = cd[top10])
write.csv(top10_df, file.path(TAB_DIR, "top10_cooks_distance.csv"), row.names = FALSE)

m_sens <- update(m3_run2_clean, data = df_m[-top10, ])
sens_hc3 <- robust_hc3_table(m_sens)
write.csv(sens_hc3, file.path(TAB_DIR, "sensitivity_refit_HC3_table.csv"), row.names = FALSE)

# ---------- 6) Simple LaTeX table writer ----------
write_latex_table <- function(df, file, caption, label, digits = 3) {
  fmt_num <- function(x) sprintf(paste0("%.", digits, "f"), x)
  fmt_p <- function(p) ifelse(p < 1e-4, sprintf("%.2e", p), sprintf("%.4f", p))
  
  lines <- c(
    "\\begin{table}[H]",
    "\\centering",
    "\\scriptsize",
    paste0("\\caption{", caption, "}"),
    paste0("\\label{", label, "}"),
    "\\resizebox{\\textwidth}{!}{%",
    "\\begin{tabular}{lrrrrrr}",
    "\\toprule",
    "Term & Estimate & Robust SE & $t$ & $p$ & CI Low & CI High \\\\",
    "\\midrule"
  )
  
  for (i in seq_len(nrow(df))) {
    term <- df$term[i]
    row <- paste0(
      "\\vtt{", term, "} & ",
      fmt_num(df$estimate[i]), " & ",
      fmt_num(df$robust_se[i]), " & ",
      fmt_num(df$t_value[i]), " & ",
      fmt_p(df$p_value[i]), " & ",
      fmt_num(df$ci_low[i]), " & ",
      fmt_num(df$ci_high[i]), " \\\\"
    )
    lines <- c(lines, row)
  }
  
  lines <- c(
    lines,
    "\\bottomrule",
    "\\end{tabular}}",
    "\\end{table}"
  )
  
  writeLines(lines, con = file)
}

write_latex_table(
  final_hc3,
  file.path(TAB_DIR, "final_model_HC3_table.tex"),
  caption = "Final selected model (\\vtt{m3\\_run2\\_clean}) with HC3 robust SE and 95\\% CI.",
  label = "tab:final_hc3"
)

# ---------- 7) Nested ANOVA for quadratics ----------
anova_quad <- anova(m3, m3_y2_clean, m3_run2_clean)
write.csv(as.data.frame(anova_quad), file.path(TAB_DIR, "anova_m3_quadratics.csv"), row.names = TRUE)

# ---------- 8) Model selection stats ----------
aic_df <- AIC(m3, m3_run2_clean, m3_spl2)
bic_df <- BIC(m3, m3_run2_clean, m3_spl2)

adjr2_df <- data.frame(
  model = c("m3", "m3_run2_clean", "m3_spl2"),
  adj_r2 = c(summary(m3)$adj.r.squared,
             summary(m3_run2_clean)$adj.r.squared,
             summary(m3_spl2)$adj.r.squared)
)

write.csv(aic_df,  file.path(TAB_DIR, "aic_models.csv"), row.names = TRUE)
write.csv(bic_df,  file.path(TAB_DIR, "bic_models.csv"), row.names = TRUE)
write.csv(adjr2_df, file.path(TAB_DIR, "adjr2_models.csv"), row.names = FALSE)

# ---------- 9) BP test for heteroskedasticity (final model) ----------
bp <- bptest(m3_run2_clean)
capture.output(bp, file = file.path(TAB_DIR, "bptest_final_model.txt"))

# ---------- 10) 10-fold CV (RMSE + MAE) ----------
cv_lm <- function(formula, data, k = 10, seed = 123) {
  set.seed(seed)
  n <- nrow(data)
  folds <- sample(rep(1:k, length.out = n))
  
  rmse <- numeric(k)
  mae  <- numeric(k)
  
  for (j in 1:k) {
    test_idx <- which(folds == j)
    train <- data[-test_idx, , drop = FALSE]
    test  <- data[test_idx,  , drop = FALSE]
    
    fit <- lm(formula, data = train)
    y_test <- model.response(model.frame(formula, data = test))
    y_hat  <- predict(fit, newdata = test)
    
    rmse[j] <- sqrt(mean((y_test - y_hat)^2, na.rm = TRUE))
    mae[j]  <- mean(abs(y_test - y_hat), na.rm = TRUE)
  }
  
  c(RMSE_mean = mean(rmse), RMSE_sd = sd(rmse),
    MAE_mean  = mean(mae),  MAE_sd  = sd(mae))
}

cv_m3   <- cv_lm(formula(m3),           data = df_m, k = 10, seed = 123)
cv_run2 <- cv_lm(formula(m3_run2_clean), data = df_m, k = 10, seed = 123)
cv_spl2 <- cv_lm(formula(m3_spl2),       data = df_m, k = 10, seed = 123)

cv_metrics <- rbind(
  data.frame(model = "m3",            t(cv_m3)),
  data.frame(model = "m3_run2_clean", t(cv_run2)),
  data.frame(model = "m3_spl2",       t(cv_spl2))
)
write.csv(cv_metrics, file.path(TAB_DIR, "cv_metrics_models.csv"), row.names = FALSE)

# ---------- 11) Plots for Section 5 ----------
# Big-4 diagnostics for final model
png(file.path(FIG_DIR, "diagnostics_final_model_big4.png"), width = 1400, height = 900, res = 150)
par(mfrow = c(2,2)); plot(m3_run2_clean); par(mfrow = c(1,1))
dev.off()

# Observed vs Predicted + Residuals vs Fitted (log scale)
df_plot <- data.frame(
  y = df_m$log_rev,
  yhat = fitted(m3_run2_clean),
  resid = resid(m3_run2_clean)
)

p1 <- ggplot(df_plot, aes(x = yhat, y = y)) +
  geom_point(alpha = 0.35) +
  geom_abline(intercept = 0, slope = 1, linewidth = 1.2) +
  labs(
    title = "Final model: Observed vs Predicted (log1p revenue)",
    x = "Predicted log(1+revenue)",
    y = "Observed log(1+revenue)"
  )
ggsave(file.path(FIG_DIR, "final_obs_vs_pred.png"), p1, width = 9, height = 6, dpi = 200)

p2 <- ggplot(df_plot, aes(x = yhat, y = resid)) +
  geom_point(alpha = 0.35) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 1) +
  geom_smooth(se = FALSE, linewidth = 1.2) +
  labs(
    title = "Final model: Residuals vs Fitted",
    x = "Fitted values",
    y = "Residuals"
  )
ggsave(file.path(FIG_DIR, "final_residuals_vs_fitted.png"), p2, width = 9, height = 6, dpi = 200)

# ---------- 12) Print paths (handy for LaTeX write-up) ----------
cat("\nSaved outputs to:\n")
cat("Figures:", FIG_DIR, "\n")
cat("Tables :", TAB_DIR, "\n")

