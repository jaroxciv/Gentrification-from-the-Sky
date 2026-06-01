"""Y — Gentrification score (the target variable).

Builds the census-based ground truth from ONS / IMD data at LSOA level
(paper §3.1): loads the four socioeconomic measures (age, education, housing,
income) for 2011 and 2021, computes the neighborhood index, and derives the
gentrification score as its change over time.
"""
