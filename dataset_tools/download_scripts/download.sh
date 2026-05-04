BASE_URL="https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets"

# Loop from 2008 to 2017
for year in $(seq 2008 2017); do
    FILE="${year}_30m_cdls.zip"
    echo "Downloading $FILE..."
    wget "${BASE_URL}/${FILE}"
done