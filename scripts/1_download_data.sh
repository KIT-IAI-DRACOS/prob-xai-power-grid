cd ./External_data/ENTSO-E-20-23

# To access the sftp you need an account for the ENTSO-E Transparency platform 
sftp -v user.email@institute.com@sftp-transparency.entsoe.eu << EOF
cd TP_export

# Actual load 
cd ActualTotalLoad_6.1.A
get 202[0-4]*.csv

# Load forecast
cd ..
cd DayAheadTotalLoadForecast_6.1.B
get 202[0-4]*.csv

# Generation forecast
cd ..
cd DayAheadAggregatedGeneration_14.1.C
get 202[0-4]*.csv

# Generation per type
cd ..
cd AggregatedGenerationPerType_16.1.B_C
get 202[0-4]*.csv

# Wind/solar forecast
cd ..
cd DayAheadGenerationForecastForWindAndSolar_14.1.D
get 202[0-4]*.csv

# Day ahead prices
cd ..
cd DayAheadPrices_12.1.D
get 202[0-4]*.csv

EOF
