import pandas as pd
import os

data = pd.read_csv(os.path.join(".", "immo_data.csv"))

print(data.shape)



data2 = data[data.regio2 == "Köln"]
#filter out trash
#totalRent nicht sinnvoll da oft NaS
data2 = data2[data2.baseRent > 100]
data2 = data2[data2.baseRent < 10000]

#remove a weired datapoint manually
data2 = data2[data2.geo_plz != 76530]

#drop useless columns
data2 = data2.drop(["regio1", "regio2", "telekomTvOffer", "telekomHybridUploadSpeed", "houseNumber", "street", "streetPlain",
             "description", "electricityBasePrice", "electricityKwhPrice", "date", "scoutId", "geo_bln",
             "geo_krs", "pricetrend", "telekomUploadSpeed", "baseRentRange", "livingSpaceRange", "facilities", 'serviceCharge',
             'heatingType', 'totalRent', 'firingTypes', 'yearConstructedRange', 'thermalChar', 'noRoomsRange', 'facilities',
             'heatingCosts', 'energyEfficiencyClass', 'lastRefurbish'], axis=1)





#thermalChar removen? Sehr sparse, und gibt nicht so richtig sinn
#geokoordinaten: sind oft falsch, bzw sparse. Ausserdem schwer zu verarbeiten
#pricetrend removen? -> da steckt schon preis drinn...
# total preis sollte sinnvoll berechnet sein. z.b. total preis wenn nAn, ansonsten base preis + flex, wenn nur basepreis vllt +10% oder so

#save data into new csv
data2.to_csv("cologneData.csv")