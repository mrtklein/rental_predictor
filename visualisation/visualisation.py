import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

data2 = pd.read_csv(os.path.join(".", "cologneData.csv"))

#print uniqe values and their count
data2.interiorQual.value_counts(dropna=False)

for col in data2.columns:
    x = data2[col]
    y = x.dropna()
    print("Column " + col + " containes " + (str(x.size - y.size) + " NaN Values.") )

# Plot
fig, ax = plt.subplots()
plt.title('Rent vs. Living Space (Red = has Kitchen)')
plt.ylabel('Base Rent')
plt.xlabel('Living Space')
color= ['red' if l else 'blue' for l in data2.hasKitchen]
plt.scatter( data2.livingSpace, data2.baseRent, alpha=0.5, color = color)
plt.savefig("./visuals/Rent vs. Living Space (Red = has Kitchen).png")
plt.show()

# Plot
plt.scatter(data2.picturecount, data2.baseRent, alpha=0.5)
plt.title('Rent vs. Number of Pictures Posted ')
plt.ylabel('Base Rent')
plt.xlabel('Picture count')
plt.savefig("./visuals/Rent vs. Number of Pictures Posted.png")
plt.show()


# Plot
plt.scatter(data2.yearConstructed, data2.baseRent, alpha=0.5)
plt.title('Rent vs. Year of building construction')
plt.ylabel('Base Rent')
plt.xlabel('Year Constructed')
plt.savefig("./visuals/Rent vs. Year of building construction.png")
plt.show()

#bargraphs
pets = data2.petsAllowed.value_counts(dropna=False)
plt.bar(np.arange(pets.size), pets.values)
plt.xticks(np.arange(pets.size), pets.index)
plt.title('Pets allowed')
plt.savefig("./visuals/BarPetsAllowed.png")
plt.show()

balcony = data2.balcony.value_counts(dropna=False)
plt.bar(np.arange(balcony.size), balcony.values)
plt.xticks(np.arange(balcony.size), balcony.index)
plt.title('Has Balcony')
plt.savefig("./visuals/.png")
plt.show()

noRooms = data2.noRooms.value_counts(dropna=False)
plt.bar(np.arange(noRooms.size), noRooms.values)
plt.xticks(np.arange(noRooms.size), noRooms.index)
plt.title('Number of Rooms')
plt.savefig("./visuals/BArNumberOfRooms.png")
plt.show()

typeOfFlat = data2.typeOfFlat.value_counts(dropna=False)
plt.bar(np.arange(typeOfFlat.size), typeOfFlat.values)
plt.xticks(np.arange(typeOfFlat.size), typeOfFlat.index)
plt.title('Type of Flat')
plt.savefig("./visuals/BarTypeOfFlat.png")
plt.show()

heatingType = data2.heatingType.value_counts(dropna=False)
plt.bar(np.arange(heatingType.size), heatingType.values)
plt.xticks(np.arange(heatingType.size), heatingType.index, ha='right', rotation=45)
plt.title('Heating Type')
plt.savefig("./visuals/BarHeatingType.png")
plt.show()

data2.livingSpace.hist()
plt.title("Size of Living space in square Meters")
plt.savefig("./visuals/Livingspace.png")
plt.show()

plz_shape_df = gpd.read_file('./plz-gebiete.shp', dtype={'plz': str})

plzArr = data2.geo_plz.unique()
plzArr = np.char.mod('%d', plzArr) #convert ints to string

mdf = data2.groupby("geo_plz", as_index=False).mean()
mdf["geo_plz"] = mdf["geo_plz"].astype(str)

plz_shape_df = plz_shape_df.query('plz in @plzArr') #only consider shaped that are in cologne

merged = pd.merge(plz_shape_df, mdf, left_on="plz", right_on="geo_plz")
#add baseprice per sqm
merged["pricePerSqm"] = merged.baseRent / merged.livingSpace


fig, ax = plt.subplots()
merged.plot(
    ax=ax,
    column='baseRent',
    categorical=False,
    legend=True,
    cmap='autumn_r',
)
ax.set(
    title='Average total Base Rent',
    aspect=1.3,
    facecolor='lightblue'
);

plt.savefig("./visuals/MapRent.png")
plt.show()






fig, ax = plt.subplots()
merged.plot(
    ax=ax,
    column='pricePerSqm',
    categorical=False,
    legend=True,
    cmap='autumn_r',
)
ax.set(
    title='Average Base Rent per square meter',
    aspect=1.3,
    facecolor='lightblue'
);

plt.savefig("./visuals/MapRentPerSqm.png")
plt.show()

fig, ax = plt.subplots()
merged.plot(
    ax=ax,
    column='livingSpace',
    categorical=False,
    legend=True,
    cmap='autumn_r',
)
ax.set(
    title='Average living Space',
    aspect=1.3,
    facecolor='lightblue'
)
plt.savefig("./visuals/MapLivingSpace.png")
plt.show()

count_per_plz = data2.geo_plz.value_counts()
count_per_plz = pd.DataFrame({'geo_plz':count_per_plz.index, 'count':count_per_plz.values})
count_per_plz["geo_plz"] = count_per_plz["geo_plz"].astype(str)
merged = pd.merge(merged, count_per_plz, on="geo_plz")

fig, ax = plt.subplots()
merged.plot(
    ax=ax,
    column='count',
    categorical=False,
    legend=True,
    cmap='autumn_r',
)
ax.set(
    title='Number of Datapoints',
    aspect=1.3,
    facecolor='lightblue'
)
plt.savefig("./visuals/MapCountOfDatapoints.png")
plt.show()
