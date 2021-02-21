import numpy as np
from osgeo import gdal
import os

def image_open(img_path):
    data = gdal.Open(img_path)

    Red = data.GetRasterBand(3).ReadAsArray().astype(np.float32)
    Blue = data.GetRasterBand(1).ReadAsArray().astype(np.float32)
    Green = data.GetRasterBand(2).ReadAsArray().astype(np.float32)
    Nir = data.GetRasterBand(4).ReadAsArray().astype(np.float32)

    NDVI = (Nir-Red) / (Nir+Red)
    NDVI[np.isnan(NDVI)] = 0  # 空值转0
    NDVI = NDVI.astype(np.float32)
    NDWI = (Green-Nir) / (Green+Nir)
    NDWI[np.isnan(NDWI)] = 0
    NDWI = NDWI.astype(np.float32)
    IRRG = Nir * Red / Green
    IRRG[np.isnan(IRRG)] = 0
    IRRG = IRRG.astype(np.float32)
    SAVI = (1+0.5)*(Nir-Red)/(Nir+Red+0.5)
    SAVI[np.isnan(SAVI)] = 0
    Grass = (Red+Green+Blue)/3.0 * SAVI
    Grass = Grass.astype(np.float32)

    Red = Red[:,:,np.newaxis]
    Blue = Blue[:,:,np.newaxis]
    Green = Green[:,:,np.newaxis]




    return np.concatenate([Red, Green, Blue], axis=2), Nir, NDVI, NDWI, IRRG, SAVI, Grass
