bands_combination = {
    'RGB':['LE7 B3 (red)', 'LE7 B2 (green)','LE7 B1 (blue)'],
    '6-8' : ['LE7 B6_VCID_2 (high-gain thermal infrared)','LE7 B6_VCID_1 (low-gain thermal infrared)','LE7 B7 (shortwave infrared 2)'],
    '5-4-2' : ['LE7 B5 (shortwave infrared 1)','LE7 B4 (near infrared)','LE7 B2 (green)'],
    '11-12-13' : ['NDVI (vegetation index)', 'NDSI (snow index)', 'NDWI (water index)'],
    # 'LE7' : ['LE7 B1 (blue)', 'LE7 B2 (green)',
    #              'LE7 B3 (red)', 'LE7 B4 (near infrared)', 'LE7 B5 (shortwave infrared 1)',
    #              'LE7 B6_VCID_1 (low-gain thermal infrared)', 'LE7 B6_VCID_2 (high-gain thermal infrared)', 
    #              'LE7 B7 (shortwave infrared 2)', 'LE7 B8 (panchromatic)', 'LE7 BQA (quality bitmask)'],
    #  'All' : ['LE7 B1 (blue)', 'LE7 B2 (green)',
    #              'LE7 B3 (red)', 'LE7 B4 (near infrared)', 'LE7 B5 (shortwave infrared 1)',
    #              'LE7 B6_VCID_1 (low-gain thermal infrared)', 'LE7 B6_VCID_2 (high-gain thermal infrared)', 
    #              'LE7 B7 (shortwave infrared 2)', 'LE7 B8 (panchromatic)', 'LE7 BQA (quality bitmask)',
    #              'NDVI (vegetation index)', 'NDSI (snow index)', 'NDWI (water index)',
    #              'SRTM 90 elevation', 'SRTM 90 slope']
}

feature_names = ['LE7 B1 (blue)', 'LE7 B2 (green)',
                 'LE7 B3 (red)', 'LE7 B4 (near infrared)', 'LE7 B5 (shortwave infrared 1)',
                 'LE7 B6_VCID_1 (low-gain thermal infrared)', 'LE7 B6_VCID_2 (high-gain thermal infrared)', 
                 'LE7 B7 (shortwave infrared 2)', 'LE7 B8 (panchromatic)', 'LE7 BQA (quality bitmask)',
                 'NDVI (vegetation index)', 'NDSI (snow index)', 'NDWI (water index)',
                 'SRTM 90 elevation', 'SRTM 90 slope']


channels_stats = {
    "means": [151.98861694335938, 141.67233276367188, 147.73736572265625, 126.35386657714844, 41.389190673828125, 86.37078094482422, 54.272972106933594, 34.75004959106445, 129.8513641357422, 1297.374755859375, -0.09093914926052094, -0.534999668598175, 0.4840894341468811, 5337.916015625, 26.90826988220215], 
    "stds": [85.87097930908203, 89.88019561767578, 88.11711883544922, 79.6325454711914, 33.70036697387695, 21.227155685424805, 36.88022232055664, 30.684616088867188, 86.3227310180664, 466.8656311035156, 0.11548927426338196, 0.35953786969184875, 0.32466045022010803, 826.7803955078125, 14.102063179016113],
    'mins': [0,0,0,0,0,0,0,0,0,0,-1,-1,-1,0,0],
    'maxs': [256,256,256,256,256,256,256,256,256,3000,1,1,1,8848,90],
}