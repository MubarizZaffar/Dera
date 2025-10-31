This is a finetuned BoQ [1] model trained on two different types of data sources: 

a) Google street view data from around the world which is usually considered GPS-accurate but does not contain much variations of the same place.
b) Mapillary data from around the world, which was verified by the author for accuracy using geometric verification with MAST3R and field-of-view overlap. This data is much more diverse than GSV.

Please feel free to use this model under CC BY license.

The file dera.py was used for the croco-dl benchmark. It is compatible with hloc, Lamar, and crocodl pipelines. You can ofcourse also choose to use the model independent of these pipelines.

[1] Amar Ali-Bey, BoQ: A Place is Worth a Bag of learnable Queries, CVPR, 2024. 
