# What is Dera?
This is a finetuned BoQ [1] model trained on two different types of data sources for visual place recognition. It is also the name of my hometown. :)

The two data sources are:

- Google street view data from around the world which is usually considered GPS-accurate but does not contain much variations of the same place.
- Mapillary data from around the world, which was verified by the author for accuracy using geometric verification with MAST3R and field-of-view overlap. This data is much more diverse than GSV.

Please feel free to use this model under CC BY license. I intend to continue updating the model checkpoint on HuggingFace over time.

# How to use this repo?
The file dera.py was used for the croco-dl benchmark. It is compatible with hloc, Lamar, and crocodl pipelines. You can for example copy this to /hloc/extractors/ in the hloc codebase [2]. 

Ofcourse, you can also choose to use the model independent of these pipelines.

[1] Amar Ali-Bey, BoQ: A Place is Worth a Bag of learnable Queries, CVPR, 2024. 

[2] https://github.com/cvg/Hierarchical-Localization/tree/master/hloc/extractors
