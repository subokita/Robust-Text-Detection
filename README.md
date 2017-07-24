### This is my upcoming project: https://plusplusone.herokuapp.com
### Please register your email address if you're interested in it.
------

Robust-Text-Detection
=====================

Robust Text Detection implementation based on 
Chen, Huizhong, et al. "Robust Text Detection in Natural Images with Edge-Enhanced Maximally Stable Extremal Regions." Image Processing (ICIP), 2011 18th IEEE International Conference on. IEEE, 2011.

which you can find the original literature here:  http://www.stanford.edu/~hchen2/papers/ICIP2011_RobustTextDetection.pdf
It is also partly based on Automatically Detect and Recognize Text in Natural Images example
which is available here: http://www.mathworks.de/de/help/vision/examples/automatically-detect-and-recognize-text-in-natural-images.html#zmw57dd0e829

This implementation is partly motivated by the fact that helperGrowEdges and helperStrokeWidth functions on Matlab which aren't openly available to the public or non latest Matlab owner, thus those 2 functions are implemented from the scratch based on the literature and some of my own assumptions (e.g. how many pixels to prune, etc)

Feel free to correct my code, if you spotted the mistakes
