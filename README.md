# Description
Download real_name_mi.tar.gz file from https://fh295.github.io/frozen.html and put them inside src/image_classification
Thus, you should have a directory src/image_classification/real_name_mi that will contain the data

1.fromage 2 baseline unconstrained  
python -u real_mi.py --num_ways 2 --baseline

2.fromage 2 content-free unconstrained      		  
python -u real_mi.py --num_ways 2 

3.fromage 2 ordered unconstrained 
python -u real_mi.py --num_ways 2 --order --baseline


4.fromage 2 content-free constrained
python -u real_mi.py --num_words 7 --num_ways 2 --constraint

5.fromage 2 baseline constrained
python -u real_mi.py --num_ways 2 --constraint --baseline

6.fromage 2 ordered constrained 
python -u real_mi.py --num_ways 2 --constraint --order --baseline				


7.fromage 5 content-free constrained 
python -u real_mi.py --num_words 7 --num_ways 5 --constraint 

8.fromage 5 ordered constrained   	
python -u real_mi.py --num_words 7 --num_ways 5 --constraint --order --baseline 
			
9.fromage 5 baseline constrained 
python -u real_mi.py --num_words 7 --num_ways 5 --constraint --baseline  
