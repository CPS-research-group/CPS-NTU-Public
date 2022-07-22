#!/usr/bin/bash

echo "Downloading raw data from Dr. NTU..."
curl -o raw.tgz https://researchdata.ntu.edu.sg/file.xhtml?fileId=69950&version=1.0

echo "Uncompressing raw data..."
tar -xfp raw.gz

echo "Creating training set for BVAE..."
mkdir train_bvae
mkdir train_bvae/rain
mkdir train_bvae/brightness
cp raw/scene?_empty0/*.png train_bvae/rain/
cp raw/scene?_empty1/*.png train_bvae/rain/
cp raw/scene?_empty0/*.png train_bvae/brightness/
cp raw/scene?_empty1/*.png train_bvae/brightness/
python prep_vids.py --input train_bvae/rain/ \
	            --rain_level 0.003 \
		    --rain_speed 480 \
		    --min_level 0.0 \
		    --mode 4
python prep_vids.py --input train_bvae/brightness/ \
	            --brightness_level 0.5 \
		    --min_level -0.5 \
		    --mode 4

echo "Creating calibration set for BVAE..."
mkdir calibration_bvae
mkdir calibration_bvae/rain
mkdir calibration_bvae/brightness
for i in {1..5};
do
  python prep_vids.py --input raw/scene${i}_empty2 \
	              --folder2vid
  python prep_vids.py --input scene${i}_empty2.avi \
	              --rain_level 0.003 \
		      --rain_speed 480 \
		      --min_level 0.0 \ 
		      --mode 5
  rm scene${i}_empty2.avi
done
mv *.avi calibration_bvae/rain
for i in {1..5};
do
  python prep_vids.py --input raw/scene${i}_empty2 \
	              --folder2vid
  python prep_vids.py --input scene${i}_empty2.avi \
	              --brightness_level 0.5 \
		      --min_level -0.5 \
		      --mode 5
  rm scene${i}_empty2.avi
done
mv *.avi calibration_bvae/brightness

echo "Creating test set for BVAE..."
mkdir test_bvae
for i in {1..5};
do
  python prep_vids.py --input raw/scene${i}_empty3 \
	              --folder2vid
  for j in {1..3};
  do
    python prep_vids.py --input scene${i}_empty3.avi \
	                --rain_level 0.01 \
                        --rain_speed 480 \
		        --min_level 0.004 \
			--mode ${j}
    python prep_vids.py --input scene${i}_empty3.avi \
	                --brightness_level -0.5 \
			--min_level -1
                        --mode ${j}
  done
done
mv *.avi test_bvae/

echo "Creating training set for OF..."
mkdir train_of
mkdir train_bvae/id
cp raw/scene?_empty0/*.png train_bvae/id/
cp raw/scene?_empty1/*.png train_bvae/id/
cp raw/scene?_empty2/*.png train_bvae/id/


echo "Creating test set for OF..."
mkdir test_of
for i in {1..5};
do
  python prep_vids.py --input raw_scene${i}_empty3 \
	              --folder2vid
  for j in {1..3};
  do
    python prep_vids.py --input scene${i}_empty3.avi \
	                --rain_level 0.003 \
                        --mode ${j}
    python prep_vids.py --input scene${i}_empty3.avi \
	                --snow_level 0.003 \
			--mode ${j}
  done
done
mv *.avi test_of/
