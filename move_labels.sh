# SUMMARY:
# 2400 train images (#1-2400) per class
# 601 test images (#2401-3001) per class; EXCEPT 600 for del (no 3001th image)


# initially 3000 training images per class, 29 classes total
# initially 1 test image per class

# want to move 600 images (20% of images) from training to test
# so move images #2401 - 3000 for each class

###############################################################################
# FIRST: rename all test images and zone identifiers to have index 3001, and
# move to class folder.
# "A_test.jpg" -> "A/A3001.jpg"
# "A_test.jpg:Zone.Identifier" -> "A/A3001.jpg:ZoneIdentifier"

# I renamed it manually lol this code doesn't work
# cd "Kaggle/asl_alphabet_test/asl_alphabet_test"
# pwd

# for file in *; do
# 	echo "$file"
# 	# rename "_test.jpg" "3001.jpg" ?_test.jpg
# 	# rename "_test.jpg:Zone.Identifier" "3001.jpg:Zone.Identifier" ?_test.jpg:Zone.Identifier
# done

######################## To move folders

# cd "Kaggle/asl_alphabet_test"
# for folder in *; do
# 	if [ -d $folder ]; then
# 		from_jpg="asl_alphabet_test/${folder}3001.jpg"
# 		from_zone="asl_alphabet_test/${folder}3001.jpg:Zone.Identifier"
# 		to_jpg="$folder/${folder}3001.jpg"
# 		to_zone="$folder/${folder}3001.jpg:Zone.Identifier"
# 		mv $from_jpg $to_jpg
# 		mv $from_zone $to_zone
# 	fi
# done

###############################################################################
# SECOND: move 600 images (#2401 - 3000) from train to test, per class
# Change file structure for test to match train (e.g. each class has set of files)

# data_dir="Kaggle"
# train_dir="asl_alphabet_train"
# test_dir="asl_alphabet_test"

# cd $data_dir
# cd $train_dir
# # Move each file from train to test dir
# echo "Moving files..."

# for folder in *; do
# 	if [ -d $folder ]; then
# 		cd $folder;
# 		for i in $(seq 2401 3000); do
# 			jpg_name="$folder$i.jpg"
# 			zone_name="$folder$i.jpg:Zone.Identifier"
# 			# echo "$jpg_name"
# 			# echo "$zone_name"
# 			mv "$jpg_name" "../../$test_dir/$folder/$file/$jpg_name"
# 			mv "$zone_name" "../../$test_dir/$folder/$file/$zone_name"
# 		done
# 		cd ..
# 	fi
# done
# echo "Moving of files completed"

###############################################################################
# THIRD: update .csv file for testing and training by just regenerating it

# cd "Kaggle/asl_alphabet_train"
# index=0
# # iterate each class
# for folder in *; do
# 	if [ -d $folder ]; then
# 		echo $folder;
# 		for i in $(seq 1 2400); do
# 			output="$folder/$folder$i.jpg, $index"
# 			echo $output >> new_training_labels.csv;
# 		done
# 		index=$((index+1))
# 	fi
# done

# cd "Kaggle/asl_alphabet_test"
# index=0
# # iterate each class
# for folder in *; do
# 	if [ -d $folder ]; then
# 		echo $folder;
# 		for i in $(seq 2401 3001); do
# 			output="$folder/$folder$i.jpg, $index"
# 			echo $output >> new_test_labels.csv;
# 		done
# 		index=$((index+1))
# 	fi
# done

# MANUALLY REMOVE THE ENTRY FOR DELETE 3001!!!!!!!!



