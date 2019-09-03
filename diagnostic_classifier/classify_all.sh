echo "---" >> exp_classification.log
ls data/*_train_data.npy | sed "s/_train_data.npy//" | while read LANG;
do
python classifier.py $LANG
done
