sudo apt install wget unzip

mkdir datasets
cd datasets

# Spam detection
wget https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
unzip sms+spam+collection.zip
rm readme sms+spam+collection.zip
mv SMSSpamCollection spam.csv

# Book recommendation application
wget https://web.archive.org/web/20230623001827/http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip
unzip BX-CSV-Dump.zip
rm BX-Books.csv BX-Users.csv BX-CSV-Dump.zip

# Access log application
wget https://raw.githubusercontent.com/pyduan/amazonaccess/master/data/train.csv
mv train.csv amazon.csv

# MovieLens
wget https://files.grouplens.org/datasets/movielens/ml-32m.zip
unzip ml-32m.zip
mv ml-32m/ratings.csv movielens.csv
rm -r ml-32m
rm ml-32m.zip

# DOROTHEA
wget https://archive.ics.uci.edu/static/public/169/dorothea.zip
unzip dorothea.zip
rm -r DOROTHEA
rm dorothea.zip dorothea_valid.labels Dataset.pdf
