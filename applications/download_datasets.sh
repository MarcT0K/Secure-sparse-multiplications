sudo apt install wget unzip

# Spam detection
wget https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
unzip sms+spam+collection.zip
rm readme sms+spam+collection.zip
mv SMSSpamCollection spam.csv

# Book recommendation application
wget http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip
unzip BX-CSV-Dump.zip
rm BX-Books.csv BX-Users.csv BX-CSV-Dump.zip

# Access log application
wget https://raw.githubusercontent.com/pyduan/amazonaccess/master/data/train.csv
mv train.csv amazon.csv

