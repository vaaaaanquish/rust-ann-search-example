wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xvf food-101.tar.gz

mv food-101/images/sushi/* /app/img/
mv food-101/images/ramen/* /app/img/
mv food-101/images/gyoza/* /app/img/
mv food-101/images/pizza/* /app/img/
mv food-101/images/takoyaki/* /app/img/

rm -rf food-101
