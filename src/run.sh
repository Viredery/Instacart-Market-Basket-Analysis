cd feature

python3 preprocess.py

python3 model_order.py # need change~
python3 model_product.py
python3 model_product_vector.py
python3 model_user.py
python3 model_aisle.py
python3 model_department.py

python3 model_user_aisle.py
python3 model_user_department.py

python3 model_user_product.py
python3 model_user_product_recent.py
python3 model_user_product_dependent.py

python3 construct_dataset.py
python3 construct_dataset_features.py

cd ..
cd model

python3 lightbgm_classifier.py #shuffle