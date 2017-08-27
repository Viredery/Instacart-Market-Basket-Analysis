class ParamConfig:
    def __init__(self):
        self.root_dir = '../..'

        self.raw_data_folder = '{}/data/raw'.format(self.root_dir)
        self.raw_aisles_path = '{}/aisles.csv'.format(self.raw_data_folder)
        self.raw_departments_path = '{}/departments.csv'.format(self.raw_data_folder)
        self.raw_order_products_prior_path = '{}/order_products__prior.csv'.format(self.raw_data_folder)
        self.raw_order_products_train_path = '{}/order_products__train.csv'.format(self.raw_data_folder)
        self.raw_orders_path = '{}/orders.csv'.format(self.raw_data_folder)
        self.raw_products_path = '{}/products.csv'.format(self.raw_data_folder)

        self.feat_folder = '{}/feat'.format(self.root_dir)

        self.data_folder = '{}/data/processed'.format(self.root_dir)
        self.products_path = '{}/products.pkl'.format(self.data_folder)
        self.orders_path = '{}/orders.pkl'.format(self.data_folder)
        self.order_products_prior_path = '{}/order_products_prior.pkl'.format(self.data_folder)
        self.order_products_train_path = '{}/order_products_train.pkl'.format(self.data_folder)

        self.random_seed = 2017

config = ParamConfig()
