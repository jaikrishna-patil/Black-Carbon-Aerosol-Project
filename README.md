# Black-Carbon-Aerosol-Project
## Results for Evaluation on Mean absolute error and trained using Mean squared error:
Type of split for evaluation | Best results yet(Mean absolute error)
------------- | -------------
Random split | Mean absolute error on test set [q_abs, q_sca, g]:-   [0.00238034 0.00192617 0.00298648]
Fractal_dimension=2.1, 2.2 left out for evaluation  | Mean absolute error on test set [q_abs, q_sca, g]:-   [0.01714607 0.0415282  0.02505202]
Fraction of coating=40, 50 left out for evaluation  | Mean absolute error on test set [q_abs, q_sca, g]:-   [0.00939356 0.00609494 0.01331163]

## Results for inverse problem:(Trained using MSE and evaulated using mean absolute error)
### Standardized output values too

Type of split for evaluation | Best results yet(Mean absolute error)- Using wavelength, vol_equi_radius_outer, primary_particle_size, q_ext,q_abs,q_sca,g as input
------------- | -------------
Random split | Mean absolute error on test set [fractal_dimension, fraction_of_coating]:-   [0.02767658 0.4789153 ]

## Installing the required packages
You can install all required packages to run the experiments using [conda](https://docs.conda.io/en/latest/). The following command should be enough: `conda env create -f conda_env.yml`. If you want GPU support, use `conda_env_gpu.yml`instead.

## Visualizing the experiments
We use [sacred](https://sacred.readthedocs.io/en/stable/) and [omniboard](https://github.com/vivekratnavel/omniboard) to keep track of experiments. Both can be run either in a local installation or using docker.

### Using Docker
The follwoing instructions should work on all platforms that support docker.

1. Install [docker](https://docs.docker.com/get-docker/) and [docker-compose](https://docs.docker.com/compose/install/).
2. `cd SacredTemplate/omniboard`
3. `docker-compose build`
4. `docker-compose up -d`
5. If you add runs for a new experiment, you need to restart Omniboard and the config generator for them to show up: `docker-compose restart config_generator omniboard`
6. To stop the services from running type `docker-compose down`.

The `docker-compose.yml` also contains an instance of [MongoExpress](https://github.com/mongo-express/mongo-express) that can be reached by opening http://localhost:8081/. This is useful for debugging problems with MongoDB.
It also exposes the MongoDB instance itself on port 27017, so CLI tools can also be used from the host.

### Without using docker
The following instructions should work on all linux distributions. Windows should be similar, but it has not been tested yet.

1. Install [MongoDB](https://docs.mongodb.com/manual/installation/) and [OmniBoard](https://vivekratnavel.github.io/omniboard/#/quick-start). 
2. Create and empty directory `mkdir -p ~/mongo/data/db`.
3. Start a mongodb instance locally `sudo mongod --dbpath ~/mongo/data/db`.
4. Run the config generation script `python SacredTemplate/scripts/generate_mongo_config_file.py`. This will generate a file called `db_config.json`. You can also specify a different name by using the `--out_file` option of the script.
5. Start the Omniboard session `OMNIBOARD_CONFIG=db_config.json omniboard`. The environment variable `OMNIBOARD_CONFIG` should point to the config file generated in the previous step.
6. As with the docker installation you need to re-run the config generator (step 4) and restart omniboard (step 5) if you add a new experiment. 

Finally, you can open http://localhost:9000/ in your browser to access omniboard.

## Importing a MongoDB dump
Sacred and omniboard use MongoDB as their storage backend, which means that experiment results can be shared as a MongoDB dump. The commands to import the dump depend on how omniboard/MongoDB was installed:

### Using docker
1. Start the docker containers: `docker-compose up`
2. Import the dump `docker exec -i omniboard_mongo_1 /usr/bin/mongorestore -u root -p example --archive --gzip < dump.gz`, where `dump.gz` is the name of the dump file.
3. To see the changes, restart the config generator and omniboard: `docker-compose restart config_generator omniboard`

### Without using docker
1. Start the MongoDB instance: `sudo mongod --dbpath ~/mongo/data/db`
2. Import the dump `mongorestore --archive --gzip < dump.gz`, where `dump.gz` is the name of the dump file.
3. Re-run the config generator and restart omniboard (steps 4 and 6 in the installation instructions above).
