This directory contains all scripts relevant to the project.

# Visualizing the experiments
## Using Docker
1. Install [docker](https://docs.docker.com/get-docker/) and [docker-compose](https://docs.docker.com/compose/install/).
2. `cd omniboard`
3. `docker-compose build`
4. `docker-compose up -d`
5. If you add runs for a new experiment, you need to restart Omniboard and the config generator for them to show up: `docker-compose restart config_generator omniboard`
6. To stop the services from running type `docker-compose down`.

The `docker-compose.yml` also contains an instance of [MongoExpress](https://github.com/mongo-express/mongo-express) that can be reached by opening http://localhost:8081/. This is useful for debugging problems with MongoDB.
It also exposes the MongoDB instance itself on port 27017, so CLI tools can also be used from the host.

## Without using docker
1. Install [MongoDB](https://docs.mongodb.com/manual/installation/) and [OmniBoard](https://vivekratnavel.github.io/omniboard/#/quick-start). 
2. Create and empty directory `mkdir -p ~/mongo/data/db`.
3. Start a mongodb instance locally `sudo mongod --dbpath ~/mongo/data/db`.
4. Run the config generation script `python scripts/generate_mongo_config_file.py`. This will generate a file called `db_config.json`. You can also specify a different name by using the `--out_file` option of the script.
5. Start the Omniboard session `OMNIBOARD_CONFIG=db_config.json omniboard`. The environment variable `OMNIBOARD_CONFIG` should point to the config file generated in the previous step.
6. As with the docker installation you need to re-run the config generator (step 5) and restart omniboard (step 6) if you add a new experiment. 

Finally, you can open http://localhost:9000/ in your browser to access omniboard.
