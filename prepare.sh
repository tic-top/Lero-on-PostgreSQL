## tpcds

## tpch

#!/bin/bash

DB_NAME="tpch_db"
SCALE_FACTOR=1
PG_USER="postgres"
DATA_DIR="./tpch_data"
REPO_DIR="./tpch-dbgen"

if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning TPC-H generator..."
    git clone https://github.com/electrum/tpch-dbgen.git "$REPO_DIR"
    cd "$REPO_DIR"
    make -f makefile.suite
    cd ..
fi

mkdir -p "$DATA_DIR"
if [ ! -f "$DATA_DIR/customer.tbl" ]; then
    echo "Generating TPC-H data (SF=$SCALE_FACTOR)..."
    cd "$REPO_DIR"
    ./dbgen -vf -s "$SCALE_FACTOR"
    
    mv *.tbl "../$DATA_DIR" 2>/dev/null || echo "No .tbl files to move"

    if [ ! -f "$DATA_DIR/delete.1" ]; then
        echo "Generating refresh data..."
        ./dbgen -v -U "$SCALE_FACTOR"
        mv *.tbl.* "../$DATA_DIR" 2>/dev/null || echo "No refresh files to move"
    fi
    cd ..
fi

echo "Setting up PostgreSQL database..."
psql -U "$PG_USER" -h localhost -c "DROP DATABASE IF EXISTS $DB_NAME;"
psql -U "$PG_USER" -h localhost -c "CREATE DATABASE $DB_NAME;"

echo "Creating tables..."
psql -U "$PG_USER" -h localhost -d "$DB_NAME" -f "$REPO_DIR/dss.ddl"

echo "Loading data..."
for table in nation region part supplier partsupp customer orders lineitem; do
    if [ -f "$DATA_DIR/$table.tbl" ]; then
        echo "  Loading $table..."
        psql -U "$PG_USER" -h localhost -d "$DB_NAME" -c "\copy $table FROM '$DATA_DIR/$table.tbl' WITH DELIMITER '|' CSV;"
    else
        echo "  WARNING: $table.tbl not found in $DATA_DIR!"
    fi
done

echo "Adding constraints..."
psql -U "$PG_USER" -h localhost -d "$DB_NAME" -f "$REPO_DIR/dss.ri"

## stats
psql -d 'postgres' -c "DROP DATABASE IF EXISTS stats;"
psql -d 'postgres' -c "CREATE DATABASE stats;"
psql -d stats -f stats_db.sql

## imdb

# https://github.com/gregrahn/join-order-benchmark


# This script was intended to get raw IMDB datasets into Postgres
# The script worked on my mac setup.
#
# You may have to update your psql command with appropriate -U and -d flags and may have to 
# provide appropriate permissions to new folders.
#
# Customise as you see fit and for your setup.
#
# Tables are NOT optomised, nor have any indexes been created
# The point is to "just get the data into Postgres"
#
# Remember to allow execution rights before trying to run it:
# chmod 755 imdb_postgres_setup.sh
#
# Just for interest's sake, below are the terminal commands for my Linux box:
# su - postgres
# curl -O https://gist.githubusercontent.com/IllusiveMilkman/2a7a6614193c74804db7650f6d3c2bd2/raw/c8c0b4dbac00cf7539dd5cc9670fe00b38430f7d/imdb_postgres_setup.sh
# chmod 755 imdb_postgres_setup.sh
# ./imdb_postgres_setup.sh
#
# If you don't know the password for postgres (new install, some default VM setups, etc)
# sudo passwd postgres
# set a new password, then continue above
#

printf "Script starting at %s. \n" "$(date)"

printf "Removing old folders \n"
rm -rf imdb-datasets/

printf "Creating new folders \n"
mkdir imdb-datasets/

printf "Downloading datasets from https://datasets.imdbws.com \n"
cd imdb-datasets
curl -O https://datasets.imdbws.com/name.basics.tsv.gz
curl -O https://datasets.imdbws.com/title.akas.tsv.gz
curl -O https://datasets.imdbws.com/title.basics.tsv.gz
curl -O https://datasets.imdbws.com/title.crew.tsv.gz
curl -O https://datasets.imdbws.com/title.episode.tsv.gz
curl -O https://datasets.imdbws.com/title.principals.tsv.gz
curl -O https://datasets.imdbws.com/title.ratings.tsv.gz

printf "Unzipping datasets... \n"
gzip -dk *.gz
cd ..

printf "Creating Database \n"
psql -d 'postgres' -c "DROP DATABASE IF EXISTS imdb;"
psql -d 'postgres' -c "CREATE DATABASE imdb;"

printf "Creating tables in imdb database \n"
psql -d imdb -c "CREATE table title_ratings (tconst VARCHAR(10),average_rating NUMERIC,num_votes integer);"
psql -d imdb -c "CREATE TABLE name_basics (nconst varchar(10), primaryName text, birthYear smallint, deathYear smallint, primaryProfession text, knownForTitles text );"
psql -d imdb -c "CREATE TABLE title_akas (titleId TEXT, ordering INTEGER, title TEXT, region TEXT, language TEXT, types TEXT, attributes TEXT, isOriginalTitle BOOLEAN);"
psql -d imdb -c "CREATE TABLE title_basics (tconst TEXT, titleType TEXT, primaryTitle TEXT, originalTitle TEXT, isAdult BOOLEAN, startYear SMALLINT, endYear SMALLINT, runtimeMinutes INTEGER, genres TEXT);"
psql -d imdb -c "CREATE TABLE title_crew (tconst TEXT, directors TEXT, writers TEXT);"
psql -d imdb -c "CREATE TABLE title_episode (const TEXT, parentTconst TEXT, seasonNumber TEXT, episodeNumber TEXT);"
psql -d imdb -c "CREATE TABLE title_principals (tconst TEXT, ordering INTEGER, nconst TEXT, category TEXT, job TEXT, characters TEXT);"

printf "Inserting data into tables \n"
psql -d imdb -c "COPY title_ratings FROM '$(pwd)/imdb-datasets/title.ratings.tsv' DELIMITER E'\t' QUOTE E'\b' NULL '\N' CSV HEADER"
psql -d imdb -c "COPY name_basics FROM '$(pwd)/imdb-datasets/name.basics.tsv' DELIMITER E'\t' QUOTE E'\b' NULL '\N' CSV HEADER"
psql -d imdb -c "COPY title_akas FROM '$(pwd)/imdb-datasets/title.akas.tsv' DELIMITER E'\t' QUOTE E'\b' NULL '\N' CSV HEADER"
psql -d imdb -c "COPY title_basics FROM '$(pwd)/imdb-datasets/title.basics.tsv' DELIMITER E'\t' QUOTE E'\b' NULL '\N' CSV HEADER"
psql -d imdb -c "COPY title_crew FROM '$(pwd)/imdb-datasets/title.crew.tsv' DELIMITER E'\t' QUOTE E'\b' NULL '\N' CSV HEADER"
psql -d imdb -c "COPY title_episode FROM '$(pwd)/imdb-datasets/title.episode.tsv' DELIMITER E'\t' QUOTE E'\b' NULL '\N' CSV HEADER"
psql -d imdb -c "COPY title_principals FROM '$(pwd)/imdb-datasets/title.principals.tsv' DELIMITER E'\t' QUOTE E'\b' NULL '\N' CSV HEADER"

printf "Done! \n"

printf "Script done at %s. \n" "$(date)"
