import random
from collections import namedtuple
from configparser import ConfigParser
from datetime import datetime

import click
import numpy as np
import psycopg2
import requests
import scipy.io
from PIL import Image


def config(filename="database.ini", section="postgresql"):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(
            "Section {0} not found in the {1} file".format(section, filename)
        )

    return db

def storeInitial(start, end, channels_first):
    """Connect to the PostgreSQL database server"""
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        print("PostgreSQL database version:")
        cur.execute("SELECT version()")

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)

        #check if table exists and create if necessary
        query = """
            SELECT EXISTS (
                SELECT FROM pg_tables
                WHERE tablename  = 'dogs'
            );
        """
        cur.execute(query)
        if not cur.fetchone()[0]:
            print("creating table")
            query = """
                CREATE TABLE dogs(
                    id bigint PRIMARY KEY,
                    ts timestamp without time zone,
                    input real[],
                    prediction text,
                    label text
                )
            """
            cur.execute(query)

        for i,row in enumerate(generateRows(start, end)):
            print("sample " + str(i))
            data = (np.asarray(row.image)/255).flatten().tolist()
            print("generating prediction")
            prediction = sendPredictionRequest(data, channels_first)
            print("storing in DB")
            
            insertStatement = """ insert into dogs(ts, id, input, prediction, label)
                                  values(%s, %s, %s, %s, %s); """
            cur.execute(insertStatement, (datetime.now(), i, input, prediction, row.label))
            print()
        # close the communication with the PostgreSQL
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")

def storeNoisy(start, end, channels_first):
    """Connect to the PostgreSQL database server"""
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        print("PostgreSQL database version:")
        cur.execute("SELECT version()")

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)

        cur.execute("delete from dogs where id>999")

        for i, row in enumerate(generateRows(start,end)):
            print("sample " + str(i))
            dark = row.image.point(lambda i: i * 0.5)
            data = (np.asarray(dark)/255).flatten().tolist()
            print("generating prediction")
            prediction = sendPredictionRequest(data, channels_first)
            print("storing in DB")
            insertStatement = """ insert into dogs(ts, id, input, prediction, label)
                                  values(%s, %s, %s, %s, %s); """
            cur.execute(insertStatement, (datetime.now(), i+1000, input, prediction, row.label))
            print()
        # close the communication with the PostgreSQL
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")

def generateRows(start, end):
    mat = scipy.io.loadmat('dataset/test_list.mat')
    file_list = [str(path[0][0]) for path in mat['file_list']]
    random.Random(42).shuffle(file_list)
    for path in file_list[start : end]:
        groundTruth = path.split('/')[0].split('-',1)[1].replace('_', ' ').lower()
        image = Image.open('dataset/Images/' + path)
        image = image.resize((224,224))
        row = namedtuple('Row', ['image', 'label'])
        yield row(image, groundTruth)

imageNetLabels = []
with open("ImageNetLabels.txt") as file:
    for line in file:
        imageNetLabels.append(line.rstrip()) 

def sendPredictionRequest(image, channels_first):
    if channels_first:
        shape = [1,3,224,224]
    else:
        shape = [1,224,224,3]

    body = {
        "id": "1",
        "inputs": [
            {
                "name": "input_1",
                "shape": shape,
                "datatype": "FP32",
                "data": image
            }
        ],
        "outputs": [{"name": "output_1"}]
    }
    response = requests.post(
            "http://dogs-predictor-default.evaluations.panoptes.uk/v2/models/efficientnet/infer",
            json = body,
            headers = {"accept-encoding": None}
        )
    logits = response.json()["outputs"][0]["data"]
    prediction = imageNetLabels[logits.index(max(logits)) + 1].lower()
    return prediction

@click.option('-c','--channels-first', is_flag=True)
@click.command()
def initial(channels_first):
    storeInitial(0, 100, channels_first)

@click.option('-c','--channels-first', is_flag=True)
@click.command()
def noisy(channels_first):
    storeNoisy(0, 100, channels_first)

@click.group()
def main():
    pass

main.add_command(initial)
main.add_command(noisy)

if __name__=='__main__':
    main()