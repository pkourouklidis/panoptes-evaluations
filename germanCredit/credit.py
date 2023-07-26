import copy
import random
from configparser import ConfigParser
from datetime import datetime

import click
import numpy as np
import pandas as pd
import psycopg2
from util import generateFeatures, sendInferenceRequest


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


def storeInitial():
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
                WHERE tablename  = 'credit'
            );
        """
        cur.execute(query)
        if not cur.fetchone()[0]:
            print("creating table")
            query = """
                CREATE TABLE credit (
                    id bigint PRIMARY KEY,
                    ts timestamp without time zone,
                    sex text,
                    prediction int,
                    label int
                )
            """
        else:
            print("clearing table")
            query = "delete from credit"
        cur.execute(query)

        #store data
        df_credit = pd.read_csv("./dataset/german_credit_data.csv",index_col=0)
        sample = df_credit.sample(n=100, random_state=102)
        X,y = generateFeatures(copy.deepcopy(sample))
        print("Generating predictions")
        predictions = [sendInferenceRequest(row[1].to_list()) for row in X.iterrows()]
        print("done")
        print(predictions)
        labels = y.to_list()

        print("storing in DB")
        for i, row,prediction,label in zip(range(100), sample.itertuples(), predictions, labels):
            insertStatement = """ insert into credit(ts, id, sex, prediction, label)
                                  values(%s, %s, %s, %s, %s); """
            cur.execute(insertStatement, (datetime.now(), i, row.Sex, prediction, label))

        # close the communication with the PostgreSQL
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")

def shiftSex():
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

        cur.execute("delete from credit where id>999")

        df_credit = pd.read_csv("./dataset/german_credit_data.csv",index_col=0)
        sample = df_credit.sample(n=100, random_state=102)
        sample.Sex = [random.choice(['male', 'female']) for _ in range(sample.shape[0])]
        X,y = generateFeatures(copy.deepcopy(sample))
        labels = y.to_list()

        for i, row, sex, label in zip(range(1000,1100), X.iterrows(), sample.Sex.to_list(), labels):
            # print(i, row, sex, label)
            input = row[1].to_list()
            prediction = sendInferenceRequest(input)
            print("storing in DB")
            insertStatement = """ insert into credit(ts, id, sex, prediction, label)
                                    values(%s, %s, %s, %s, %s); """
            cur.execute(insertStatement, (datetime.now(), i, sex, prediction, label))
        # close the communication with the PostgreSQL
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")

def conceptShift():
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

        cur.execute("delete from credit where id>999")

        df_credit = pd.read_csv("./dataset/german_credit_data.csv",index_col=0)
        sample = df_credit.sample(n=200, random_state=102)
        sample['Risk'] = np.where(sample.Housing == 'own',[ random.choice(['good', 'bad']) for _ in range(sample.shape[0])], sample.Risk)
        X,y = generateFeatures(copy.deepcopy(sample))
        labels = y.to_list()

        for i, row, sex, label in zip(range(1000,1200), X.iterrows(), sample.Sex.to_list(), labels):
            # print(i, row, sex, label)
            input = row[1].to_list()
            prediction = sendInferenceRequest(input)
            print("storing in DB")
            insertStatement = """ insert into credit(ts, id, sex, prediction, label)
                                    values(%s, %s, %s, %s, %s); """
            cur.execute(insertStatement, (datetime.now(), i, sex, prediction, label))
        # close the communication with the PostgreSQL
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")

@click.command()
def initial():
    storeInitial()

@click.command()
def covariate_shift():
    shiftSex()

@click.command()
def concept_shift():
    conceptShift()

@click.group()
def main():
    pass

main.add_command(initial)
main.add_command(covariate_shift)
main.add_command(concept_shift)

if __name__=="__main__":
    random.seed(42)
    main()