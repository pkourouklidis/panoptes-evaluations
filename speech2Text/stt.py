from configparser import ConfigParser
import psycopg2
from datetime import datetime
from util import generateRows, readFile, sendInfernceRequest, addNoise

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
                WHERE tablename  = 'stt'
            );
        """
        cur.execute(query)
        if not cur.fetchone()[0]:
            print("creating table")
            query = """
                CREATE TABLE stt(
                    id bigint PRIMARY KEY,
                    ts timestamp without time zone,
                    input real[],
                    prediction text,
                    label text
                )
            """
            cur.execute(query)

        for row in generateRows(0,100):
            print("sample " + str(row.Index))
            data = readFile(row.path)
            print("generating prediction")
            prediction = sendInfernceRequest(data)
            print("storing in DB")
            insertStatement = """ insert into stt(ts, id, input, prediction, label)
                                  values(%s, %s, %s, %s, %s); """
            cur.execute(insertStatement, (datetime.now(), row.Index, data, prediction, row.sentence))
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

def storeNoisy():
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

        cur.execute("delete from stt where id>999")

        for row in generateRows(200,300):
            print("sample " + str(row.Index))
            data = readFile(row.path)
            data = addNoise(input)
            data = data.tolist()
            print("generating prediction")
            prediction = sendInfernceRequest(data)
            print("storing in DB")
            insertStatement = """ insert into stt(ts, id, input, prediction, label)
                                  values(%s, %s, %s, %s, %s); """
            cur.execute(insertStatement, (datetime.now(), row.Index+1000, data, prediction, row.sentence))
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

@click.command()
def initial():
    storeInitial()

@click.command()
def noisy():
    storeNoisy()

@click.group()
def main():
    pass

main.add_command(initial)
main.add_command(noisy)

if __name__=='__main__':
    main()