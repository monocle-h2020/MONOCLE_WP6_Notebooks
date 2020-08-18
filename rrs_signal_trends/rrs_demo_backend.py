#!/usr/bin/python3
"""
Read in preprocessed data.
"""

import os
import sqlite3

sd = os.path.dirname(os.path.realpath(__file__))
dbpath = os.path.join(sd, 'rflex_2015_reanalysis_20200601b.db')


def connect_db(file):
    """Connect to the sqlite3 database file
    :param db_dict: database information dictionary
    :type db_dict: dictionary
    :raises Exception: Exception
    :return: conn, cur
    :rtype: sqlite3 object, sqlite3 cursor
    """
    if not os.path.exists(file):
        print("Cannot find database")
        raise IOError
    try:
        # connect to db file
        conn = sqlite3.connect(file)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
    except Exception as err:
        msg = "Error connecting to database: \n {0}".format(err)
        raise Exception(msg)
    return conn, cur


def query_db(file, degradation_factor_min=1.0, degradation_factor_max=1.0,
             sunzenithmax=60.0, c3fitmin=0.9,
             Ed400min=100, lsed400max=1.0):
    conn, cur = connect_db(file)
    sql = """
          SELECT * FROM rad_fits 
          WHERE degfactor >= ?
          AND degfactor <= ?
          AND sunzenith <= ?
          AND Ed3C_slope >= ?
          AND Ed400 >= ?
          AND Ls_Ed_400 <= ?
          """
    sql_vars = [degradation_factor_min, degradation_factor_max,
                sunzenithmax, c3fitmin, Ed400min, lsed400max]
    cur.execute(sql, sql_vars)
    results = cur.fetchall()
    conn.close()
    return results






