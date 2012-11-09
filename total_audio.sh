#!/bin/bash
sqlite3 db.sqlite "select sum(endtime-starttime) from samples;"

