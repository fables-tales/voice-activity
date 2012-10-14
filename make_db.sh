#!/bin/bash
sqlite3 db.sqlite "create table if not exists 'samples' (sourceFile varchar(300), startTime FLOAT, endTime FLOAT, voice BOOLEAN, keyboard BOOLEAN);"
