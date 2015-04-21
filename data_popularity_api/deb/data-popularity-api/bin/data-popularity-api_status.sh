#!/bin/sh
RUNNING=$(supervisorctl status data-popularity-api | grep RUNNING | wc -l)

if [ "$RUNNING" == "1" ]; then
	echo "0;OK"
else
	echo "2;Failed"
fi
